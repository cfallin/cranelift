//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

/*
 * Algorithm:
 *
 * - After collecting a set of tree-prefixes and actions:
 *   - Sort the prefixes by first op.
 *   - Build the prefixes into a single prefix table.
 *
 * - When lowering a block:
 *   - For each value that we want to generate:
 *     - Construct an opcode tree. Here we slurp all single-use, side-effect-free
 *       values into the tree greedily.
 *     - Find the prefixes starting with the root opcode, and try to match each in turn.
 *     - For each that matches, invoke the lowering action. The first lowering action
 *       that returns `true` terminates the codegen for this value.
 *       - The lowering action is given the instructions and their argument registers
 *         in the order they are mentioned in the tree prefix, and the result regs
 *         of the root instruction.
 *       - The lowering action, in turn, can invoke ctx.emit(machinst) to emit a
 *         machine instruction and/or ctx.unused(inst) to note that inst is now unused
 *         and need not be generated.
 */

use crate::cursor::FuncCursor;
use crate::entity::SecondaryMap;
use crate::ir::{Ebb, Function, Inst, InstructionData, Opcode, Value, ValueDef};
use crate::machinst::pattern::*;
use crate::machinst::pattern_prefix::*;
use crate::machinst::*;
use crate::num_uses::NumUses;
use crate::HashSet;
use std::marker::PhantomData;
use std::mem;

use alloc::vec::Vec;

/// An action to perform when matching a tree of ops. Returns `true` if codegen was successful.
/// Otherwise, another pattern/action should be used instead.
pub type LowerAction<Op, Arg> = for<'a> fn(ctx: &mut LowerCtx<'a, Op, Arg>, inst: Inst) -> bool;

/// Greedily extracts opcode trees from a block.
pub struct BlockTreeExtractor<'a> {
    func: &'a Function,
    num_uses: &'a NumUses,
    ebb: Ebb,
}

impl<'a> BlockTreeExtractor<'a> {
    /// Create a new tree-extractor for an Ebb.
    pub fn new(func: &'a Function, num_uses: &'a NumUses, ebb: Ebb) -> BlockTreeExtractor<'a> {
        BlockTreeExtractor {
            func,
            num_uses,
            ebb,
        }
    }

    fn arg_included(&self, inst: Inst, v: Value) -> Option<Inst> {
        let v = self.func.dfg.resolve_aliases(v);
        match self.func.dfg.value_def(v) {
            ValueDef::Result(def_inst, _) => {
                if self.num_uses.use_count(def_inst) == 1
                    && self.func.dfg.inst_results(def_inst).len() == 1
                {
                    Some(def_inst)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get the PatternPrefix opcode tree rooted at the given instruction.
    pub fn get_tree<'pool>(
        &self,
        pattern_pool: &'pool mut PatternPrefixPool,
        inst: Inst,
    ) -> PatternPrefixRange {
        let pat = pattern_pool.build();
        self.get_tree_internal(pat, inst).build()
    }

    fn get_tree_internal<'pool>(
        &self,
        mut pat: PatternPrefixBuilder<'pool>,
        inst: Inst,
    ) -> PatternPrefixBuilder<'pool> {
        let sub_args: SmallVec<[Option<Inst>; 4]> = self
            .func
            .dfg
            .inst_args(inst)
            .iter()
            .map(|arg| self.arg_included(inst, *arg))
            .collect();

        let op = self.func.dfg[inst].opcode();
        if sub_args.iter().all(|a| a.is_none()) {
            pat.opcode(op)
        } else {
            pat = pat.opcode_with_args(op);
            for sub_arg in sub_args.into_iter() {
                if let Some(def_inst) = sub_arg {
                    pat = self.get_tree_internal(pat, def_inst);
                } else {
                    pat = pat.any();
                }
            }
            pat.args_end()
        }
    }
}

/// Context passed to lowering actions, containing in-progress emitted instructions and a record of
/// which instructions are already generated, as well as a mapping from Values to machine
/// registers.
pub struct LowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    func: &'a Function,
    rev_insts: Vec<MachInst<Op, Arg>>,
    vregs: SecondaryMap<Value, MachReg>,
    unused: SecondaryMap<Inst, bool>,
    constraints: &'a mut MachRegConstraints,
    reg_ctr: &'a mut MachRegCounter,
}

impl<'a, Op: MachInstOp, Arg: MachInstArg> LowerCtx<'a, Op, Arg> {
    fn new(
        func: &'a Function,
        vregs: SecondaryMap<Value, MachReg>,
        constraints: &'a mut MachRegConstraints,
        reg_ctr: &'a mut MachRegCounter,
    ) -> LowerCtx<'a, Op, Arg> {
        LowerCtx {
            func,
            rev_insts: vec![],
            vregs,
            unused: SecondaryMap::with_default(false),
            constraints,
            reg_ctr,
        }
    }

    /// Emit a machine instruction. The machine instructions will eventually be ordered in reverse
    /// of the calls to this function.
    pub fn emit(&mut self, inst: MachInst<Op, Arg>) {
        self.rev_insts.push(inst);
    }

    /// Look up the register for a given value.
    pub fn reg(&self, v: Value) -> MachReg {
        self.vregs[v].clone()
    }

    /// Get the MachReg corresponding to the `idx`-th input of `inst`.
    pub fn input(&self, inst: Inst, idx: usize) -> MachReg {
        let val = self.func.dfg.inst_args(inst)[idx];
        self.reg(val)
    }

    /// Get the MachReg corresponding to the `idx`-th output of `inst`.
    pub fn output(&self, inst: Inst, idx: usize) -> MachReg {
        let val = self.func.dfg.inst_results(inst)[idx];
        self.reg(val)
    }

    /// Get the instruction that produces the `idx`-th input of `inst`.
    pub fn input_inst(&self, inst: Inst, idx: usize) -> Inst {
        let val = self.func.dfg.inst_args(inst)[idx];
        match self.func.dfg.value_def(val) {
            ValueDef::Result(def_inst, _) => def_inst,
            _ => panic!("input_inst() on input not defined by another instruction"),
        }
    }

    /// Get the instruction data for a given instruction.
    pub fn inst(&self, inst: Inst) -> &InstructionData {
        &self.func.dfg[inst]
    }

    /// Mark this instruction as "unused". This means that its value does not need to be generated
    /// (machine instruction lowering can skip it), usually because it was matchedas part of
    /// another IR instruction's pattern during instruction selection.
    pub fn unused(&mut self, inst: Inst) {
        self.unused[inst] = true;
    }

    /// Has this instruction been marked as unused (no need to generate its value)?
    pub fn is_unused(&self, inst: Inst) -> bool {
        self.unused[inst]
    }

    /// Allocate a new machine register.
    pub fn alloc_reg(&mut self) -> MachReg {
        self.reg_ctr.alloc()
    }

    /// Fix a virtual register to a particular physical register.
    pub fn fix_reg(&mut self, reg: &MachReg, ru: RegUnit) {
        self.constraints.add(reg, MachRegConstraint::from_fixed(ru));
    }

    /// Allocate a new machine register and constrain it.
    pub fn fixed(&mut self, ru: RegUnit) -> MachReg {
        let r = self.alloc_reg();
        self.fix_reg(&r, ru);
        r
    }
}

/// Result of lowering a function to machine code.
#[derive(Clone, Debug)]
pub struct LowerResult<Op: MachInstOp, Arg: MachInstArg> {
    constraints: MachRegConstraints,
    insts: Vec<MachInst<Op, Arg>>,
}

impl<Op: MachInstOp, Arg: MachInstArg> LowerResult<Op, Arg> {
    /// Lower a function to virtual-reg machine insts.
    pub fn lower(func: &Function, lower_table: &LowerTable<Op, Arg>) -> LowerResult<Op, Arg> {
        // Assign virtual register numbers.
        let mut reg_ctr = MachRegCounter::new();
        let mut constraints = MachRegConstraints::new();
        let mut vregs = SecondaryMap::with_default(MachReg::Virtual(0));

        let mut values: SmallVec<[Value; 64]> = SmallVec::new();
        for ebb in func.layout.ebbs() {
            for param in func.dfg.ebb_params(ebb) {
                values.push(*param);
            }
            for inst in func.layout.ebb_insts(ebb) {
                for result in func.dfg.inst_results(inst) {
                    values.push(*result);
                }
            }
        }
        for value in values.into_iter() {
            let reg = reg_ctr.alloc();
            let ty = func.dfg.value_type(value);
            let rc = Arg::regclass_for_type(ty);
            constraints.add(&reg, MachRegConstraint::from_class(rc));
            vregs[value] = reg;
        }

        // Set up the context passed to lowering actions.
        let mut ctx = LowerCtx::new(func, vregs, &mut constraints, &mut reg_ctr);

        // Compute the number of uses of each value, for use during greedy tree extraction.
        let num_uses = NumUses::compute(func);

        // Create a pattern-prefix pool for the temporary prefixes extracted from insns.
        let mut prefix_pool = PatternPrefixPool::new();

        // Lower each EBB in turn, in postorder (so that when we reverse the instructions, they are
        // in RPO).
        let ebbs: SmallVec<[Ebb; 16]> = func.layout.ebbs().collect();
        for ebb in ebbs.into_iter().rev() {
            // Create the block tree extractor.
            let bte = BlockTreeExtractor::new(func, &num_uses, ebb);

            // For each instruction, in reverse order, extract the tree.
            for inst in func.layout.ebb_insts(ebb).rev() {
                if !ctx.is_unused(inst) {
                    let ckpt = prefix_pool.checkpoint();
                    let tree = bte.get_tree(&mut prefix_pool, inst);

                    // Look up the entries for the root opcode and try
                    // them, if they match, one at a time.
                    let root_op = prefix_pool.get(&tree).root_op();
                    if let Some(entries) = lower_table.get_entries(root_op) {
                        let mut lowered = false;
                        for entry in entries {
                            let pat = lower_table.pool().get(&entry.prefix);
                            let subject = prefix_pool.get(&tree);
                            if pat.matches(&subject) {
                                let action = entry.action;
                                if action(&mut ctx, inst) {
                                    // Action was successful -- no need to try further patterns.
                                    lowered = true;
                                    break;
                                }
                            }
                        }

                        if !lowered {
                            panic!(
                                "Unable to lower instruction {:?}: {:?}",
                                inst, func.dfg[inst]
                            );
                        }
                    }

                    prefix_pool.rewind(ckpt);
                }
            }
        }

        let insts: Vec<_> = ctx.rev_insts.into_iter().rev().collect();

        LowerResult { constraints, insts }
    }

    /// Return the list of lowered machine instructions.
    pub fn insts(&self) -> &[MachInst<Op, Arg>] {
        &self.insts[..]
    }

    /// Return the register constraints for the virtual registers in the lowered machine
    /// instructions.
    pub fn reg_constraints(&self) -> &MachRegConstraints {
        &self.constraints
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::condcodes::*;
    use crate::ir::types::*;
    use crate::ir::{Function, InstBuilder, ValueDef};
    use crate::num_uses::NumUses;

    #[test]
    fn test_tree_extractor() {
        let mut func = Function::new();
        let ebb0 = func.dfg.make_ebb();
        let arg0 = func.dfg.append_ebb_param(ebb0, I32);
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_ebb(ebb0);

        let v0 = pos.ins().iconst(I32, 1);
        let v1 = pos.ins().iadd(arg0, v0);
        let v2 = pos.ins().isub(arg0, v1);
        let v3 = pos.ins().iadd(v2, v2);

        let ins0 = func.dfg.value_def(v0).unwrap_inst();
        let ins1 = func.dfg.value_def(v1).unwrap_inst();
        let ins2 = func.dfg.value_def(v2).unwrap_inst();
        let ins3 = func.dfg.value_def(v3).unwrap_inst();

        let num_uses = NumUses::compute(&func);
        let bte = BlockTreeExtractor::new(&func, &num_uses, ebb0);

        let mut pool = PatternPrefixPool::new();
        let tree0 = bte.get_tree(&mut pool, ins0);
        let tree1 = bte.get_tree(&mut pool, ins1);
        let tree2 = bte.get_tree(&mut pool, ins2);
        let tree3 = bte.get_tree(&mut pool, ins3);

        assert!(pool.get(&tree0).root_op() == Opcode::Iconst);
        assert!(pool.get(&tree1).root_op() == Opcode::Iadd);
        assert!(pool.get(&tree2).root_op() == Opcode::Isub);
        assert!(pool.get(&tree3).root_op() == Opcode::Iadd);

        let tree0_expected = pool.build().opcode(Opcode::Iconst).build();
        let tree1_expected = pool
            .build()
            .opcode_with_args(Opcode::Iadd)
            .any()
            .opcode(Opcode::Iconst)
            .args_end()
            .build();
        let tree2_expected = pool
            .build()
            .opcode_with_args(Opcode::Isub)
            .any()
            .opcode_with_args(Opcode::Iadd)
            .any()
            .opcode(Opcode::Iconst)
            .args_end()
            .args_end()
            .build();
        let tree3_expected = pool.build().opcode(Opcode::Iadd).build();

        assert!(pool.get(&tree0) == pool.get(&tree0_expected));
        assert!(pool.get(&tree1) == pool.get(&tree1_expected));
        assert!(pool.get(&tree2) == pool.get(&tree2_expected));
        assert!(pool.get(&tree3) == pool.get(&tree3_expected));
    }
}
