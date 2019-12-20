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
///
/// This is a trait so that lowering actions can be evaluated with arbitrary contexts, for example
/// to extract symbolic representations of the lowered code or to assist with statically generating
/// specialized/optimized backend.
pub trait LowerCtx<Op: MachInstOp, Arg: MachInstArg> {
    /// Emit a machine instruction. The machine instructions will eventually be ordered in reverse
    /// of the calls to this function.
    fn emit(&mut self, inst: MachInst<Op, Arg>);

    /// Get the Value corresponding to the `idx`-th input of `inst`. Can access both
    /// DataFlowGraph-tracked inputs and "extra" inputs.
    fn input(&self, inst: Inst, idx: usize) -> Value;

    /// Get the Value corresponding to the `idx`-th output of `inst`.
    fn output(&self, inst: Inst, idx: usize) -> Value;

    /// Get the type of the (first) result of the given instruction.
    fn ty(&self, inst: Inst) -> Type;

    /// Get the instruction that produces the `idx`-th input of `inst`.
    fn input_inst(&self, inst: Inst, idx: usize) -> Inst;

    /// Get the instruction data for a given instruction.
    fn inst(&self, inst: Inst) -> &InstructionData;

    /// Add an extra input argument to a given IR inst. If the lowered instructions for a given IR
    /// inst use a Value from another point in the program, then it must be added as an extra arg
    /// to that IR inst. This returns an index that can be passed to `input`.
    fn add_input(&mut self, inst: Inst, value: Value) -> usize;

    /// Mark this instruction as "unused". This means that its value does not need to be generated
    /// (machine instruction lowering can skip it), usually because it was matchedas part of
    /// another IR instruction's pattern during instruction selection.
    fn mark_unused(&mut self, inst: Inst);

    /// Fix an existing Value to a given fixed register.
    fn fix_reg(&mut self, value: Value, ru: RegUnit);

    // TODO: temporaries? So far, we haven't needed to allocate temporaries within the MachInsts
    // for a given IR inst.
}

/// The canonical implementation of the lowering context, used to emit instructions directly to the
/// machine-instruction portion of the function.
pub struct LowerCtxImpl<'a, Op: MachInstOp, Arg: MachInstArg> {
    func: &'a mut Function,
    machinsts: &'a mut MachInstsImpl<Op, Arg>,
    constraints: &'a mut MachRegConstraints,
    cur_inst: Option<Inst>,
    unused: SecondaryMap<Inst, bool>,
}

impl<'a, Op: MachInstOp, Arg: MachInstArg> LowerCtxImpl<'a, Op, Arg> {
    fn new(
        func: &'a Function,
        machinsts: &'a mut MachInstsImpl<Op, Arg>,
        constraints: &'a mut MachRegConstraints,
    ) -> LowerCtxImpl<'a, Op, Arg> {
        LowerCtxImpl {
            func,
            machinsts,
            constraints,
            cur_inst: None,
            unused: SecondaryMap::with_default(false),
        }
    }

    /// Called from lowering traversal: start of processing the givenIR inst.
    pub fn begin_inst(&mut self, inst: Inst) {
        self.cur_inst = Some(inst);
    }

    /// Called from lowering traversal: end of processing the current IR inst.
    pub fn end_inst(&mut self) {
        self.cur_inst = None;
    }

    fn cur(&self) -> Inst {
        assert!(self.cur_inst.is_some());
        self.cur_inst.cloned().unwrap()
    }

    /// Is the given instruction marked as unused?
    pub fn is_unused(&self, inst: Inst) -> bool {
        self.unused[inst]
    }
}

impl<'a, Op: MachInstOp, Arg: MachInstArg> LowerCtx<Op, Arg> for LowerCtxImpl<'a, Op, Arg> {
    /// Emit a machine instruction. The machine instructions will eventually be ordered in reverse
    /// of the calls to this function.
    fn emit(&mut self, inst: MachInst<Op, Arg>) {
        self.machinsts.add_inst(self.cur(), inst);
    }

    /// Get the Value corresponding to the `idx`-th input of `inst`.
    fn input(&self, inst: Inst, idx: usize) -> Value {
        self.machinsts.get_arg(&self.func.dfg, inst, idx)
    }

    /// Get the Value corresponding to the `idx`-th output of `inst`.
    fn output(&self, inst: Inst, idx: usize) -> Value {
        self.func.dfg.inst_results(inst)[idx]
    }

    /// Get the type of the (first) result of the given instruction.
    fn ty(&self, inst: Inst) -> Type {
        let val = self.func.dfg.inst_results(inst)[0];
        self.func.dfg.value_type(val)
    }

    /// Get the instruction that produces the `idx`-th input of `inst`.
    fn input_inst(&self, inst: Inst, idx: usize) -> Inst {
        let val = self.input(inst, idx);
        match self.func.dfg.value_def(val) {
            ValueDef::Result(def_inst, _) => def_inst,
            _ => panic!("input_inst() on input not defined by another instruction"),
        }
    }

    /// Get the instruction data for a given instruction.
    fn inst(&self, inst: Inst) -> &InstructionData {
        &self.func.dfg[inst]
    }

    /// Add an extra input to the given IR inst, corresponding to an use of this input by the
    /// lowered machine instructions.
    fn add_input(&mut self, inst: Inst, value: Value) -> usize {
        assert!(inst == self.cur());
        self.func.dfg.inst_results(inst).len() + self.machinsts.add_extra_arg(inst, value)
    }

    /// Mark this instruction as "unused". This means that its value does not need to be generated
    /// (machine instruction lowering can skip it), usually because it was matchedas part of
    /// another IR instruction's pattern during instruction selection.
    fn mark_unused(&mut self, inst: Inst) {
        self.unused[inst] = true;
    }

    /// Fix a virtual register to a particular physical register.
    fn fix_reg(&mut self, value: Value, ru: RegUnit) {
        self.constraints
            .add(value, MachRegConstraint::from_fixed(ru));
    }
}

/// Lower a function to virtual-reg machine insts.
pub fn lower(func: &mut Function, lower_table: &LowerTable<Op, Arg>) {
    // Create the MachInsts instance.
    let mut machinsts = MachInstsImpl::new();

    let mut reg_constraints = MachRegConstraints::new();

    // Set up the context passed to lowering actions.
    let mut ctx = LowerCtxImpl::new(func, &mut machinsts, &mut reg_constraints);

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

        // For each instruction, in reverse order, extract the tree and lower it.
        for inst in func.layout.ebb_insts(ebb).rev() {
            if !ctx.is_unused(inst) {
                let ckpt = prefix_pool.checkpoint();
                let tree = bte.get_tree(&mut prefix_pool, inst);

                ctx.begin_inst(inst);

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
                }

                ctx.end_inst();

                if !lowered {
                    panic!(
                        "Unable to lower instruction {:?}: {:?}",
                        inst, func.dfg[inst]
                    );
                }

                prefix_pool.rewind(ckpt);
            }
        }
    }

    func.machinsts = Some(Box::new(machinsts));
    func.reg_constraints = Some(reg_constraints);
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
