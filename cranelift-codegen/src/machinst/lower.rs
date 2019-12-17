//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

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
pub type LowerAction<Op, Arg> =
    for<'a> fn(ctx: &mut MachInstLowerCtx<'a, Op, Arg>, inst: Inst) -> bool;

/// Dummy stub.
pub struct MachInstLowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    _phantom0: PhantomData<&'a ()>,
    _phantom1: PhantomData<Op>,
    _phantom2: PhantomData<Arg>,
}

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
        let mut pat = pattern_pool.build();
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

/*
 * TODO: LowerCtx:
 * - get src insts from an inst's op
 * - get src/dst regs from an inst
 */

/*

/// A block of machine instructions with virtual registers.
pub struct MachInstBlock<Op: MachInstOp, Arg: MachInstArg> {
    insts: Vec<MachInst<Op, Arg>>,
}

/// A carrier struct for context related to lowering a function into machine instructions.
pub struct MachInstLowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    func: &'a Function,
    ebb: Ebb,
    lower_table: &'a LowerTable<Op, Arg>,
    reg_ctr: &'a mut MachRegCounter,
    constraints: &'a mut MachRegConstraints,
    num_uses: &'a NumUses,
    rev_insts: Vec<MachInst<Op, Arg>>,
    vregs: SecondaryMap<Value, MachReg>,
    unused: SeconaryMap<Inst, bool>,
}

// Basic strategy:
//
// Given a function and given a NumUses analysis result, for each EBB:
//
// - Initialize vreg map:
//   - Iterate through the EBB's parameters, assigning vregs to each.
//   - Iterate through the instructions, allocating vregs for each result of each instruction.
//   - (we assume that the instructions have already been legalized so that every value
//      can fit in a single register of some class.)
//
// - Iterate through the instructions backward, examining each instruction in turn.
//
// TODO: register constraints and move-insertion points for regalloc.

impl<'a, Op: MachInstOp, Arg: MachInstArg> MachInstLowerCtx<'a, Op, Arg> {
    fn new(
        func: &'a Function,
        ebb: Ebb,
        lower_table: &'a LowerTable<Op, Arg>,
        num_uses: &'a NumUses,
        reg_ctr: &'a mut MachRegCounter,
        constraints: &'a mut MachRegConstraints,
    ) -> MachInstLowerCtx<'a, Op, Arg> {
        MachInstLowerCtx {
            func,
            ebb,
            lower_table,
            reg_ctr,
            constraints,
            num_uses,
            rev_insts: vec![],
            vregs: SecondaryMap::with_default(MachReg::Virtual(0)),
            unused: SecondaryMap::with_default(false),
        }
    }

    fn alloc_vreg_for_value(&mut self, v: Value) -> MachReg {
        let r = self.reg_ctr.alloc();
        let ty = self.func.dfg.value_type(v);
        let rc = Arg::regclass_for_type(ty);
        self.constraints.add(&r, MachRegConstraint::from_class(rc));
        self.vregs[v] = r.clone();
        r
    }

    fn alloc_vregs_for_func(&mut self) {
        for param in self.func.dfg.ebb_params(self.ebb) {
            self.alloc_vreg_for_value(*param);
        }
        for inst in self.func.layout.ebb_insts(self.ebb) {
            let data = &self.func.dfg[inst];
            for result in self.func.dfg.inst_results(inst) {
                self.alloc_vreg_for_value(*result);
            }
        }
    }

    fn take_insts(&mut self) -> Vec<MachInst<Op, Arg>> {
        let mut v = mem::replace(&mut self.rev_insts, vec![]);
        v.reverse();
        v
    }

    fn lower_inst(&mut self, ins: Inst) {
        if self.unused[ins] {
            return;
        }

        // Build a tree greedily, including single-use instructions.
        let mut pool = PatternPrefixPool::new();
        let mut pattern = pool.build();
        pattern = self.build_inst_tree(pattern, ins);
    }

    fn build_inst_tree<'a>(&self, mut pattern: PatternPrefixBuilder<'a>, ins: Inst) -> PatternPrefixBuilder<'a> {
        // Check args to determine which, if any, can be included.

    }

    // ----------------- API for use by lowering actions -------------------

    /// Emit a machine instruction. The machine instructions will eventually be ordered in reverse
    /// of the calls to this function.
    pub fn emit(&mut self, inst: MachInst<Op, Arg>) {
        self.rev_insts.push(inst);
    }

    /// Look up the register for a given value.
    pub fn reg(&self, v: Value) -> MachReg {
        self.vregs[v].clone()
    }

    /// Constrain a virtual register to a register class.
    pub fn constrain_rc(&mut self, r: &MachReg, rc: RegClass) {
        self.constraints.add(r, MachRegConstraint::from_class(rc));
    }

    /// Constrain a virtual register to a fixed physical register.
    pub fn constrain_fixed(&mut self, r: &MachReg, ru: RegUnit) {
        self.constraints.add(r, MachRegConstraint::from_fixed(ru));
    }
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInstBlock<Op, Arg> {
    /// Lower a function to a list of blocks of machine instructions.
    pub fn lower(
        func: &Function,
        num_uses: &NumUses,
        reg_ctr: &mut MachRegCounter,
        constraints: &mut MachRegConstraints,
    ) -> Vec<MachInstBlock<Op, Arg>> {
        let mut ctx = MachInstLowerCtx::new(func, ebb, num_uses, reg_ctr, constraints);
        ctx.alloc_vregs_for_func();
        let mut ret = vec![];
        for ebb in func.layout.ebbs() {
            for ins in func.layout.ebb_insts(ebb).rev() {
                ctx.lower_inst(ins);
            }
            let block = MachInstBlock {
                insts: ctx.take_insts(),
            };
            ret.push(block);
        }
        ret
    }
}

*/

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
