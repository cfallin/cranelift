//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::cursor::FuncCursor;
use crate::entity::SecondaryMap;
use crate::ir::{Ebb, Function, Inst, Opcode, Value, InstructionData};
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
pub type LowerAction<Op, Arg> = for<'a> fn(
    ctx: &mut MachInstLowerCtx<'a, Op, Arg>,
    insts: &[&'a InstructionData],
    regs: &[MachReg],
    results: &[MachReg],
) -> bool;

/// Dummy stub.
pub struct MachInstLowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    _phantom0: PhantomData<&'a ()>,
    _phantom1: PhantomData<Op>,
    _phantom2: PhantomData<Arg>,
}

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
