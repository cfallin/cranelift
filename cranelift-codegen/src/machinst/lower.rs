//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::cursor::FuncCursor;
use crate::entity::SecondaryMap;
use crate::ir::{Ebb, Function, Inst, Opcode, Value};
use crate::machinst::*;
use crate::machinst::{MachInst, MachInstArg, MachInstOp};
use crate::num_uses::NumUses;
use std::mem;

use alloc::vec::Vec;

/// A block of machine instructions with virtual registers.
pub struct MachInstBlock<Op: MachInstOp, Arg: MachInstArg> {
    insts: Vec<MachInst<Op, Arg>>,
}

/// A carrier struct for context related to lowering a function into machine instructions.
pub struct MachInstLowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    func: &'a Function,
    reg_ctr: &'a mut MachRegCounter,
    constraints: &'a mut MachRegConstraints,
    num_uses: &'a NumUses,
    rev_insts: Vec<MachInst<Op, Arg>>,
    vregs: SecondaryMap<Value, MachReg>,
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
        num_uses: &'a NumUses,
        reg_ctr: &'a mut MachRegCounter,
        constraints: &'a mut MachRegConstraints,
    ) -> MachInstLowerCtx<'a, Op, Arg> {
        MachInstLowerCtx {
            func,
            reg_ctr,
            constraints,
            num_uses,
            rev_insts: vec![],
            vregs: SecondaryMap::with_default(MachReg::Virtual(0)),
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
        for ebb in self.func.layout.ebbs() {
            for param in self.func.dfg.ebb_params(ebb) {
                self.alloc_vreg_for_value(*param);
            }
            for inst in self.func.layout.ebb_insts(ebb) {
                let data = &self.func.dfg[inst];
                for result in self.func.dfg.inst_results(inst) {
                    self.alloc_vreg_for_value(*result);
                }
            }
        }
    }

    fn take_insts(&mut self) -> Vec<MachInst<Op, Arg>> {
        let mut v = mem::replace(&mut self.rev_insts, vec![]);
        v.reverse();
        v
    }

    fn lower_inst(&mut self, ins: Inst) {
        // TODO. Use tables.
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
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInstBlock<Op, Arg> {
    /// Lower a function to a list of blocks of machine instructions.
    pub fn lower(
        func: &Function,
        num_uses: &NumUses,
        reg_ctr: &mut MachRegCounter,
        constraints: &mut MachRegConstraints,
    ) -> Vec<MachInstBlock<Op, Arg>> {
        let mut ctx = MachInstLowerCtx::new(func, num_uses, reg_ctr, constraints);
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
