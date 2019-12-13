//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::cursor::FuncCursor;
use crate::entity::SecondaryMap;
use crate::ir::{Ebb, Function, Inst, Opcode, Value};
use crate::machinst::*;
use crate::machinst::{MachInst, MachInstArg, MachInstOp};
use crate::num_uses::NumUses;

use alloc::vec::Vec;

/// A block of machine instructions with virtual registers.
struct MachInstBlock<Op: MachInstOp, Arg: MachInstArg> {
    insts: Vec<MachInst<Op, Arg>>,
}

struct MachInstLowerCtx<'a, Op: MachInstOp, Arg: MachInstArg> {
    func: &'a Function,
    ebb: Ebb,
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
//   - Recursively:
//     - Look up
//
impl<'a, Op: MachInstOp, Arg: MachInstArg> MachInstLowerCtx<'a, Op, Arg> {
    fn new(
        func: &'a Function,
        ebb: Ebb,
        num_uses: &'a NumUses,
        reg_ctr: &'a mut MachRegCounter,
        constraints: &'a mut MachRegConstraints,
    ) -> MachInstLowerCtx<'a, Op, Arg> {
        MachInstLowerCtx {
            func,
            ebb,
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

    fn alloc_vregs(&mut self) {
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

    fn to_insts(self) -> Vec<MachInst<Op, Arg>> {
        let mut v = self.rev_insts;
        v.reverse();
        v
    }

    fn lower_inst(&mut self, ins: Inst) {
        // TODO.
    }
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInstBlock<Op, Arg> {
    pub fn lower(
        func: &Function,
        ebb: Ebb,
        num_uses: &NumUses,
        reg_ctr: &mut MachRegCounter,
        constraints: &mut MachRegConstraints,
    ) -> MachInstBlock<Op, Arg> {
        let mut ctx = MachInstLowerCtx::new(func, ebb, num_uses, reg_ctr, constraints);
        ctx.alloc_vregs();
        for ins in func.layout.ebb_insts(ebb).rev() {
            ctx.lower_inst(ins);
        }
        MachInstBlock {
            insts: ctx.to_insts(),
        }
    }
}
