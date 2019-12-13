//! This module implements lowering (instruction selection) from Cranelift IR to Arm64 instructions
//! with virtual registers. This is *almost* the final machine code, except for register
//! allocation.

use alloc::vec::Vec;

use crate::cursor::FuncCursor;
use crate::ir::{Ebb, Function, Inst, Opcode};
use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::regclass_for_type;
use crate::isa::machinst::*;
use crate::num_uses::NumUses;

/// A block of Arm64 instructions with virtual registers.
struct Arm64Block {
    insts: Vec<Arm64Inst>,
}

struct Arm64LowerCtx<'a> {
    func: &'a Function,
    ebb: Ebb,
    reg_ctr: &'a mut MachRegCounter,
    constraints: &'a mut MachRegConstraints,
    num_uses: &'a NumUses,
    rev_insts: Vec<Arm64Inst>,
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
impl<'a> Arm64LowerCtx<'a> {
    fn new(
        func: &'a Function,
        ebb: Ebb,
        num_uses: &'a NumUses,
        reg_ctr: &'a mut MachRegCounter,
        constraints: &'a MachRegConstraints,
    ) -> Arm64LowerCtx<'a> {
        Arm64LowerCtx {
            func,
            ebb,
            reg_ctr,
            constraints,
            num_uses,
            rev_insts: vec![],
            vregs: SecondaryMap::new(),
        }
    }

    fn alloc_vreg_for_value(&mut self, v: Value) -> MachReg {
        let r = self.reg_ctr.alloc();
        let ty = self.func.dfg.value_type(v);
        let rc = regclass_for_type(ty);
        self.constraints.add(&r, MachRegConstraint::from_class(rc));
        self.vregs[v] = r.clone();
        r
    }

    fn alloc_vregs(&mut self) {
        for param in self.func.dfg.ebb_params(self.ebb) {
            self.alloc_vreg_for_value(param);
        }
        let mut curs = FuncCursor::new(self.func).at_top(ebb);
        while let Some(inst) = curs.next_inst() {
            let data = &self.func.dfg[inst];
            for result in self.func.dfg.inst_results(inst) {
                self.alloc_vreg_for_value(result);
            }
        }
    }

    fn to_insts(self) -> Vec<Arm64Inst> {
        let v = self.rev_insts;
        v.reverse();
        v
    }

    fn lower_inst(&mut self, ins: Inst) {
        // TODO.
    }
}

impl Arm64Block {
    pub fn lower(
        func: &Function,
        num_uses: &NumUses,
        ebb: Ebb,
        reg_ctr: &mut MachRegCounter,
        constraints: &mut MachRegConstraints,
    ) -> Arm64Block {
        let mut curs = FuncCursor::new(func).at_bottom(ebb);
        let mut insts = vec![];
        let mut ctx = Arm64LowerCtx::new(func, num_uses, reg_ctr, constraints);
        ctx.alloc_vregs();
        while let Some(ins) = curs.prev_inst() {
            ctx.lower_inst(ins);
        }
        Arm64Block {
            insts: ctx.to_insts(),
        }
    }
}
