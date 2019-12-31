//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::machinst::{MachInst
use crate::ir::{Inst, Function, InstructionData};
use crate::isa::registers::{RegClass, RegUnit};
use crate::ir::entity::SecondaryMap;
use crate::num_uses::NumUses;

/// A context that machine-specific lowering code can use to emit lowered instructions.
pub trait LowerCtx<I: MachInst> {
    /// Get the instdata for a given IR instruction.
    fn data(&self, ir_inst: Inst) -> &InstructionData;
    /// Emit a machine instruction.
    fn emit(&mut self, mach_inst: I);
    /// Reduce the use-count of an IR instruction. Use this when, e.g., isel incorporates the
    /// computation of an input instruction directly.
    fn dec_use(&self, ir_inst: Inst);
    /// Get the `idx`th input to the given IR instruction as a virtual register.
    fn input(&self, ir_inst: Inst, idx: usize) -> MachReg;
    /// Get the `idx`th output of the given IR instruction as a virtual register.
    fn output(&self, ir_inst: Inst, idx: usize) -> MachReg;
    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass) -> MachReg;
}

/// A backend's lowering logic, to be driven by the machine-independent portion of instruction
/// lowering.
pub trait LowerBackend {
    /// The machine instruction type.
    pub type MInst: MachInst;

    /// Lower a single instruction. Instructions are lowered in reverse order.
    fn lower(&mut self, ctx: &mut dyn LowerCtx<Inst>, inst: Inst);
}

/// Machine-independent lowering driver / machine-instruction container. Maintains a correspondence
/// from original ir::Inst to MachInsts.
pub struct Lower<'a, I: MachInst> {
    // The function to lower.
    f: &'a Function,

    // Lowered machine instructions. In arbitrary order; map from original IR program order using
    // `inst_indices` below.
    insts: Vec<I>,

    // Number of active uses (minus `dec_use()` calls by backend) of each instruction.
    num_uses: SecondaryMap<Inst, u32>,

    // Range of indices in `insts` corresponding to a given Cranelift instruction:
    inst_indices: SecondaryMap<Inst, (u32, u32)>,

    // Mapping from `Value` (SSA value in IR) to virtual register.
    value_regs: SecondaryMap<Value, MachReg>,

    // Next virtual register number to allocate.
    next_vreg: usize,

    // Current IR instruction which we are lowering.
    cur_inst: Option<Inst>,
}

fn alloc_vreg(reg: &mut MachReg, next_vreg: &mut usize) {
    match reg {
        &mut MachReg::Undefined => {
            let v = *next_vreg;
            *next_vreg += 1;
            *reg = MachReg::Virtual(v);
        }
        _ => {}
    }
}

impl<'a, I: MachInst> Lower<'a, I> {
    /// Prepare a new lowering context for the given IR function.
    pub fn new(f: &'a Function) -> Lower<'a, I> {
        let num_uses = NumUses::compute(f).take_uses();

        let mut next_vreg = 0;
        let mut value_regs = SecondaryMap::with_default(MachReg::Undefined);
        for ebb in f.layout.ebbs() {
            for param in f.dfg.ebb_params(ebb) {
                alloc_vreg(&mut value_regs[param], &mut next_vreg);
            }
            for inst in f.layout.ebb_insts(ebb) {
                for arg in f.dfg.inst_args(inst) {
                    alloc_reg(&mut value_regs[arg], &mut next_vreg);
                }
                for result in f.dfg.inst_results(inst) {
                    alloc_reg(&mut value_regs[result], &mut next_vreg);
                }
            }
        }

        Lower {
            f,
            insts: vec![],
            num_uses,
            inst_indices: SecondaryMap::with_default((0, 0)),
            value_regs,
            next_vreg, 
            cur_inst: None,
        }
    }

    /// Lower the function.
    pub fn lower(&mut self, backend: &mut dyn LowerBackend<MInst=I>) {
        // Work backward (postorder for EBBs, reverse through each EBB), skipping insns with
        // zero uses.
        let ebbs: SmallVec<[Ebb; 16]> = self.f.layout.ebbs().collect();
        for ebb in ebbs.into_iter().rev() {
            for inst in self.f.layout.ebb_insts(ebb).rev() {
                if self.num_uses[inst] > 0 {
                    self.start_inst(inst);
                    backend.lower(self, inst);
                    self.end_inst();
                }
            }
        }
    }

    fn start_inst(&mut self, inst: ir::Inst) {
        self.cur_inst = Some(inst);
        let l = self.insts.len();
        self.inst_indices[inst]  = (l, l);
    }

    fn end_inst(&mut self) {
        self.cur_inst = None;
    }
}

impl<'a, I: MachInst> LowerCtx<I> for Lower<'a, I> {
    /// Get the instdata for a given IR instruction.
    fn data(&self, ir_inst: Inst) -> &InstructionData {
        self.f.dfg[ir_inst]
    }

    /// Emit a machine instruction.
    fn emit(&mut self, mach_inst: I) {
        let cur_inst = self.cur_inst.clone().unwrap();
        self.insts.push(mach_inst);
        // Bump the end of the range.
        self.inst_indices[cur_inst].1 = self.insts.len();
    }

    /// Reduce the use-count of an IR instruction. Use this when, e.g., isel incorporates the
    /// computation of an input instruction directly.
    fn dec_use(&self, ir_inst: Inst) {
        assert!(self.num_uses[ir_inst] > 0);
        self.num_uses[ir_inst] -= 1;
    }

    /// Get the `idx`th input to the given IR instruction as a virtual register.
    fn input(&self, ir_inst: Inst, idx: usize) -> MachReg {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        self.value_regs[val]
    }

    /// Get the `idx`th output of the given IR instruction as a virtual register.
    fn output(&self, ir_inst: Inst, idx: usize) -> MachReg {
        let val = self.f.dfg.inst_results(ir_inst)[idx];
        self.value_regs[val]
    }

    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass) -> MachReg {
        let v = self.next_vreg;
        self.next_vreg += 1;
        MachReg::Virtual(v)
    }
}
