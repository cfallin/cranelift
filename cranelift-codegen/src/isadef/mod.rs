//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::CodeSink;
use crate::ir::{Opcode, Type};
use crate::isa::RegUnit;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A machine register in a machine instruction. Can be virtual (pre-regalloc) or allocated
/// (post-regalloc).
pub enum MachReg {
    Virtual(usize, MachRegConstraint),
    Allocated(RegUnit),
}

/// A constraint on a virtual register in a machine instruction.
pub enum MachRegConstraint {
    Any,
    RegClass(RegClassMask),
    FixedReg(RegUnit),
}

/// A machine instruction's virtual interface, allowing the architecture-independent backend
/// (regalloc, code emission) to perform register allocation, reason about dependences, and emit
/// code.
pub trait MachInst {
    fn name(&self) -> &'static str;
    fn num_regs(&self) -> usize;
    fn reg(&self, arg: usize) -> &MachReg;
    fn reg_mut(&mut self, arg: usize) -> &mut MachReg;
    fn size(&self) -> usize;
    fn emit(&self, sink: &mut CodeSink);
    // TODO: branch relaxation?
    // TODO: relocation for addresses/branch targets?
    // TODO: has_side_effects to inhibit DCE?
}

// TODO: pass over function (backward, tracking single vs multi-use / roots) to generate machinsts.
