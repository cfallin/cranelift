//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::CodeSink;
use crate::ir::{Opcode, Type};
use crate::isa::registers::RegClassMask;
use crate::isa::RegUnit;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A machine register in a machine instruction. Can be virtual (pre-regalloc) or allocated
/// (post-regalloc).
#[derive(Clone, Debug)]
pub enum MachReg {
    /// A virtual register, with some constraints that specify how it should be mapped to a real
    /// register during register allocation.
    Virtual(usize, MachRegConstraint),
    /// A real register assigned by register allocation.
    Allocated(RegUnit),
}

/// A constraint on a virtual register in a machine instruction.
#[derive(Clone, Debug)]
pub enum MachRegConstraint {
    /// Any register in one of the given register classes.
    RegClass(RegClassMask),
    /// A particular, fixed register.
    FixedReg(RegUnit),
}

/// A machine instruction's virtual interface, allowing the architecture-independent backend
/// (regalloc, code emission) to perform register allocation, reason about dependences, and emit
/// code.
pub trait MachInst {
    /// Returns the name of this machine instruction.
    fn name(&self) -> &'static str;
    /// Returns the number of register arguments this machine instruction has.
    fn num_regs(&self) -> usize;
    /// Returns a borrow to the given register argument.
    fn reg(&self, idx: usize) -> &MachReg;
    /// Returns a borrow to the given register argument, allowing mutation.
    fn reg_mut(&mut self, idx: usize) -> &mut MachReg;
    /// Returns the encoded size of this instruction in the machine code.
    fn size(&self) -> usize;
    /// Emits machine code for this instruction.
    fn emit(&self, sink: &mut CodeSink);
    // TODO: branch relaxation?
    // TODO: relocation for addresses/branch targets?
    // TODO: has_side_effects to inhibit DCE?
}

// TODO: pass over function (backward, tracking single vs multi-use / roots) to generate machinsts.
