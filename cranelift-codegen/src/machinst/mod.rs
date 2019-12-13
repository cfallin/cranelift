//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::CodeSink;
use crate::ir::{Opcode, Type};
use crate::isa::registers::{RegClass, RegClassMask};
use crate::isa::RegUnit;
use crate::HashMap;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A machine register in a machine instruction. Can be virtual (pre-regalloc) or allocated
/// (post-regalloc).
#[derive(Clone, Debug)]
pub enum MachReg {
    /// A virtual register.
    Virtual(usize),
    /// A real register assigned by register allocation.
    Allocated(RegUnit),
}

impl MachReg {
    /// If this is a virtual register, return the virtual register number.
    pub fn as_virtual(&self) -> Option<usize> {
        match self {
            &MachReg::Virtual(index) => Some(index),
            _ => None,
        }
    }
    /// If this is an allocated register, return the concrete machine register number.
    pub fn as_allocated(&self) -> Option<RegUnit> {
        match self {
            &MachReg::Allocated(reg) => Some(reg),
            _ => None,
        }
    }
    /// Is this a virtual register?
    pub fn is_virtual(&self) -> bool {
        self.as_virtual().is_some()
    }
    /// Is this an allocated register?
    pub fn is_allocated(&self) -> bool {
        self.as_allocated().is_some()
    }
}

/// A constraint on a virtual register in a machine instruction.
#[derive(Clone, Debug)]
pub enum MachRegConstraint {
    /// Any register in one of the given register classes.
    RegClass(RegClassMask),
    /// A particular, fixed register.
    FixedReg(RegUnit),
}

impl MachRegConstraint {
    /// Create a machine-register constraint that chooses a register from a single register class
    /// (or its subclasses).
    pub fn from_class(rc: RegClass) -> MachRegConstraint {
        MachRegConstraint::RegClass(rc.subclasses)
    }
    /// Create a machine-register constraint that chooses a fixed register.
    pub fn from_fixed(ru: RegUnit) -> MachRegConstraint {
        MachRegConstraint::FixedReg(ru)
    }
}

/// A simple counter that allocates virtual-register numbers.
pub struct MachRegCounter {
    next: usize,
}

impl MachRegCounter {
    /// Create a new virtual-register number allocator.
    pub fn new() -> MachRegCounter {
        MachRegCounter { next: 1 }
    }

    /// Allocate a fresh virtual register number.
    pub fn alloc(&mut self) -> MachReg {
        let idx = self.next;
        self.next += 1;
        MachReg::Virtual(idx)
    }
}

/// A set of constraints on virtual registers, typically held at the Function level to be used by
/// regalloc.
pub struct MachRegConstraints {
    constraints: Vec<(MachReg, MachRegConstraint)>,
}

impl MachRegConstraints {
    /// Create a new set of register constraints.
    pub fn new() -> MachRegConstraints {
        MachRegConstraints {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to a register.
    pub fn add(&mut self, reg: &MachReg, constraint: MachRegConstraint) {
        assert!(reg.is_virtual());
        self.constraints.push((reg.clone(), constraint));
    }

    /// Return a list of all constraints with their associated registers.
    pub fn constraints(&self) -> &[(MachReg, MachRegConstraint)] {
        &self.constraints[..]
    }
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
