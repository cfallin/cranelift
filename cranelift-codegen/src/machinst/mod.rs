//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::CodeSink;
use crate::ir::{Opcode, Type};
use crate::isa::registers::{RegClass, RegClassMask};
use crate::isa::RegUnit;
use crate::HashMap;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::Sum;
use smallvec::SmallVec;
use std::hash::Hash;

pub mod lower;
pub mod pattern;

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

/// The trait implemented by an architecture-specific opcode type.
pub trait MachInstOp: Clone + Debug {
    /// The name of the opcode.
    fn name(&self) -> &'static str;
}

/// An enum type that defines the kinds of a machine-specific argument.
pub trait MachInstArgKind: Clone + Debug + Hash + PartialEq + Eq {}

/// The trait implemented by an architecture-specific argument data blob. The purpose of this trait
/// is to allow the arch-specific part to inform the arch-independent part how many register slots
/// the argument has.
pub trait MachInstArg: Clone + Debug + MachInstArgGetKind {
    /// How many register slots this argument has.
    fn num_regs(&self) -> usize;
    /// What register class should be used for a value of the given type?
    fn regclass_for_type(ty: Type) -> RegClass;
}

/// A helper trait providing the link between a MachInstArg and its Kind. Automatically implemented
/// by the `mach_args!` macro.
pub trait MachInstArgGetKind {
    /// The kind enum for this arg type.
    type Kind: MachInstArgKind;
    /// What kind of argument is this?
    fn kind(&self) -> Self::Kind;
}

/// The argument slots of a machine instruction, parameterized on the architecture-specific
/// argument data.
pub type MachInstArgs<Arg: MachInstArg> = SmallVec<[Arg; 3]>;

/// The register slots of a machine instruction.
pub type MachInstRegs = SmallVec<[(MachReg, MachRegDefUse); 3]>;

/// Is a machine register reference a def, an use, or both?
#[derive(Clone, Debug)]
pub enum MachRegDefUse {
    /// A definition only.
    Def,
    /// An use only.
    Use,
    /// Both an use and a definition.
    DefUse,
}

/// A machine instruction.
pub struct MachInst<Op: MachInstOp, Arg: MachInstArg> {
    /// The opcode.
    pub op: Op,
    /// The argument data: this is target-specific. It does not include the registers; these are
    /// kept separately so that the machine-independent regalloc can access them.
    pub args: MachInstArgs<Arg>,
    /// The registers accessed and/or modified by this instruction.
    pub regs: MachInstRegs,
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInst<Op, Arg> {
    /// Create a new machine instruction.
    pub fn new(op: Op) -> MachInst<Op, Arg> {
        MachInst {
            op,
            args: SmallVec::new(),
            regs: SmallVec::new(),
        }
    }

    /// Add an argument to a machine instruction.
    pub fn with_arg(mut self, arg: Arg) -> Self {
        assert!(arg.num_regs() == 0);
        self.args.push(arg);
        self
    }

    /// Add an argument that takes a single register, as a def.
    pub fn with_arg_reg_def(mut self, arg: Arg, reg: MachReg) -> Self {
        assert!(arg.num_regs() == 1);
        self.args.push(arg);
        self.regs.push((reg, MachRegDefUse::Def));
        self
    }

    /// Add an argument that takes a single register, as an use.
    pub fn with_arg_reg_use(mut self, arg: Arg, reg: MachReg) -> Self {
        assert!(arg.num_regs() == 1);
        self.args.push(arg);
        self.regs.push((reg, MachRegDefUse::Use));
        self
    }

    /// Add an argument that takes two registers, both as uses.
    pub fn with_arg_2reg(mut self, arg: Arg, reg1: MachReg, reg2: MachReg) -> Self {
        assert!(arg.num_regs() == 2);
        self.args.push(arg);
        self.regs.push((reg1, MachRegDefUse::Use));
        self.regs.push((reg2, MachRegDefUse::Use));
        self
    }

    /// Returns the name of this machine instruction.
    pub fn name(&self) -> &'static str {
        self.op.name()
    }

    /// Returns the number of register arguments this machine instruction has.
    fn num_regs(&self) -> usize {
        self.regs.len()
    }

    /// Returns a borrow to the given register argument.
    fn reg(&self, idx: usize) -> &MachReg {
        &self.regs[idx].0
    }

    /// Returns a borrow to the given register argument, allowing mutation.
    fn reg_mut(&mut self, idx: usize) -> &mut MachReg {
        &mut self.regs[idx].0
    }

    /// Is the register a def?
    fn reg_is_def(&self, idx: usize) -> bool {
        match &self.regs[idx].1 {
            &MachRegDefUse::Def | &MachRegDefUse::DefUse => true,
            _ => false,
        }
    }

    /// Is the register an use?
    fn reg_is_use(&self, idx: usize) -> bool {
        match &self.regs[idx].1 {
            &MachRegDefUse::Use | &MachRegDefUse::DefUse => true,
            _ => false,
        }
    }

    // TODO: encoder (size, emit); branch relaxation; relocations.
}

/// A macro to allow a machine backend to define an opcode type.
#[macro_export]
macro_rules! mach_ops {
    ($name:ident, { $($op:ident),* }) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum $name {
            $($op),*
        }

        impl crate::machinst::MachInstOp for $name {
            fn name(&self) -> &'static str {
                match self {
                    $($name::$op => stringify!($op)),*
                }
            }
        }
    };
}

/// A macro to allow a machine backend to define its argument type.
#[macro_export]
macro_rules! mach_args {
    ($name:ident, $kind:ident, { $($op:ident($($oparg:tt)*)),* }) => {
        /// Kinds of machine-specific argument.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $kind {
            $($op),*
        }

        /// Machine-specific instruction argument type.
        #[derive(Clone, Debug)]
        pub enum $name {
            $(
                $op($($oparg)*)
            ),*
        }

        impl crate::machinst::MachInstArgKind for $kind {}

        impl crate::machinst::MachInstArgGetKind for $name {
            type Kind = $kind;

            fn kind(&self) -> Self::Kind {
                match self {
                    $(
                        &$name::$op(..) => $kind::$op
                    ),*
                }
            }
        }
    }
}
