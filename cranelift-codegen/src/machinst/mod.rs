//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::CodeSink;
use crate::ir::{Opcode, Type, Value};
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
pub mod pattern_prefix;

pub use lower::*;
pub use pattern::*;

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

/// A set of constraints on virtual registers, typically held at the Function level to be used by
/// regalloc.
#[derive(Clone, Debug)]
pub struct MachRegConstraints {
    constraints: Vec<(Value, MachRegConstraint)>,
}

impl MachRegConstraints {
    /// Create a new set of register constraints.
    pub fn new() -> MachRegConstraints {
        MachRegConstraints {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to a register.
    pub fn add(&mut self, value: Value, constraint: MachRegConstraint) {
        self.constraints.push((value, constraint));
    }

    /// Return a list of all constraints with their associated registers.
    pub fn constraints(&self) -> &[(Value, MachRegConstraint)] {
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
/// is to allow the arch-specific part to inform the arch-independent part about the `Value`s
/// (registers) that the argument defines and uses.
pub trait MachInstArg: Clone + Debug + MachInstArgGetKind {
    /// What values does this arg define?
    fn defs(&self) -> &[Value];
    /// What values does this arg use?
    fn uses(&self) -> &[Value];
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
pub type MachInstArgs<Arg> = SmallVec<[Arg; 3]>;

/// A machine instruction.
#[derive(Clone, Debug)]
pub struct MachInst<Op: MachInstOp, Arg: MachInstArg> {
    /// The opcode.
    pub op: Op,
    /// The argument data: this is target-specific. It does not include the registers; these are
    /// kept separately so that the machine-independent regalloc can access them.
    pub args: MachInstArgs<Arg>,
}

impl<Op: MachInstOp, Arg: MachInstArg, Reg: Debug + Clone> MachInst<Op, Arg, Reg> {
    /// Create a new machine instruction.
    pub fn new(op: Op) -> MachInst<Op, Arg, Reg> {
        MachInst {
            op,
            args: SmallVec::new(),
        }
    }

    /// Add an argument to a machine instruction.
    pub fn with_arg(mut self, arg: Arg) -> Self {
        assert!(arg.num_regs() == 0);
        self.args.push(arg);
        self
    }

    /// Returns the name of this machine instruction.
    pub fn name(&self) -> &'static str {
        self.op.name()
    }

    // TODO: encoder (size, emit); branch relaxation; relocations.
}

/// A trait wrapping a list of machine instructions, held by `Function`. This allows the Cranelift
/// IR, which holds the machine-specific instructions after lowering, to remain unparameterized on
/// Op/Arg (and machine-specific types in general). The trait provides methods to allow for codegen
/// but does not (cannot) leak individual MachInst instances.
pub trait MachInsts: Clone + Debug {
    fn clear(&mut self);
}

#[derive(Clone, Debug)]
pub struct MachInstsImpl<Op: MachInstOp, Arg: MachInstArg> {
    insts: Vec<MachInst<Op, Arg>>,
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
