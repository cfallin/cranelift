//! This module exposes the machine-specific backend definition pieces.
//!
//! Machine backends define two types, an `Op` (opcode) and `Arg` (instruction argument, meant to
//! encapsulate, register, immediate, and memory references).
//!
//! Machine instructions (`MachInst` instances) are statically parameterized on these types for efficiency.
//! They in turn are held by an implementation of the `MachInsts` trait that the `Function` holds
//! by dynamic reference, in order to avoid parameterizing the entire IR world on machine backend.
//!
//! The `Function` requests a lowering of its IR (`ir::Inst`s) into machine-specific code, and the
//! results are kept alongside the original IR, with a 1-to-N correspondence: each Cranelift IR
//! instruction can correspond to N contiguous machine instructions. (N=0 is possible, if e.g. two
//! IR instructions are fused into a single machine instruction: then the final value-producing
//! instruction is the only one that has machine instructions.)
//!
//! To keep the interface with the register allocator simple, the control-flow and the register
//! defs/uses of the Cranelift IR remain mostly canonical, even after lowering. There is one
//! exception: because instruction lowering may require extra temps within a sequence of machine
//! instructions (a Value that is def'd and use'd immediately), or may use a value from an earlier
//! IR instruction if fusing instructions, we need to be able to add new args and results to
//! Cranelift IR instructions. Rather than rewrite the instructions in place, or somehow alter
//! their format, the `MachInsts` container keeps extra `Value` args and results for instructions
//! as it goes. The register allocator queries this as well as the original instruction. (Why not
//! just rewrite the whole list of defs/uses and make the regalloc ignore the originals? The common
//! case is that no defs/uses change; the "exceptions" list should in the common case be very
//! short or empty, leading to less memory overhead.)

use crate::binemit::CodeSink;
use crate::entity::EntityRef;
use crate::entity::SecondaryMap;
use crate::ir::{DataFlowGraph, Inst, Opcode, Type, Value};
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
pub trait MachInstArg: Clone + Debug + MachInstArgGetKind {}

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

impl<Op: MachInstOp, Arg: MachInstArg> MachInst<Op, Arg> {
    /// Create a new machine instruction.
    pub fn new(op: Op) -> MachInst<Op, Arg> {
        MachInst {
            op,
            args: SmallVec::new(),
        }
    }

    /// Add an argument to a machine instruction.
    pub fn with_arg(mut self, arg: Arg) -> Self {
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
///
/// This trait represents, in addition to the lowered machine instructions per IR instruction, an
/// overlay over IR  instructions that can define new inputs (args), new outputs (results), and new
/// Values with types.
pub trait MachInsts: Debug {
    /// Clear the list of machine instructions.
    fn clear(&mut self);
    /// Get the number of machine instructions.
    fn num_machinsts(&self) -> usize;
    /// Get the extra arg values for a given IR instruction added during lowering.
    fn extra_args(&self, inst: Inst) -> &[Value];
    /// Get the extra result values for a given IR instruction added during lowering.
    fn extra_results(&self, inst: Inst) -> &[Value];
    /// Get the number of normal and extra args for the given IR instruction.
    fn num_total_args(&self, dfg: &DataFlowGraph, inst: Inst) -> usize;
    /// Get the given normal or extra arg for the given IR instruction.
    fn get_arg(&self, dfg: &DataFlowGraph, inst: Inst, idx: usize) -> Value;
    /// Get the number of normal and extra results for the given IR instruction.
    fn num_total_results(&self, dfg: &DataFlowGraph, inst: Inst) -> usize;
    /// Get the given normal or extra arg for the given IR instruction.
    fn get_result(&self, dfg: &DataFlowGraph, inst: Inst, idx: usize) -> Value;
    /// Get the type of the given Value.
    fn get_type(&self, dfg: &DataFlowGraph, value: Value) -> Type;
}

/// Canonical implementation of MachInsts.
#[derive(Clone, Debug)]
pub struct MachInstsImpl<Op: MachInstOp, Arg: MachInstArg> {
    entries: SecondaryMap<Inst, MachInstsEntry>,
    mach_insts: Vec<MachInst<Op, Arg>>,
    extra_args: Vec<Value>,
    extra_results: Vec<Value>,
    first_extra_value: usize,
    next_value: usize,
    extra_values: Vec<Type>,
    last_inst: Option<Inst>,
}

#[derive(Clone, Debug, Default)]
struct MachInstsEntry {
    // Indices into the respective dense Vecs. Pack the u32s together for efficiency: this struct
    // should be 16 bytes total with padding.
    inst_start: u32,
    extra_arg_start: u32,
    extra_result_start: u32,
    inst_count: u8,
    extra_arg_count: u8,
    extra_result_count: u8,
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInstsImpl<Op, Arg> {
    /// Create a new MachInstsImpl.
    pub fn new(dfg: &DataFlowGraph) -> MachInstsImpl<Op, Arg> {
        let dfg_values = dfg.num_values();
        MachInstsImpl {
            entries: SecondaryMap::with_default(Default::default()),
            mach_insts: vec![],
            extra_args: vec![],
            extra_results: vec![],
            first_extra_value: dfg_values,
            next_value: dfg_values,
            extra_values: vec![],
            last_inst: None,
        }
    }

    fn start_inst(&mut self, from: Inst) {
        let entry = &mut self.entries[from];
        if &self.last_inst != &Some(from) {
            assert!(entry.inst_count == 0);
            assert!(entry.extra_arg_count == 0);
            entry.inst_start = self.mach_insts.len() as u32;
            entry.extra_arg_start = self.extra_args.len() as u32;
            entry.extra_result_start = self.extra_results.len() as u32;
            self.last_inst = Some(from);
        }
    }

    fn new_value(&mut self, ty: Type) -> Value {
        let idx = self.next_value;
        self.next_value += 1;
        self.extra_values.push(ty);
        Value::new(idx)
    }

    /// Add a new MachInst corresponding to the given IR inst.
    pub fn add_inst(&mut self, from: Inst, inst: MachInst<Op, Arg>) {
        self.start_inst(from);
        let entry = &mut self.entries[from];
        entry.inst_count += 1;
        self.mach_insts.push(inst);
    }

    /// Add a new extra arg corresponding to the given IR inst. Returns the index of this extra
    /// arg.
    pub fn add_extra_arg(&mut self, from: Inst, value: Value) -> usize {
        self.start_inst(from);
        let entry = &mut self.entries[from];
        let idx = entry.extra_arg_count as usize;
        entry.extra_arg_count += 1;
        self.extra_args.push(value);
        idx
    }

    /// Add a new extra output corresponding to the given IR inst. Returns the index of this extra
    /// arg.
    pub fn add_extra_result(&mut self, from: Inst, ty: Type) -> usize {
        self.start_inst(from);
        let entry = &mut self.entries[from];
        let idx = entry.extra_arg_count as usize;
        entry.extra_result_count += 1;
        let value = self.new_value(ty);
        self.extra_results.push(value);
        idx
    }
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInsts for MachInstsImpl<Op, Arg> {
    /// Clear the list of machine instructions.
    fn clear(&mut self) {
        self.entries.clear();
        self.mach_insts.clear();
        self.extra_args.clear();
        self.extra_results.clear();
    }

    /// Get the number of machine instructions.
    fn num_machinsts(&self) -> usize {
        self.mach_insts.len()
    }

    /// Get the extra arg values for a given IR instruction added during lowering.
    fn extra_args(&self, inst: Inst) -> &[Value] {
        let entry = &self.entries[inst];
        let start = entry.extra_arg_start as usize;
        let end = start + (entry.extra_arg_count as usize);
        &self.extra_args[start..end]
    }

    /// Get the extra result values for a given IR instruction added during lowering.
    fn extra_results(&self, inst: Inst) -> &[Value] {
        let entry = &self.entries[inst];
        let start = entry.extra_result_start as usize;
        let end = start + (entry.extra_result_count as usize);
        &self.extra_results[start..end]
    }

    fn num_total_args(&self, dfg: &DataFlowGraph, inst: Inst) -> usize {
        dfg.inst_args(inst).len() + self.extra_args(inst).len()
    }

    fn num_total_results(&self, dfg: &DataFlowGraph, inst: Inst) -> usize {
        dfg.inst_results(inst).len() + self.extra_results(inst).len()
    }

    fn get_arg(&self, dfg: &DataFlowGraph, inst: Inst, idx: usize) -> Value {
        let args = dfg.inst_args(inst);
        if idx < args.len() {
            args[idx]
        } else {
            self.extra_args(inst)[idx - args.len()]
        }
    }

    fn get_result(&self, dfg: &DataFlowGraph, inst: Inst, idx: usize) -> Value {
        let results = dfg.inst_results(inst);
        if idx < results.len() {
            results[idx]
        } else {
            self.extra_results(inst)[idx - results.len()]
        }
    }

    fn get_type(&self, dfg: &DataFlowGraph, v: Value) -> Type {
        if v.index() < self.first_extra_value {
            dfg.value_type(v)
        } else {
            self.extra_values[v.index() - self.first_extra_value]
        }
    }
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

        impl crate::machinst::MachInstArg for $name {}
    }
}
