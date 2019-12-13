//! This module allows a backend to construct a lookup table used by the instruction lowering
//! process.

use crate::ir::{Function, Inst, InstructionData, Opcode, Type, Value};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::HashMap;

use smallvec::SmallVec;
use std::hash::Hash;

/*
 * ARM64 contexts:
 *
 * - Register
 * - Register, Register-shifted
 * - Register, Register-shifted, immediate
 * - Address
 */

/// Note: the below patterns are defined in terms of "contexts" (this is one dimension of the
/// lookup key, along with opcode, etc). A context describes a particular mode in which we want the
/// result: for example, a machine might have a context that describes an address for a load/store,
/// and another for an arbitrary result in a register. Within a particular context, we may or may
/// not have a means of generating code for a Value (result of an Inst), but if we do, we can do so
/// and provide a `Result` that describes (in a machine-dependent way) where the result lives.

/// This represents a table of lowering patterns.
pub struct MachInstLowerTable<Op: MachInstOp, Arg: MachInstArg> {
    patterns: HashMap<MachInstLowerKey<Arg>, SmallVec<[MachInstLowerPattern<Op, Arg>; 2]>>,
}

/// Lookup key for a lowering pattern table. A pattern applies to:
///
/// - A particular `Opcode` (IR instruction),
/// - With a particular `Type`, for polymorphic instructions,
/// - For one of possibly several values produced by this instruction,
/// - Producing the result into a particular kind of arg (register, immediate, address, etc) -- the
///   "context".
pub type MachInstLowerKey<Arg: MachInstArg> = (Opcode, Option<Type>, usize, Arg::Kind);

/// A function that actually emits whatever instruction is necessary and returns a machine-argument
/// representation of a `Value`, given the original Value.
///
/// If this returns a `None`, then it declines to generate the value, and another pattern should be
/// used.
pub type MachInstLowerFunc<Op: MachInstOp, Arg: MachInstArg> = for<'a> fn(
    &InstructionData,
    &[Value],
    &[Value],
    &mut MachInstLowerCtx<'a, Op, Arg>,
) -> Option<Arg>;

/// An individual pattern that generates code to produce
pub struct MachInstLowerPattern<Op: MachInstOp, Arg: MachInstArg> {
    key: MachInstLowerKey<Arg>,
    args: Vec<Arg::Kind>,
    emitter: MachInstLowerFunc<Op, Arg>,
}

impl<Op: MachInstOp, Arg: MachInstArg> MachInstLowerTable<Op, Arg> {
    /// Create a new lowering table.
    pub fn new() -> MachInstLowerTable<Op, Arg> {
        MachInstLowerTable {
            patterns: HashMap::new(),
        }
    }

    /// Add a pattern to this lowering table.
    pub fn add(
        &mut self,
        op: Opcode,
        ctrl_type: Option<Type>,
        result_idx: usize,
        result_kind: Arg::Kind,
        arg_kinds: &[Arg::Kind],
        emitter: MachInstLowerFunc<Op, Arg>,
    ) {
        let key = (op, ctrl_type, result_idx, result_kind);
        let pattern = MachInstLowerPattern {
            key: key.clone(),
            args: arg_kinds.iter().cloned().collect(),
            emitter,
        };
        self.patterns
            .entry(key)
            .or_insert_with(|| SmallVec::new())
            .push(pattern);
    }
}

/// Macro to help define lowering patterns.
///
/// Examples:
///
/// mach_patterns!(table, Arm64ArgKind, func, inst, args, results, lower, {
///
///   pattern(op Iadd, ty I64, (Reg, Reg) -> Reg, {
///     lower.emit_inst(MachInst::new(Arm64Op::AddExtend)
///       .with_arg_reg_def(Arm64Arg::Reg, lower.reg(results[0])),
///       .with_arg_reg_use(Arm64Arg::Reg, lower.reg(args[0])),
///       .with_arg_reg_use(Arm64Arg::Reg, lower.reg(args[1])));
///     Some(Arm64Arg::Reg)
///   };
///
///   pattern(op Iconst, ty I64, () -> Reg, {
///     if let &InstructionData::UnaryImm { ref imm, .. } = inst {
///       let val = imm.into::<i64>();
///       if arm64_imm12_in_range(val) {
///         lower.emit_inst(MachInst::new(Arm64Op::OrI)
///           .with_arg_reg_def(Arm64Arg::Reg, lower.reg(results[0])),
///           .with_arg_reg_use(Arm64Arg::Reg, lower.fixed_reg(XZR)),
///           .with_arg(Arm64Arg::Imm(val)));
///         return Some(Arm64Arg::Reg);
///       }
///     }
///     None
///   };
///
///   pattern(op Iconst, ty I64, () -> Imm12, {
///     ...
///   };
/// }
#[macro_export]
macro_rules! mach_patterns {
    ($table:expr, $kind:ty, $inst:ident, $args:ident, $results:ident,
     $lower:ident, { $(pattern($($arg:tt)*);)* }) => {
        $(
            crate::mach_pattern!($table, $kind, $inst, $args, $results, $lower, $($arg)*);
        )*
    }
}

/// Macro to help define one lowering pattern. Do not use directly; use mach_patterns! to reduce
/// the notation overhead of giving the table, kind, inst, lower symbols every invocations.
#[macro_export]
macro_rules! mach_pattern {
    // Four variants for two optional parameters: the polymorphic-instruction control type
    // (defaults to None), and the result index (defaults to 0).

    ($table:expr, $kind:ty, $inst:ident, $args:ident, $results:ident, $lower: ident,
     op $op:tt, ty $ctrl_ty:tt, ($($argkind:ident),*) -> $resultkind:ident @ $idx:expr,
     { $($body:tt)* })
    => {
        $table.add(crate::ir::Opcode::$op, Some($ctrl_ty), $idx, $kind::$resultkind,
        &[$($kind::$argkind),*], |$inst, $args, $results, $lower| { $($body)* });
    };

    ($table:expr, $kind:ty, $inst:ident, $args:ident, $results:ident, $lower: ident,
     op $op:tt, ($($argkind:ident),*) -> $resultkind:ident @ $idx:expr,
     { $($body:tt)* })
    => {
        $table.add(crate::ir::Opcode::$op, None, $idx, $kind::$resultkind,
        &[$($kind::$argkind),*], |$inst, $args, $results, $lower| { $($body)* });
    };

    ($table:expr, $kind:ty, $inst:ident, $args:ident, $results:ident, $lower: ident,
     op $op:tt, ty $ctrl_ty:tt, ($($argkind:ident),*) -> $resultkind:ident,
     { $($body:tt)* })
    => {
        $table.add(crate::ir::Opcode::$op, Some($ctrl_ty), 0, $kind::$resultkind,
        &[$($kind::$argkind),*], |$inst, $args, $results, $lower| { $($body)* });
    };

    ($table:expr, $kind:ty, $inst:ident, $args:ident, $results:ident, $lower: ident,
     op $op:tt, ($($argkind:ident),*) -> $resultkind:ident,
     { $($body:tt)* })
    => {
        $table.add(crate::ir::Opcode::$op, None, 0, $kind::$resultkind,
        &[$($kind::$argkind),*], |$inst, $args, $results, $lower| { $($body)* });
    };
}
