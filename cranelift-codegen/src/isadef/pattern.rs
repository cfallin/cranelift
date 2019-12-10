//! This module exposes a pattern-matching system used to match on, and rewrite, instructions in
//! order to legalize a function and gradually lower it to machine instructions.

use crate::binemit::CodeSink;
use crate::cursor::FuncCursor;
use crate::ir::Opcode;
use crate::isa::{RegClass, RegUnit};
use alloc::vec::Vec;

/// A pattern matches an instruction by opcode and arguments (optionally recursively matching the
/// arguments' defining instructions as well). When a match occurs, a pattern specifies either
/// another instruction to replace the matched instruction, or else a binary encoding to generate.
pub struct Pattern {
    /// The opcode. A pattern must match a specific opcode.
    pub op: Opcode,
    /// The arguments. Each argument may further constrain the pattern, and may capture bindings.
    pub args: Vec<PatternArg>,
}

/// A pattern argument corresponds to one operand, and may or may not match. The match status can
/// depend on both the kind and the binding:
///
/// - The kind may constrain the operand to a specific register or register class, or may specify a
///   nested pattern on the defining instruction of this value.
/// - The binding may specify a match against a previously-captured specific value.
///
/// If the argument matches, and its binding performs a capture, then the specific value is
/// captured for use in other arguments.
///
/// Note that all pattern arguments that perform captures are evaluated for matches before
/// arguments that match against prior captures; or, in other words, dependencies from capturing to
/// matching pattern args are respected.
///
/// TODO: for tree-patterns, consider how to handle registers when they may be redefined between
/// original and current instruction. (i.e., we are not matching on SSA anymore, so we have to be
/// mindful of bindings as (register, time), not just register.)
pub struct PatternArg {
    /// The kind of pattern arg.
    pub kind: PatternArgKind,
}

/// The kind of pattern arg: this defines what conditions the pattern imposes on the operand.
pub enum PatternArgKind {
    /// Any register in a specified register class.
    RegClass(RegClass),
    /// A specific register.
    Reg(RegUnit),
}

/// A function that gives the size of an encodable instruction. Returned as part of a pattern
/// match.
pub type SizeFunc = fn(&FuncCursor) -> usize;

/// A function that emits an encodable instruction to a code sink. Returned as part of a pattern
/// match.
pub type EmitFunc = fn(&FuncCursor, &mut dyn CodeSink);

/// A function that replaces a non-encodable instruction with other instructions. Returned as part
/// of a pattern match.
pub type ReplaceFunc = fn(&FuncCursor);

/// An action is the work that is executed when a pattern matches. An action can either specify a
/// machine instruction into which this IR instruction can be encoded, or it can specify a
/// legalization step that replaces this instruction with another.
pub enum Action {
    /// The result of the pattern match is that we have an encoding to a concrete machine
    /// instruction. The SizeFunc gives the encoded size of the machine code, and the EmitFunc
    /// actually emits it.
    Encoding(SizeFunc, EmitFunc),
    /// The result of the pattern match is that we need to replace the instruction with others,
    /// getting closer to machine-encodable instructions. The ReplaceFunc is given a cursor and
    /// has the job of actually doing the replacement.
    Legalize(ReplaceFunc),
}

/// A PatternAction connects one or more patterns with an action to take if any of those patterns
/// match.
pub struct PatternAction {
    /// If any of these patterns match, invoke the action.
    pub patterns: Vec<Pattern>,
    /// The action to invoke.
    pub action: Action,
}

// Macros to create arrays of PatternAction clauses.

/// Define a `PatternAction` using a very simple pattern->action DSL. Examples:
///
/// let pats = isa_patterns! {
///   (Add, rc:GPR, rc:GPR, rc:GPR) => emit(curs, sink) { /* ... */ } size(curs) { 4 },
///   (Add, rc:GPR, r:R0, rc:GPR) => replace(curs) { curs.remove_inst(); }
/// };
///
/// The first arg after the opcode is the insn dest and the rest are the operands.
///
/// The actions for now are thunks of Rust code given the raw FuncCursor, able to perform arbitrary
/// instruction updates, insertions or removals (replace case) or arbitrary code emission (emit
/// case). We should also eventually have alternates that compactly encode the most common
/// behaviors: e.g., emitting a fixed-length RISC insn with a bit encoding inserting registers and
/// immediates; and replacing an insn with others using the same (Op, args) syntax as the patterns.
///
/// Eventually we want to support captures and matches against those captures as well,
/// so we can do things like:
///
///   (Add, ra@rc:GPR, ra, ra) => replace_inst (Lsh, ra, 2)
///
/// Fn(&[PatternAction]) -> EncodingLookupTable
///
#[macro_export]
macro_rules! isa_patterns {
    ($global:ident, { $($head:tt => action { $($action:tt)* };)* }) =>
        (pub static $global: &'static [crate::isadef::PatternAction] = &[$(crate::isadef::PatternAction {
            patterns: crate::isa_pattern_head!($head),
            action: crate::isa_pattern_action!($($action)*),
        }),*];);
}

/// Helper macro to create a vec of `Pattern` instances from the head of an ISA-pattern rule.
#[macro_export]
macro_rules! isa_pattern_head {
    ($(($op:tt, $(($($args:tt)*)),*))|*) =>
     (vec![$(crate::isadef::Pattern { op: crate::ir::Opcode::$op, args: vec![
         $(crate::isa_pattern_head_arg!($($args)*)),*] }),*]);
}

/// Helper macro to create an `Action` value from the body of an ISA-pattern rule.
#[macro_export]
macro_rules! isa_pattern_action {
    (emit($emitcurs:ident, $sink:ident) $emitbody:block size($sizecurs:ident) $sizebody:block) => {
        crate::isadef::Action::Encoding(|$sizecurs| $sizebody, |$emitcurs, $sink| $emitbody)
    };
    (replace($curs:ident) $body:block) => {
        crate::isadef::Action::Legalize(|$curs| $body)
    };
}

/// Helper macro.
#[macro_export]
macro_rules! isa_pattern_head_arg {
    (r : $reg:ident) => {
        crate::isadef::PatternArg {
            kind: crate::isadef::PatternArgKind::Reg(&$reg),
        }
    };
    (rc : $regclass:ident) => {
        crate::isadef::PatternArg {
            kind: crate::isadef::PatternArgKind::RegClass(&$regclass),
        }
    };
}

/// Macro to define register banks and classes for an ISA.
#[macro_export]
macro_rules! isa_regs {
    ($reginfo:ident,
     banks { $($bankclause: ident $bankargs:tt ;)* }
     classes { $($classclause:ident $classargs:tt ;)* }
    ) => (
        $(crate::isa_regs_class!{$reginfo, $classclause $classargs})*

        pub static $reginfo: crate::isa::RegInfo = crate::isa::RegInfo {
            banks: &[$(crate::isa_regs_bank!{$bankclause $bankargs}),*],
            classes: &[$(&$classclause),*],
        };
    )
}

/// Helper macro.
#[macro_export]
macro_rules! isa_regs_bank {
    ($bankname:ident(
            from($from:expr),
            to($to:expr),
            units($units:expr),
            track_pressure($tp:expr),
            name_prefix($pre:ident))) => {
        crate::isa::registers::RegBank {
            name: stringify!($bankname),
            first_unit: $from,
            units: $to + 1 - $from,
            names: &[],
            prefix: stringify!($pre),
            first_toprc: 0, // TODO
            num_toprcs: 0, // TODO
            pressure_tracking: $tp,
        }
    };
    ($bankname:ident(
            from($from:expr),
            to($to:expr),
            units($units:expr),
            track_pressure($tp:expr),
            names([$($name:ident),*]))) => {
        crate::isa::registers::RegBank {
            name: stringify!($bankname),
            first_unit: $from,
            units: $to + 1 - $from,
            names: &[$(stringify!($name)),*],
            prefix: stringify!($bankname),
            first_toprc: 0, // TODO
            num_toprcs: 0, // TODO
            pressure_tracking: $tp,
        }
    };
}

/// Helper macro.
#[macro_export]
macro_rules! isa_regs_class {
    ($reginfo:ident, $name:ident(index($index:expr), bank($bank:expr), first($first:expr), mask $mask:tt)) => {
        pub static $name: crate::isa::registers::RegClassData =
            crate::isa::registers::RegClassData {
                name: stringify!($name),
                index: $index,
                width: 1,
                bank: $bank,
                toprc: $index,
                first: $first,
                subclasses: 0,
                mask: crate::isa_regs_mask!($mask),
                info: &$reginfo,
                pinned_reg: None,
            };
    };
}

/// Helper macro.
#[macro_export]
macro_rules! isa_regs_mask {
    (($arg1:tt, $arg2:tt, $arg3:tt)) => {
        [
            crate::isa_regs_mask_one!(0, $arg1),
            crate::isa_regs_mask_one!(32, $arg2),
            crate::isa_regs_mask_one!(64, $arg3),
        ]
    };
}

/// Helper macro.
#[macro_export]
macro_rules! isa_regs_mask_one {
    ($base:expr, []) => {
        0
    };
    ($base:expr, [$from:expr, $to:expr]) => {
        ((((1u64 << ($to - $base + 1)) - 1) - ((1u64 << ($from - $base)) - 1)) & 0xffff_ffff) as u32
    };
}
