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
#[macro_export]
macro_rules! isa_patterns {
    (($head:tt => $action:tt);*) =>
        (&[$(PatternAction {
            patterns: isa_pattern_head!($head),
            action: isa_pattern_action!($action),
        }),*]);
}

/// Helper macro to create a vec of `Pattern` instances from the head of an ISA-pattern rule.
#[macro_export]
macro_rules! isa_pattern_head {
    ($($op:tt, $(args:tt),*)|*) =>
     (vec![$(Pattern { op: Opcodes::$op, args: vec![
         $(isa_pattern_head_arg!($args)),*] }),*]);
}

/// Helper macro to create an `Action` value from the body of an ISA-pattern rule.
#[macro_export]
macro_rules! isa_pattern_action {
    (emit($emitcurs:ident, $sink:ident) $emitbody:block size($sizecurs:ident) $sizebody:block) => {
        Action::Encoding(|$emitcurs, $sink| $emitbody, |$sizecurs| $sizebody)
    };
    (replace($curs:ident) $body:block) => {
        Action::Legalize(|$curs| $body)
    };
}

macro_rules! isa_pattern_head_arg {
    (r:$reg:ident) => {
        PatternArgKind::Reg($reg)
    };
    (rc:$regclass:ident) => {
        PatternArgKind::RegClass($regclass)
    };
}

#[macro_export]
macro_rules! isa_regclass {
}
