//! This module exposes a pattern-matching system used to match on, and rewrite, instructions in
//! order to legalize a function and gradually lower it to machine instructions.

use crate::cursor::FuncCursor;
use crate::fx::FxHashMap;
use crate::ir::{Inst, Opcode};
use crate::isa::{RegClass, RegUnit};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A pattern matches an instruction by opcode and arguments (optionally recursively matching the
/// arguments' defining instructions as well). When a match occurs, a pattern specifies either
/// another instruction to replace the matched instruction, or else a binary encoding to generate.
pub struct Pattern {
    /// The opcode. A pattern must match a specific opcode.
    pub op: Opcode,
    /// The arguments. Each argument may further constrain the pattern, and may capture bindings.
    pub args: Vec<PatternArg>,
    /// The number of bindings this pattern captures.
    pub num_bindings: PatternBindingIndex,
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
pub struct PatternArg {
    /// The kind of pattern arg.
    pub kind: PatternArgKind,
    /// The binding for this arg: may capture or further constrain value.
    pub binding: PatternBinding,
}

/// The kind of pattern arg: this defines what conditions the pattern imposes on the operand.
pub enum PatternArgKind {
    /// Any register in a specified register class.
    RegClass(RegClass),
    /// A specific register.
    Reg(RegUnit),
    /// A nested pattern.
    Insn(Box<Pattern>),
}

/// An index into the bindings captured by the pattern.
pub type PatternBindingIndex = usize;

/// A binding occurs inside a pattern's argument, and either captures or further constrains the
/// specific value of the argument.
pub enum PatternBinding {
    /// Capture a value as a binding.
    Capture(PatternBindingIndex),
    /// Match a previously-captured value.
    Match(PatternBindingIndex),
}

/// An individual value captured by a pattern: one or both of a defining instruction and a
/// register.
pub struct PatternBindingValue {
    /// The instruction that defines this value, if any.
    pub def_insn: Option<Inst>,
    /// The register in which this value lives, if any.
    pub register: Option<RegUnit>,
}

/// The result of a pattern match.
pub struct PatternMatch {
    /// The bound values captured by the pattern.
    pub bindings: FxHashMap<PatternBindingIndex, PatternBindingValue>,
}

impl Pattern {
    /// Evaluate this pattern with the given cursor (which must point at an instruction), returning
    /// the match if successful.
    pub fn eval(&self, curs: &FuncCursor) -> Option<PatternMatch> {
        None
    }
}
