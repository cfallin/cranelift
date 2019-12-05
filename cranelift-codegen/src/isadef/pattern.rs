//! This module exposes a pattern-matching system used to match on, and rewrite, instructions in
//! order to legalize a function and gradually lower it to machine instructions.

use crate::cursor::{Cursor, FuncCursor};
use crate::fx::{FxHashMap, FxHashSet};
use crate::ir::{Inst, Opcode};
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
///
/// TODO: for tree-patterns, consider how to handle registers when they may be redefined between
/// original and current instruction. (i.e., we are not matching on SSA anymore, so we have to be
/// mindful of bindings as (register, time), not just register.)
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
}

/// An index into the bindings captured by the pattern.
pub type PatternBindingIndex = usize;

/// A binding occurs inside a pattern's argument, and either captures or further constrains the
/// specific value of the argument.
pub enum PatternBinding {
    /// Do not capture this register.
    None,
    /// Capture a value as a binding.
    Capture(PatternBindingIndex),
}

/// An individual value captured by a pattern.
pub struct PatternBindingValue {
    /// The register in which this value lives, if any.
    pub register: RegUnit,
}

/// The result of a pattern match.
pub struct PatternMatch {
    /// The bound values captured by the pattern.
    pub bindings: FxHashMap<PatternBindingIndex, PatternBindingValue>,
}

/// An error discovered when compiling a pattern to an execution plan.
pub enum PatternError {
    /// A binding index was never defined by a capture.
    UncapturedBinding(PatternBindingIndex),
    /// A binding index was out of bounds.
    OutOfBoundsBinding(PatternBindingIndex),
}

impl Pattern {
    /// Check the pattern for errors.
    pub fn check(&self) -> Result<(), Vec<PatternError>> {
        let mut errors = vec![];
        let mut captures = FxHashSet();
        for arg in &self.args {
            if let &PatternBinding::Capture(idx) = &arg.binding {
                if idx >= self.num_bindings {
                    errors.push(PatternError::OutOfBoundsBinding(idx));
                } else {
                    captures.insert(idx);
                }
            }
        }
        for idx in 0..self.num_bindings {
            if !captures.contains(&idx) {
                errors.push(PatternError::UncapturedBinding(idx));
            }
        }

        if errors.len() > 0 {
            Err(errors)
        } else {
            Ok(())
        }
    }

    /// Evaluate the pattern at a particular instruction, given by the cursor `curs`. Return the
    /// match's bindings if the pattern matches, or `None` otherwise.
    pub fn eval(&self, curs: &FuncCursor) -> Option<PatternMatch> {
        assert!(curs.current_inst().is_some());
        let inst = curs.current_inst().unwrap();
        let op = curs.func.dfg[inst].opcode();
        None
    }
}
