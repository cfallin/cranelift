//! This module allows a backend to construct a lookup table used by the instruction lowering
//! process.

use crate::ir::{Function, Inst, Opcode, Type};
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
pub type MachInstLowerFunc<Op: MachInstOp, Arg: MachInstArg> =
    for<'a> fn(&Function, Inst, &mut MachInstLowerCtx<'a, Op, Arg>) -> Arg;

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
