//! This module allows a backend to construct a lookup table used by the instruction lowering
//! process.

use crate::ir::{Function, Inst, InstructionData, Opcode, Type, Value, ValueDef};
use crate::isa::registers::{RegClass, RegUnit};
use crate::machinst::lower::*;
use crate::machinst::pattern_prefix::*;
use crate::machinst::*;
use crate::num_uses::NumUses;

use crate::HashMap;

/*
 * Algorithm:
 *
 * - After collecting a set of tree-prefixes and actions:
 *   - Sort the prefixes by first op.
 *   - Build the prefixes into a single prefix table.
 *
 * - When lowering a block:
 *   - For each value that we want to generate:
 *     - Construct an opcode tree. Here we slurp all single-use, side-effect-free
 *       values into the tree greedily.
 *     - Find the prefixes starting with the root opcode, and try to match each in turn.
 *     - For each that matches, invoke the lowering action. The first lowering action
 *       that returns `true` terminates the codegen for this value.
 *       - The lowering action is given the instructions and their argument registers
 *         in the order they are mentioned in the tree prefix, and the result regs
 *         of the root instruction.
 *       - The lowering action, in turn, can invoke ctx.emit(machinst) to emit a
 *         machine instruction and/or ctx.unused(inst) to note that inst is now unused
 *         and need not be generated.
 */

/// A table of tree patterns and lowering actions.
pub struct LowerTable<Op: MachInstOp, Arg: MachInstArg> {
    prefix_pool: PatternPrefixPool,
    entries: HashMap<Opcode, SmallVec<[LowerTableEntry<Op, Arg>; 4]>>,
}

/// A single entry in the lowering table.
pub struct LowerTableEntry<Op: MachInstOp, Arg: MachInstArg> {
    /// The prefix (tree pattern) associated with this entry.
    pub prefix: PatternPrefixRange,
    /// The lowering action to perform.
    pub action: LowerAction<Op, Arg>,
}

impl<Op: MachInstOp, Arg: MachInstArg> LowerTable<Op, Arg> {
    /// Create a new lowering table.
    pub fn new() -> LowerTable<Op, Arg> {
        LowerTable {
            prefix_pool: PatternPrefixPool::new(),
            entries: HashMap::new(),
        }
    }

    /// Get the prefix pool of the table to allow building of a new prefix.
    pub fn pool<'a>(&'a mut self) -> &'a mut PatternPrefixPool {
        &mut self.prefix_pool
    }

    /// Add a pattern to the lowering table.
    pub fn add(&mut self, prefix: PatternPrefixRange, action: LowerAction<Op, Arg>) {
        let op = self.prefix_pool.get(&prefix).root_op();
        self.entries
            .entry(op)
            .or_insert_with(SmallVec::new)
            .push(LowerTableEntry { prefix, action });
    }

    /// Get the entries associated with a given root opcode.
    pub fn get_entries(&self, root_op: Opcode) -> Option<&[LowerTableEntry<Op, Arg>]> {
        self.entries.get(&root_op).map(|v| &v[..])
    }
}
