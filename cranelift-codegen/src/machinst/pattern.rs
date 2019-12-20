//! This module allows a backend to construct a lookup table used by the instruction lowering
//! process.

use crate::ir::{Function, Inst, InstructionData, Opcode, Type, Value, ValueDef};
use crate::isa::registers::{RegClass, RegUnit};
use crate::machinst::lower::*;
use crate::machinst::pattern_prefix::*;
use crate::machinst::*;
use crate::num_uses::NumUses;

use crate::HashMap;

use std::fmt;

/// A table of tree patterns and lowering actions.
#[derive(Clone, Debug)]
pub struct LowerTable<Op: MachInstOp, Arg: MachInstArg> {
    prefix_pool: PatternPrefixPool,
    entries: HashMap<Opcode, SmallVec<[LowerTableEntry<Op, Arg>; 4]>>,
}

/// A single entry in the lowering table.
#[derive(Clone)]
pub struct LowerTableEntry<Op: MachInstOp, Arg: MachInstArg> {
    /// The prefix (tree pattern) associated with this entry.
    pub prefix: PatternPrefixRange,
    /// The lowering action to perform.
    pub action: LowerAction<Op, Arg>,
}

impl<Op: MachInstOp, Arg: MachInstArg> fmt::Debug for LowerTableEntry<Op, Arg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LowerTableEntry(prefix = {:?})", self.prefix)
    }
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
    pub fn pool_mut(&mut self) -> &mut PatternPrefixPool {
        &mut self.prefix_pool
    }

    /// Get the prefix pool of the table to allow looking up a prefix.
    pub fn pool(&self) -> &PatternPrefixPool {
        &self.prefix_pool
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

/// Add a lowering pattern to a LowerTable.
#[macro_export]
macro_rules! lower_pattern {
    ($t:expr, ($($tree:tt)*), |$ctx:ident| $body:block) => {
        {
            let mut pat = $t.pool_mut().build();
            crate::lower_pattern_tree!(pat, ($($tree)*));
            let pat = pat.build();
            $t.add(pat, |$ctx| { $body });
        }
    };
    ($t:expr, $tree:ident, |$ctx:ident| $body:block) => {
        {
            let mut pat = $t.pool_mut().build();
            crate::lower_pattern_tree!(pat, $tree);
            let pat = pat.build();
            $t.add(pat, |$ctx| { $body });
        }
    };
}

/// Helper for `lower_pattern!`.
#[macro_export]
macro_rules! lower_pattern_tree {
    ($pat:ident, _) => {
        $pat = $pat.any();
    };
    ($pat:ident, $op:ident) => {
        $pat = $pat.opcode(crate::ir::Opcode::$op);
    };
    ($pat:ident, ($op:ident $($arg:tt)*)) => {
        $pat = $pat.opcode_with_args(crate::ir::Opcode::$op);
        $(
            crate::lower_pattern_tree!($pat, $arg);
        )*
        $pat = $pat.args_end();
    };
}
