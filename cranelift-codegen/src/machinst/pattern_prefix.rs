//! This module defines a "pattern prefix" data structure that provides a primitive for
//! tree-matching.

use crate::ir::Opcode;

use alloc::vec::Vec;
use std::borrow::Cow;
use std::iter;

/// A prefix for use in the one-level table. Represents a tree of opcodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternPrefix<'a> {
    // Each tuple: opcode and depth. Prefix is given in tree pre-order. If opcode is `None` then
    // any opcode matches.
    elems: Cow<'a, [PatternPrefixElem]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PatternPrefixElem {
    Any,
    Op(Opcode),
    OpWithArgs(Opcode),
    EndArgs,
}

impl PatternPrefixElem {
    fn pushes(&self) -> bool {
        match self {
            &PatternPrefixElem::OpWithArgs(..) => true,
            _ => false,
        }
    }
    fn pops(&self) -> bool {
        match self {
            &PatternPrefixElem::EndArgs => true,
            _ => false,
        }
    }
    fn is_op(&self, op: Opcode) -> bool {
        match self {
            &PatternPrefixElem::Op(o) => o == op,
            &PatternPrefixElem::OpWithArgs(o) => o == op,
            _ => false,
        }
    }
    fn get_op(&self) -> Option<Opcode> {
        match self {
            &PatternPrefixElem::Op(o) => Some(o),
            &PatternPrefixElem::OpWithArgs(o) => Some(o),
            _ => None,
        }
    }
}

impl<'a> PatternPrefix<'a> {
    fn new(elems: Cow<'a, [PatternPrefixElem]>) -> PatternPrefix<'a> {
        PatternPrefix { elems }
    }

    /// Return the root opcode in the pattern.
    pub fn root_op(&self) -> Opcode {
        assert!(self.elems.len() >= 1);
        assert!(self.elems[0].get_op().is_some());
        self.elems[0].get_op().unwrap()
    }

    fn skip_subtree<'it, I: Iterator<Item = &'it PatternPrefixElem>>(i: &mut I) {
        let mut depth = 1;
        while let Some(elem) = i.next() {
            if elem.pushes() {
                depth += 1;
            } else if elem.pops() {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
    }

    /// Does this pattern prefix match another?
    pub fn matches(&self, other: &PatternPrefix) -> bool {
        let mut self_elems = self.elems.iter();
        let mut other_elems = other.elems.iter();
        loop {
            match (self_elems.next(), other_elems.next()) {
                (None, _) => {
                    return true;
                }
                (Some(..), None) => {
                    return false;
                }
                (Some(a), Some(b)) if a == b => {
                    // Exact match: continue.
                }
                (Some(PatternPrefixElem::Op(op1)), Some(PatternPrefixElem::OpWithArgs(op2)))
                    if op1 == op2 =>
                {
                    // Pattern has only op, but subject has op-with-args, with matching opcodes.
                    // Skip subtree of match subject.
                    PatternPrefix::skip_subtree(&mut other_elems);
                }
                (Some(PatternPrefixElem::Any), Some(PatternPrefixElem::Op(..))) => {
                    // Continue.
                }
                (Some(PatternPrefixElem::Any), Some(PatternPrefixElem::OpWithArgs(..))) => {
                    // Skip subtree of match subject.
                    PatternPrefix::skip_subtree(&mut other_elems);
                }
                _ => {
                    return false;
                }
            }
        }
    }
}

/// A pool of pattern prefixes that shares storage.
pub struct PatternPrefixPool {
    elems: Vec<PatternPrefixElem>,
}

impl PatternPrefixPool {
    /// Create a new pool.
    pub fn new() -> PatternPrefixPool {
        PatternPrefixPool { elems: vec![] }
    }

    /// Create a new builder for a pattern within this pool.
    pub fn build<'a>(&'a mut self) -> PatternPrefixBuilder<'a> {
        PatternPrefixBuilder::new(self)
    }

    /// Given a pattern range (the result of a builder session), get a pattern prefix.
    pub fn get<'a>(&'a self, range: &PatternPrefixRange) -> PatternPrefix<'a> {
        PatternPrefix::new(Cow::from(&self.elems[range.start..range.end]))
    }

    /// Checkpoint the current table state, allowing pattern-prefixes allocated after the
    /// checkpoint to be freed with `rewind()`.
    pub fn checkpoint(&self) -> usize {
        self.elems.len()
    }

    /// Rewind to a previous checkpoint. All pattern-prefixes built since the checkpoint are
    /// invalidated, and their PatternPrefixRanges must not be passed to `get()`.
    pub fn rewind(&mut self, checkpoint: usize) {
        self.elems.truncate(checkpoint);
    }
}

/// A builder that builds a new pattern prefix in a pool.
pub struct PatternPrefixBuilder<'a> {
    pool: &'a mut PatternPrefixPool,
    start_idx: usize,
    depth: usize,
}

/// A handle to a pattern prefix in a pool; used to get an immutable reference once all builders
/// are done with the pool.
#[derive(Clone, Debug)]
pub struct PatternPrefixRange {
    start: usize,
    end: usize,
}

impl<'a> PatternPrefixBuilder<'a> {
    /// Create a new builder.
    pub fn new(pool: &'a mut PatternPrefixPool) -> PatternPrefixBuilder<'a> {
        let start_idx = pool.elems.len();
        PatternPrefixBuilder {
            pool,
            start_idx,
            depth: 0,
        }
    }

    /// Add an opcode without args to the pattern prefix.
    pub fn opcode(mut self, op: Opcode) -> Self {
        self.pool.elems.push(PatternPrefixElem::Op(op));
        self
    }
    /// Add an opcode with args to the pattern prefix.
    pub fn opcode_with_args(mut self, op: Opcode) -> Self {
        self.pool.elems.push(PatternPrefixElem::OpWithArgs(op));
        self.depth += 1;
        self
    }
    /// Add an any-opcode wildcard to the pattern prefix.
    pub fn any(mut self) -> Self {
        self.pool.elems.push(PatternPrefixElem::Any);
        self
    }
    /// End the args for an opcode.
    pub fn args_end(mut self) -> Self {
        assert!(self.depth > 0);
        self.depth -= 1;
        self.pool.elems.push(PatternPrefixElem::EndArgs);
        self
    }

    /// Build the pattern prefix and return a range handle, suitable for fetching a `PatternPrefix`
    /// once the user is done building into the pool.
    pub fn build(mut self) -> PatternPrefixRange {
        while self.depth > 0 {
            self = self.args_end();
        }
        PatternPrefixRange {
            start: self.start_idx,
            end: self.pool.elems.len(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pattern_prefix_match() {
        let mut pool = PatternPrefixPool::new();

        let pat1 = pool
            .build()
            // Add(Sub(..), Mul(..))
            .opcode_with_args(Opcode::Iadd)
            .opcode(Opcode::Isub)
            .opcode(Opcode::Imul)
            .args_end()
            .build();

        let pat2 = pool.build().opcode(Opcode::Iadd).build();

        let pat3 = pool
            .build()
            .opcode_with_args(Opcode::Iadd)
            .any()
            .opcode(Opcode::Imul)
            .args_end()
            .build();

        let pat4 = pool.build().any().build();

        let pat5 = pool.build().opcode(Opcode::Isub).build();

        let pat6 = pool
            .build()
            .opcode_with_args(Opcode::Iadd)
            .opcode_with_args(Opcode::Isub)
            .opcode_with_args(Opcode::Iconst)
            .opcode_with_args(Opcode::Iadd)
            .args_end()
            .args_end()
            .args_end()
            .opcode(Opcode::Imul)
            .args_end()
            .build();

        let target1 = pool
            .build()
            .opcode_with_args(Opcode::Iadd)
            .opcode_with_args(Opcode::Isub)
            .opcode(Opcode::Iconst)
            .opcode(Opcode::Iconst)
            .args_end()
            .opcode(Opcode::Imul)
            .args_end()
            .build();

        let pat1 = pool.get(&pat1);
        let pat2 = pool.get(&pat2);
        let pat3 = pool.get(&pat3);
        let pat4 = pool.get(&pat4);
        let pat5 = pool.get(&pat5);
        let pat6 = pool.get(&pat6);
        let target1 = pool.get(&target1);

        assert!(pat1.matches(&target1));
        assert!(pat2.matches(&target1));
        assert!(pat3.matches(&target1));
        assert!(pat4.matches(&target1));
        assert!(!pat5.matches(&target1));
        assert!(!pat6.matches(&target1));
    }
}
