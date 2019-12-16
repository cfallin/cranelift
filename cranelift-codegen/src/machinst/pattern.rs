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

pub struct LowerTable {
    prefix_pool: PatternPrefixPool,
}

/// An action to perform when matching a tree of ops. Returns `true` if codegen was successful.
/// Otherwise, another pattern/action should be used instead.
pub type LowerAction<Op, Arg> = for<'a> fn(
    ctx: &mut MachInstLowerCtx<'a, Op, Arg>,
    insts: &[&'a InstructionData],
    regs: &[MachReg],
    results: &[MachReg],
) -> bool;
