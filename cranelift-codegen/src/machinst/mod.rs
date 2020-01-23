//! This module exposes the machine-specific backend definition pieces.

use crate::binemit::{CodeSink, MemoryCodeSink, RelocSink, StackmapSink, TrapSink};
use crate::entity::EntityRef;
use crate::entity::SecondaryMap;
use crate::ir::ValueLocations;
use crate::ir::{DataFlowGraph, Function, Inst, Opcode, Type, Value};
use crate::isa::RegUnit;
use crate::result::CodegenResult;
use crate::settings::Flags;
use crate::HashMap;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::Sum;
use minira::Map as RegallocMap;
use minira::{RealReg, Reg, RegClass, SpillSlot, VirtualReg};
use smallvec::SmallVec;
use std::hash::Hash;

pub mod lower;
pub use lower::*;
pub mod vcode;
pub use vcode::*;
pub mod branch_splitting;
pub use branch_splitting::*;
pub mod compile;
pub use compile::*;

/// The mode in which a register is used or defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegMode {
    /// Read (used) by the instruction.
    Use,
    /// Written (defined) by the instruction.
    Def,
    /// Both read and written by the instruction.
    Modify,
}

/// A list of Regs used/def'd by a MachInst.
pub type MachInstRegs = SmallVec<[(Reg, RegMode); 4]>;

/// A machine instruction.
pub trait MachInst: Clone {
    /// Return the registers referenced by this machine instruction along with the modes of
    /// reference (use, def, modify).
    ///
    /// TODO: rework this to return the minira InstRegUses directly.
    fn regs(&self) -> MachInstRegs;

    /// Map virtual registers to physical registers using the given virt->phys
    /// maps corresponding to the program points prior to, and after, this instruction.
    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    );

    /// If this is a simple move, return the (source, destination) tuple of registers.
    fn is_move(&self) -> Option<(Reg, Reg)>;

    /// Is this a terminator (branch or ret)? If so, return its type
    /// (ret/uncond/cond) and target if applicable.
    fn is_term(&self) -> MachTerminator;

    /// Get the spill-slot size.
    fn get_spillslot_size(rc: RegClass) -> u32;

    /// Generate a spill.
    fn gen_spill(to_slot: SpillSlot, from_reg: RealReg) -> Self;

    /// Generate a reload (fill).
    fn gen_reload(to_reg: RealReg, from_slot: SpillSlot) -> Self;

    /// Generate a move.
    fn gen_move(to_reg: Reg, from_reg: Reg) -> Self;

    /// Possibly operate on a value directly in a spill-slot rather than a
    /// register. Useful if the machine has register-memory instruction forms
    /// (e.g., add directly from or directly to memory), like x86.
    fn maybe_direct_reload(&self, reg: VirtualReg, slot: SpillSlot) -> Option<Self>;

    /// Determine a register class to store the given CraneLift type.
    fn rc_for_type(ty: Type) -> RegClass;

    /// Generate a jump to another target. Used during lowering of
    /// control flow.
    fn gen_jump(target: BlockIndex) -> Self;

    /// Finalize branches once the block order (fallthrough) is known.
    fn with_fallthrough_block(&mut self, fallthrough_block: Option<BlockIndex>);

    /// Update instruction once block offsets are known.  These offsets are
    /// relative to the beginning of the function. `targets` is indexed by
    /// BlockIndex.
    fn with_block_offsets(&mut self, my_offset: usize, targets: &[usize]);

    /// Get the size of the instruction.
    fn size(&self) -> usize;
}

/// Describes a block terminator (not call) in the vcode. Because MachInsts /
/// vcode model machine code fairly directly (modulo the virtual registers), we
/// do not have a two-target conditional branch. Rather, the conditional form
/// falls through if not taken. A conditional branch should always be followed
/// by an unconditional branch; branches to the next block will be elided (to
/// allow fallthrough instead).
#[derive(Clone, Debug)]
pub enum MachTerminator {
    /// Not a terminator.
    None,
    /// A return instruction.
    Ret,
    /// An unconditional branch to another block.
    Uncond(BlockIndex),
    /// A conditional branch to one of two other blocks.
    Cond(BlockIndex, BlockIndex),
}

/// A map from virtual registers to physical registers.
pub type MachLocations = Vec<RegUnit>; // Indexed by virtual register number.

/// A trait describing the ability to encode a MachInst into binary machine code.
pub trait MachInstEmit<CS: CodeSink> {
    /// Emit the instruction.
    fn emit(&self, cs: &mut CS);
}

/// Top-level machine backend trait, which wraps all monomorphized code and
/// allows a virtual call from the machine-independent `Function::compile()`.
pub trait MachBackend {
    /// Compile the given function to memory. Consumes the function.
    fn compile_function_to_memory(
        &self,
        func: Function,
        relocs: &mut dyn RelocSink,
        traps: &mut dyn TrapSink,
        stackmaps: &mut dyn StackmapSink,
    ) -> CodegenResult<Vec<u8>>;

    /// Return flags for this backend.
    fn flags(&self) -> &Flags;
}
