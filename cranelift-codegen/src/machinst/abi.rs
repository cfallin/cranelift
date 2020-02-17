//! ABI definitions.

use crate::ir;
use crate::ir::StackSlot;
use crate::machinst::*;
use regalloc::{Reg, Set, SpillSlot, VirtualReg, Writable};

/// Trait implemented by an object that tracks ABI-related state (e.g., stack
/// layout) and can generate code while emitting the *body* of a function.
pub trait ABIBody<I: VCodeInst> {
    /// Get the liveins of the function.
    fn liveins(&self) -> Set<RealReg>;

    /// Get the liveouts of the function.
    fn liveouts(&self) -> Set<RealReg>;

    /// Number of arguments.
    fn num_args(&self) -> usize;

    /// Number of return values.
    fn num_retvals(&self) -> usize;

    /// Number of stack slots (not spill slots).
    fn num_stackslots(&self) -> usize;

    /// Generate an instruction which copies an argument to a destination
    /// register.
    fn gen_copy_arg_to_reg(&self, idx: usize, into_reg: Writable<Reg>) -> I;

    /// Generate an instruction which copies a source register to a return
    /// value slot.
    fn gen_copy_reg_to_retval(&self, idx: usize, from_reg: Reg) -> I;

    /// Generate a return instruction.
    fn gen_ret(&self) -> I;

    // -----------------------------------------------------------------
    // Every function above this line may only be called pre-regalloc.
    // Every function below this line may only be called post-regalloc.
    // `spillslots()` must be called before any other post-regalloc
    // function.
    // ----------------------------------------------------------------

    /// Update with the number of spillslots, post-regalloc.
    fn set_num_spillslots(&mut self, slots: usize);

    /// Update with the clobbered registers, post-regalloc.
    fn set_clobbered(&mut self, clobbered: Set<Writable<RealReg>>);

    /// Load from a stackslot.
    fn load_stackslot(
        &self,
        slot: StackSlot,
        offset: usize,
        ty: Type,
        into_reg: Writable<Reg>,
    ) -> I;

    /// Store to a stackslot.
    fn store_stackslot(&self, slot: StackSlot, offset: usize, ty: Type, from_reg: Reg) -> I;

    /// Load from a spillslot.
    fn load_spillslot(&self, slot: SpillSlot, ty: Type, into_reg: Writable<Reg>) -> I;

    /// Store to a spillslot.
    fn store_spillslot(&self, slot: SpillSlot, ty: Type, from_reg: Reg) -> I;

    /// Generate a prologue, post-regalloc. This should include any stack frame
    /// or other setup necessary to use the other methods (`load_arg`,
    /// `store_retval`, and spillslot accesses.)
    fn gen_prologue(&self) -> Vec<I>;

    /// Generate an epilogue, post-regalloc. Note that this must generate the
    /// actual return instruction (rather than emitting this in the lowering
    /// logic), because the epilogue code comes before the return and the two are
    /// likely closely related.
    fn gen_epilogue(&self) -> Vec<I>;

    /// Get the spill-slot size.
    fn get_spillslot_size(&self, rc: RegClass, ty: Type) -> u32;

    /// Generate a spill.
    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, ty: Type) -> I;

    /// Generate a reload (fill).
    fn gen_reload(&self, to_reg: Writable<RealReg>, from_slot: SpillSlot, ty: Type) -> I;
}

/// Trait implemented by an object that tracks ABI-related state and can
/// generate code while emitting a *call* to a function.
pub trait ABICall<I: VCodeInst> {
    // TODO.
}
