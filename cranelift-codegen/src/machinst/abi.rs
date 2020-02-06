//! ABI definitions.

use crate::ir;
use crate::machinst::*;
use regalloc::{Reg, Set, VirtualReg};

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

    /// Generate an argument load sequence, given a destination register.
    /// Note that this may store a fixup to be done once the final stack
    /// frame layout (including spill slots, etc.) is known, post-regalloc.
    ///
    /// Requires the `vcode` so that it can save the instruction indices for
    /// later fixups. Will generate instruction(s) as part of a single
    /// conceptual IR instruction and will not end the IR-instruction sequence.
    fn load_arg(&mut self, idx: usize, into_reg: Reg, vcode: &mut VCodeBuilder<I>);

    /// Generate a return-value store sequence, given a source register.
    /// Like `load_arg` above, this may store a fixup for post-regalloc.
    fn store_retval(&mut self, idx: usize, from_reg: Reg, vcode: &mut VCodeBuilder<I>);

    // -----------------------------------------------------------------
    // Every function above this line may only be called pre-regalloc.
    // Every function below this line may only be called post-regalloc.
    // `spillslots()` must be called before any other post-regalloc
    // function.
    // ----------------------------------------------------------------

    /// Update with the number of spillslots, post-regalloc.
    fn spillslots(&mut self, slots: usize);

    // TODO: spillslot accesses! These are ABI-dependent, not just
    // ISA-dependent.

    /// Generate a prologue, post-regalloc. This should include any stack frame
    /// or other setup necessary to use the other methods (`load_arg`,
    /// `store_retval`, and spillslot accesses.)
    fn gen_prologue(&self) -> Vec<I>;

    /// Generate an epilogue, post-regalloc. Note that this must generate the
    /// actual return instruction (rather than emitting this in the lowering
    /// logic), because the epilogue code comes before the return and the two are
    /// likely closely related.
    fn gen_epilogue(&self) -> Vec<I>;
}

/// Trait implemented by an object that tracks ABI-related state and can
/// generate code while emitting a *call* to a function.
pub trait ABICall<I: VCodeInst> {
    /// Store a value as an argument to the callee.
    fn store_arg(&mut self, idx: usize, from_reg: Reg, vcode: &mut VCodeBuilder<I>);

    /// Load a value as a retval from the callee.
    fn load_retval(&mut self, idx: usize, to_reg: Reg, vcode: &mut VCodeBuilder<I>);

    /// Generate the actual call.

    /// Get the clobbers of the function call (not including the args/retvals).
    fn clobbers(&self) -> Set<RealReg>;
}
