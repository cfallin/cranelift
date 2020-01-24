//! Compilation backend pipeline: optimized IR to VCode / binemit.

use crate::ir::Function;
use crate::machinst::*;

use minira::{allocate_registers, RegAllocAlgorithm};

/// Compile the given function down to VCode with allocated registers, ready
/// for binary emission.
pub fn compile<B: LowerBackend>(f: &mut Function, b: &B) -> VCode<B::MInst> {
    // This lowers the CL IR.
    let mut vcode = Lower::new(f).lower(b);

    // Perform register allocation.
    let result = allocate_registers(
        &mut vcode,
        RegAllocAlgorithm::Backtracking,
        &B::MInst::reg_universe(),
    )
    .expect("register allocation");

    // Reorder vcode into final order and copy out final instruction sequence
    // all at once.
    vcode.replace_insns_from_regalloc(result);

    // Do final passes over code to finalize branches.
    vcode.finalize_branches();

    // TODO: do final passes over code to (i) resolve fallthroughs, (ii)
    // compute offsets, and (iii) resolve branch targets. End result is VCode
    // that has all registers and branch offsets resolved and instructions in
    // machine order, ready for emission.

    vcode
}
