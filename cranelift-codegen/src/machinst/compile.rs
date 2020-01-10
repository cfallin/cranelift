//! Compilation backend pipeline: optimized IR to VCode / binemit.

use crate::ir::Function;
use crate::machinst::*;

/// Compile the given function down to VCode with allocated registers, ready
/// for binary emission.
pub fn compile<B: LowerBackend>(f: &mut Function, b: &mut B) -> VCode<B::MInst> {
    // Step 1: split critical edges.
    split_critical_edges(f);

    // Step 2: lower instructions.
    let mut vcode = Lower::new(f).lower(b);

    // Step 3: register-allocate.
    // TODO.

    vcode
}
