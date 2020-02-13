//! Compilation backend pipeline: optimized IR to VCode / binemit.

use crate::ir::Function;
use crate::machinst::*;

use log::debug;
use regalloc::{allocate_registers, RegAllocAlgorithm};

/// Compile the given function down to VCode with allocated registers, ready
/// for binary emission.
pub fn compile<B: LowerBackend>(
    f: &mut Function,
    b: &B,
    abi: Box<dyn ABIBody<B::MInst>>,
) -> VCode<B::MInst>
where
    B::MInst: ShowWithRRU,
{
    // This lowers the CL IR.
    let mut vcode = Lower::new(f, abi).lower(b);

    debug!("vcode from lowering:\n{:?}", vcode);

    // Perform register allocation.
    let result = allocate_registers(
        &mut vcode,
        RegAllocAlgorithm::Backtracking,
        &B::MInst::reg_universe(),
    )
    .expect("register allocation");

    // Reorder vcode into final order and copy out final instruction sequence
    // all at once. This also inserts prologues/epilogues.
    vcode.replace_insns_from_regalloc(result);

    debug!("vcode after regalloc:\n{:?}", vcode);

    vcode.remove_redundant_branches();

    debug!("vcode after removing redundant branches:\n{:?}", vcode);

    // Do final passes over code to finalize branches.
    vcode.finalize_branches();

    debug!("final VCode:\n{:?}", vcode);

    //println!("{}\n", vcode.show_rru(Some(&B::MInst::reg_universe())));

    vcode
}
