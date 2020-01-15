//! This implements the VCode container: a CFG of Insts that have been lowered.
//!
//! VCode is virtual-register code. An instruction in VCode is almost a machine
//! instruction; however, its register slots can refer to virtual registers in
//! addition to real machine registers.
//!
//! A `VCode` is the result of lowering an `ir::Function`. The !
//! machine-dependent lowering pass, driven by the machinst framework traversing
//! the code, performs this transformation. The register allocator then
//! receives the vcode (wrapped in a `regalloc::Function` trait implementation
//! because the regalloc is abstracted into a general library) and rewrites the
//! instructions into ones that refer only to real registers. Finally, the
//! vcode-with-real-regs can be used by the machine-dependent backend to emit
//! machine code. So we have:
//!
//! |       ir::Function                     VCode<arch_backend::Inst>
//! |    (SSA IR,              [lower]         (machine-specific instruction
//! |     machine-            ------------>     instances, referring mostly to
//! |     independent ops)                      virtual registers)
//! |
//! |                                                     |
//! |                                                     | [regalloc]
//! |                         [binemit]                   v
//! |      machine code      <----------      VCode<arch_backend::Inst>
//! |                                           (machine insts with real regs)
//!
//!
//! VCode is structured with traditional basic blocks, and
//! each block must be terminated by an unconditional branch (one target), a
//! conditional branch (two targets), or a return (no targets). Note that this
//! slightly differs from the machine code of most ISAs: in most ISAs, a
//! conditional branch has one target (and the not-taken case falls through).
//! However, we expect that machine backends will elide branches to the following
//! block (i.e., zero-offset jumps), and will be able to codegen a branch-cond /
//! branch-uncond pair if *both* targets are not fallthrough. This allows us to
//! play with layout prior to final binary emission, as well, if we want.
//!
//! Finally, note that although this module does not verify it, we specify
//! VCode to have *split critical edges*. This enables insertion of e.g.
//! spills/fills/moves during regalloc. In the normal compilation pipeline, the
//! critical-edge splitting happens just before lowering, because it is a
//! machine-independent transform.

use crate::ir;
use crate::machinst::*;

use alloc::vec::Vec;
use smallvec::SmallVec;
use std::ops::Index;

/// Index referring to an instruction in VCode.
pub type InsnIndex = u32;
/// Index referring to a basic block in VCode.
pub type BlockIndex = u32;

/// A function in "VCode" (virtualized-register code) form, after lowering.
/// This is essentially a standard CFG of basic blocks, where each basic block
/// consists of lowered instructions produced by the machine-specific backend.
pub struct VCode<I: MachInst> {
    /// Lowered machine instructions in order corresponding to the original IR.
    insts: Vec<I>,

    /// Block instruction indices.
    block_ranges: Vec<(InsnIndex, InsnIndex)>,

    /// Block successors: index range in the successor-list below.
    block_succ_range: Vec<(usize, usize)>,

    /// Block successor lists, concatenated into one Vec. The `block_succ_range`
    //list of tuples above gives (start, end) ranges within this list that
    //correspond to each basic block's successors.
    block_succs: Vec<BlockIndex>,
}

/// A builder for a VCode function body. This builder is designed for the
/// lowering approach that we take: we traverse basic blocks in forward
/// (original IR) order, but within each basic block, we generate code from
/// bottom to top; and within each IR instruction that we visit in this reverse
/// order, we emit machine instructions in *forward* order again.
///
/// Hence, to produce the final instructions in proper order, we perform two
/// swaps.  First, the machine instructions (`I` instances) are produced in
/// forward order for an individual IR instruction. Then these are *reversed*
/// and concatenated to `bb_insns` at the end of the IR instruction lowering.
/// The `bb_insns` vec will thus contain all machine instructions for a basic
/// block, in reverse order. Finally, when we're done with a basic block, we
/// reverse the whole block's vec of instructions again, and concatenate onto
/// the VCode's insts.
pub struct VCodeBuilder<I: MachInst> {
    /// In-progress VCode.
    vcode: VCode<I>,

    /// Current basic block instructions, in reverse order (because blocks are
    /// built bottom-to-top).
    bb_insns: SmallVec<[I; 32]>,

    /// Current IR-inst instructions, in forward order.
    ir_inst_insns: SmallVec<[I; 4]>,

    /// Start of succs for the current block in the concatenated succs list.
    succ_start: usize,
}

impl<I: MachInst> VCodeBuilder<I> {
    /// Create a new VCodeBuilder.
    pub fn new() -> VCodeBuilder<I> {
        VCodeBuilder {
            vcode: VCode::new(),
            bb_insns: SmallVec::new(),
            ir_inst_insns: SmallVec::new(),
            succ_start: 0,
        }
    }

    /// End the current IR instruction. Must be called after pushing any
    /// instructions and prior to ending the basic block.
    pub fn end_ir_inst(&mut self) {
        while let Some(i) = self.ir_inst_insns.pop() {
            self.bb_insns.push(i);
        }
    }

    /// End the current basic block. Must be called after emitting vcode insts
    /// for IR insts and prior to ending the function (building the VCode).
    pub fn end_bb(&mut self) {
        assert!(self.ir_inst_insns.is_empty());
        let block_num = self.vcode.block_ranges.len() as BlockIndex;
        // Push the instructions.
        let start_idx = self.vcode.insts.len() as InsnIndex;
        while let Some(i) = self.bb_insns.pop() {
            self.vcode.insts.push(i);
        }
        let end_idx = self.vcode.insts.len() as InsnIndex;
        // Add the instruction index range to the list of blocks.
        self.vcode.block_ranges.push((start_idx, end_idx));
        // End the successors list.
        let succ_end = self.vcode.block_succs.len();
        self.vcode
            .block_succ_range
            .push((self.succ_start, succ_end));
        self.succ_start = succ_end;
    }

    /// Push an instruction for the current BB and current IR inst within the BB.
    pub fn push(&mut self, insn: I) {
        match insn.is_term() {
            MachTerminator::None | MachTerminator::Ret => {}
            MachTerminator::Uncond(target) => {
                self.vcode.block_succs.push(target);
            }
            MachTerminator::Cond(true_branch, false_branch) => {
                self.vcode.block_succs.push(true_branch);
                self.vcode.block_succs.push(false_branch);
            }
        }
        self.ir_inst_insns.push(insn);
    }

    /// Build the final VCode.
    pub fn build(self) -> VCode<I> {
        assert!(self.ir_inst_insns.is_empty());
        assert!(self.bb_insns.is_empty());
        self.vcode
    }
}

impl<I: MachInst> VCode<I> {
    /// New empty VCode.
    fn new() -> VCode<I> {
        VCode {
            insts: vec![],
            block_ranges: vec![],
            block_succ_range: vec![],
            block_succs: vec![],
        }
    }
}

// TODO: implementation of regalloc::Function.
