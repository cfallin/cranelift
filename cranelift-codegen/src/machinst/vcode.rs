//! This implements the VCode container: a CFG of Insts that have been lowered.
//!
//! VCode is virtual-register code. An instruction in VCode is almost a machine
//! instruction; however, its register slots can refer to virtual registers in
//! addition to real machine registers.
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
//! See the main module comment in `mod.rs` for more details on the VCode-based
//! backend pipeline.

use crate::binemit::SizeCodeSink;
use crate::ir;
use crate::machinst::*;

use regalloc::Function as RegallocFunction;
use regalloc::Set as RegallocSet;
use regalloc::{BlockIx, InstIx, InstRegUses, MyRange, RegAllocResult, RegClass};

use alloc::boxed::Box;
use alloc::vec::Vec;
use smallvec::SmallVec;
use std::fmt;
use std::iter;
use std::ops::Index;

/// Index referring to an instruction in VCode.
pub type InsnIndex = u32;
/// Index referring to a basic block in VCode.
pub type BlockIndex = u32;

/// VCodeInst wraps all requirements for a MachInst to be in VCode: it must be
/// a `MachInst` and it must be able to emit itself at least to a `SizeCodeSink`.
pub trait VCodeInst: MachInst + MachInstEmit<SizeCodeSink> {}
impl<I: MachInst + MachInstEmit<SizeCodeSink>> VCodeInst for I {}

/// A function in "VCode" (virtualized-register code) form, after lowering.
/// This is essentially a standard CFG of basic blocks, where each basic block
/// consists of lowered instructions produced by the machine-specific backend.
pub struct VCode<I: VCodeInst> {
    /// Function liveins.
    liveins: RegallocSet<RealReg>,

    /// Function liveouts.
    liveouts: RegallocSet<RealReg>,

    /// Lowered machine instructions in order corresponding to the original IR.
    insts: Vec<I>,

    /// Entry block.
    entry: BlockIndex,

    /// Block instruction indices.
    block_ranges: Vec<(InsnIndex, InsnIndex)>,

    /// Block successors: index range in the successor-list below.
    block_succ_range: Vec<(usize, usize)>,

    /// Block successor lists, concatenated into one Vec. The `block_succ_range`
    /// list of tuples above gives (start, end) ranges within this list that
    /// correspond to each basic block's successors.
    block_succs: Vec<BlockIndex>,

    /// Block indices by Ebb.
    block_by_ebb: SecondaryMap<ir::Ebb, BlockIndex>,

    /// Order of block IDs in final generated code.
    final_block_order: Vec<BlockIndex>,

    /// Final block offsets. Computed during branch finalization and used
    /// during emission.
    final_block_offsets: Vec<CodeOffset>,

    /// Size of code, according for block layout / alignment.
    code_size: CodeOffset,
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
pub struct VCodeBuilder<I: VCodeInst> {
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

impl<I: VCodeInst> VCodeBuilder<I> {
    /// Create a new VCodeBuilder.
    pub fn new(abi: &dyn ABIBody<I>) -> VCodeBuilder<I> {
        VCodeBuilder {
            vcode: VCode::new(abi),
            bb_insns: SmallVec::new(),
            ir_inst_insns: SmallVec::new(),
            succ_start: 0,
        }
    }

    /// Return the underlying Ebb-to-BlockIndex map.
    pub fn blocks_by_ebb(&self) -> &SecondaryMap<ir::Ebb, BlockIndex> {
        &self.vcode.block_by_ebb
    }

    /// Initialize the Ebb-to-BlockIndex map. Returns the first free
    /// BlockIndex.
    pub fn init_ebb_map(&mut self, blocks: &[ir::Ebb]) -> BlockIndex {
        let mut bindex: BlockIndex = 0;
        for ebb in blocks.iter() {
            self.vcode.block_by_ebb[*ebb] = bindex;
            bindex += 1;
        }
        bindex
    }

    /// Get the BlockIndex for an Ebb.
    pub fn ebb_to_bindex(&self, ebb: ir::Ebb) -> BlockIndex {
        self.vcode.block_by_ebb[ebb]
    }

    /// Set the current block as the entry block.
    pub fn set_entry(&mut self, block: BlockIndex) {
        self.vcode.entry = block;
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
    pub fn end_bb(&mut self) -> BlockIndex {
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

        block_num
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

fn block_ranges(indices: &[InstIx], len: usize) -> Vec<(usize, usize)> {
    let v = indices
        .iter()
        .map(|iix| iix.get() as usize)
        .chain(iter::once(len))
        .collect::<Vec<usize>>();
    v.windows(2).map(|p| (p[0], p[1])).collect()
}

fn is_redundant_move<I: VCodeInst>(insn: &I) -> bool {
    if let Some((to, from)) = insn.is_move() {
        to == from
    } else {
        false
    }
}

fn inst_size<I: VCodeInst>(insn: &I) -> usize {
    let mut sizesink = SizeCodeSink::new();
    insn.emit(&mut sizesink);
    sizesink.size()
}

fn is_trivial_jump_block<I: VCodeInst>(vcode: &VCode<I>, block: BlockIndex) -> Option<BlockIndex> {
    let range = vcode.block_insns(BlockIx::new(block));
    println!(
        "is_trivial_jump_block: block {} has len {}",
        block,
        range.len()
    );
    if range.len() != 1 {
        return None;
    }
    let insn = range.first();
    println!(
        " -> only insn is: {:?} with terminator {:?}",
        vcode.get_insn(insn),
        vcode.get_insn(insn).is_term()
    );
    match vcode.get_insn(insn).is_term() {
        MachTerminator::Uncond(target) => Some(target),
        _ => None,
    }
}

fn look_through_trivial_jumps<I: VCodeInst>(vcode: &VCode<I>, block: BlockIndex) -> BlockIndex {
    let mut b = block;
    while let Some(next) = is_trivial_jump_block(vcode, b) {
        b = next;
    }
    b
}

impl<I: VCodeInst> VCode<I> {
    /// New empty VCode.
    fn new(abi: &dyn ABIBody<I>) -> VCode<I> {
        VCode {
            liveins: abi.liveins(),
            liveouts: abi.liveouts(),
            insts: vec![],
            entry: 0,
            block_ranges: vec![],
            block_succ_range: vec![],
            block_succs: vec![],
            block_by_ebb: SecondaryMap::with_default(0),
            final_block_order: vec![],
            final_block_offsets: vec![],
            code_size: 0,
        }
    }

    /// Get the entry block.
    pub fn entry(&self) -> BlockIndex {
        self.entry
    }

    /// Get the number of blocks. Block indices will be in the range `0 ..
    /// (self.num_blocks() - 1)`.
    pub fn num_blocks(&self) -> usize {
        self.block_ranges.len()
    }

    /// Get the successors for a block.
    pub fn succs(&self, block: BlockIndex) -> &[BlockIndex] {
        let (start, end) = self.block_succ_range[block as usize];
        &self.block_succs[start..end]
    }

    /// Take the results of register allocation, with a sequence of
    /// instructions including spliced fill/reload/move instructions, and replace
    /// the VCode with them.
    pub fn replace_insns_from_regalloc(&mut self, result: RegAllocResult<Self>) {
        self.final_block_order = compute_final_block_order(self);
        // We want to move instructions over in final block order, using the new
        // block-start map given by the regalloc.
        let block_ranges: Vec<(usize, usize)> =
            block_ranges(result.target_map.elems(), result.insns.len());
        let mut final_insns = vec![];
        let mut final_block_ranges: Vec<(InsnIndex, InsnIndex)> =
            iter::repeat((0, 0)).take(self.num_blocks()).collect();

        for block in &self.final_block_order {
            let (start, end) = block_ranges[*block as usize].clone();
            let final_start = final_insns.len() as InsnIndex;
            for i in start..end {
                let insn = &result.insns[i];
                if is_redundant_move(insn) {
                    continue;
                }
                final_insns.push(insn.clone());
            }
            let final_end = final_insns.len() as InsnIndex;
            final_block_ranges[*block as usize] = (final_start, final_end);
        }

        self.insts = final_insns;
        self.block_ranges = final_block_ranges;
    }

    /// Removes redundant branches, rewriting targets to point directly to the
    /// ultimate block at the end of a chain of trivial one-target jumps.
    pub fn remove_redundant_branches(&mut self) {
        // For each block, compute the actual target block, looking through all
        // blocks with single-target jumps.
        //
        // The outer `Option` indicates whether the dynamic programming algorithm
        // has computed this value yet. The inner `Option` indicates whether there
        // is a redirect.
        let block_rewrites: Vec<BlockIndex> = (0..self.num_blocks() as u32)
            .map(|bix| look_through_trivial_jumps(self, bix))
            .collect();
        let deleted: Vec<bool> = block_rewrites
            .iter()
            .enumerate()
            .map(|(i, target)| i != *target as usize)
            .collect();

        println!(
            "remove_redundant_branches: block_rewrites = {:?}",
            block_rewrites
        );
        for block in 0..self.num_blocks() as u32 {
            for insn in self.block_insns(BlockIx::new(block)) {
                self.get_insn_mut(insn)
                    .with_block_rewrites(&block_rewrites[..]);
            }
        }

        let block_order = std::mem::replace(&mut self.final_block_order, vec![]);
        self.final_block_order = block_order
            .into_iter()
            .filter(|b| !deleted[*b as usize])
            .collect();
    }

    /// Mutate branch instructions to (i) lower two-way condbrs to one-way,
    /// depending on fallthrough; and (ii) use concrete offsets.
    pub fn finalize_branches(&mut self) {
        // Compute fallthrough block, indexed by block.
        let num_final_blocks = self.final_block_order.len();
        let mut block_fallthrough: Vec<Option<BlockIndex>> = vec![None; self.num_blocks()];
        for i in 0..(num_final_blocks - 1) {
            let from = self.final_block_order[i];
            let to = self.final_block_order[i + 1];
            block_fallthrough[from as usize] = Some(to);
        }

        // Pass over VCode instructions and finalize two-way branches into
        // one-way branches with fallthrough.
        for block in 0..self.num_blocks() {
            let next_block = block_fallthrough[block];
            let (start, end) = self.block_ranges[block].clone();

            for iix in start..end {
                let insn = &mut self.insts[iix as usize];
                insn.with_fallthrough_block(next_block);
            }
        }

        // Compute block offsets.
        let mut offset = 0;
        let mut block_offsets = vec![0; self.num_blocks()];
        for block in &self.final_block_order {
            offset = I::align_basic_block(offset);
            block_offsets[*block as usize] = offset;
            let (start, end) = self.block_ranges[*block as usize].clone();
            for iix in start..end {
                offset += inst_size(&self.insts[iix as usize]) as CodeOffset;
            }
        }

        // Update branches with known block offsets. This looks like the
        // traversal above, but (i) does not update block_offsets, rather uses
        // it (so forward references are now possible), and (ii) mutates the
        // instructions.
        offset = 0;
        for block in &self.final_block_order {
            offset = I::align_basic_block(offset);
            let (start, end) = self.block_ranges[*block as usize].clone();
            for iix in start..end {
                self.insts[iix as usize].with_block_offsets(offset, &block_offsets[..]);
                offset += inst_size(&self.insts[iix as usize]) as CodeOffset;
            }
        }

        self.final_block_offsets = block_offsets;
        self.code_size = offset;
    }

    /// Get the total size of the code when emitted.
    pub fn code_size(&self) -> usize {
        // TODO: size of any ConstantData?
        self.code_size as usize
    }

    /// Emit the instructions to the given sink.
    pub fn emit<CS: CodeSink>(&self, cs: &mut CS)
    where
        I: MachInstEmit<CS>,
    {
        for block in &self.final_block_order {
            let new_offset = I::align_basic_block(cs.offset());
            while new_offset > cs.offset() {
                // Pad with NOPs up to the aligned block offset.
                let nop = I::gen_nop((new_offset - cs.offset()) as usize);
                nop.emit(cs);
            }
            assert!(cs.offset() == new_offset);

            let (start, end) = self.block_ranges[*block as usize].clone();
            for iix in start..end {
                self.insts[iix as usize].emit(cs);
            }
        }

        // TODO: constant pool at end of code? Or in rodata?
    }
}

impl<I: VCodeInst> RegallocFunction for VCode<I> {
    type Inst = I;

    fn insns(&self) -> &[I] {
        &self.insts[..]
    }

    fn insns_mut(&mut self) -> &mut [I] {
        &mut self.insts[..]
    }

    fn get_insn(&self, insn: InstIx) -> &I {
        &self.insts[insn.get() as usize]
    }

    fn get_insn_mut(&mut self, insn: InstIx) -> &mut I {
        &mut self.insts[insn.get() as usize]
    }

    fn blocks(&self) -> MyRange<BlockIx> {
        MyRange::new(BlockIx::new(0), self.block_ranges.len())
    }

    fn entry_block(&self) -> BlockIx {
        BlockIx::new(self.entry)
    }

    fn block_insns(&self, block: BlockIx) -> MyRange<InstIx> {
        let (start, end) = self.block_ranges[block.get() as usize];
        MyRange::new(InstIx::new(start), (end - start) as usize)
    }

    fn block_succs(&self, block: BlockIx) -> Vec<BlockIx> {
        let (start, end) = self.block_succ_range[block.get() as usize];
        self.block_succs[start..end]
            .iter()
            .cloned()
            .map(BlockIx::new)
            .collect()
    }

    fn is_ret(&self, insn: InstIx) -> bool {
        match self.insts[insn.get() as usize].is_term() {
            MachTerminator::Ret => true,
            _ => false,
        }
    }

    fn get_regs(&self, insn: &I) -> InstRegUses {
        insn.get_regs()
    }

    fn map_regs(
        insn: &mut I,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        insn.map_regs(pre_map, post_map);
    }

    fn is_move(&self, insn: &I) -> Option<(Reg, Reg)> {
        insn.is_move()
    }

    fn get_spillslot_size(&self, regclass: RegClass, _vreg: VirtualReg) -> u32 {
        I::get_spillslot_size(regclass)
    }

    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, _vreg: VirtualReg) -> I {
        I::gen_spill(to_slot, from_reg)
    }

    fn gen_reload(&self, to_reg: RealReg, from_slot: SpillSlot, _vreg: VirtualReg) -> I {
        I::gen_reload(to_reg, from_slot)
    }

    fn gen_move(&self, to_reg: RealReg, from_reg: RealReg, _vreg: VirtualReg) -> I {
        I::gen_move(to_reg.to_reg(), from_reg.to_reg())
    }

    fn maybe_direct_reload(&self, insn: &I, reg: VirtualReg, slot: SpillSlot) -> Option<I> {
        insn.maybe_direct_reload(reg, slot)
    }

    fn func_liveins(&self) -> RegallocSet<RealReg> {
        self.liveins.clone()
    }

    fn func_liveouts(&self) -> RegallocSet<RealReg> {
        self.liveouts.clone()
    }
}

// N.B.: Debug impl assumes that VCode has already been through all compilation
// passes, and so has a final block order and offsets.

impl<I: VCodeInst> fmt::Debug for VCode<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "VCode {{")?;
        writeln!(f, "  Entry block: {}", self.entry)?;
        writeln!(f, "  Final block order: {:?}", self.final_block_order)?;

        for block in 0..self.num_blocks() {
            writeln!(f, "Block {}:", block,)?;
            for succ in self.succs(block as BlockIndex) {
                writeln!(f, "    (successor: Block {})", succ)?;
            }
            let (start, end) = self.block_ranges[block].clone();
            writeln!(f, "    (instruction range: {} .. {})", start, end)?;
            for inst in start..end {
                writeln!(f, "  Inst {}: {:?}", inst, self.insts[inst as usize])?;
            }
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}
