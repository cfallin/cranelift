//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::binemit::CodeSink;
use crate::dce::has_side_effect;
use crate::entity::SecondaryMap;
use crate::ir::{Block, Function, Inst, InstructionData, Opcode, Type, Value, ValueDef};
use crate::isa::registers::RegUnit;
use crate::machinst::{
    ABIBody, BlockIndex, MachInst, MachInstEmit, VCode, VCodeBuilder, VCodeInst,
};
use crate::num_uses::NumUses;

use regalloc::Function as RegallocFunction;
use regalloc::{RealReg, Reg, RegClass, VirtualReg, Writable};

use alloc::boxed::Box;
use alloc::vec::Vec;
use log::debug;
use smallvec::SmallVec;
use std::ops::Range;

/// A context that machine-specific lowering code can use to emit lowered instructions. This is the
/// view of the machine-independent per-function lowering context that is seen by the machine
/// backend.
pub trait LowerCtx<I> {
    /// Get the instdata for a given IR instruction.
    fn data(&self, ir_inst: Inst) -> &InstructionData;
    /// Get the controlling type for a polymorphic IR instruction.
    fn ty(&self, ir_inst: Inst) -> Type;
    /// Emit a machine instruction.
    fn emit(&mut self, mach_inst: I);
    /// Indicate that an IR instruction has been merged, and so one of its
    /// uses is gone (replaced by uses of the instruction's inputs). This
    /// helps the lowering algorithm to perform on-the-fly DCE, skipping over
    /// unused instructions (such as immediates incorporated directly).
    fn merged(&mut self, from_inst: Inst);
    /// Get the producing instruction, if any, and output number, for the `idx`th input to the
    /// given IR instruction
    fn input_inst(&self, ir_inst: Inst, idx: usize) -> Option<(Inst, usize)>;
    /// Get the `idx`th input to the given IR instruction as a virtual register.
    fn input(&self, ir_inst: Inst, idx: usize) -> Reg;
    /// Get the `idx`th output of the given IR instruction as a virtual register.
    fn output(&self, ir_inst: Inst, idx: usize) -> Writable<Reg>;
    /// Get the number of inputs to the given IR instruction.
    fn num_inputs(&self, ir_inst: Inst) -> usize;
    /// Get the number of outputs to the given IR instruction.
    fn num_outputs(&self, ir_inst: Inst) -> usize;
    /// Get the type for an instruction's input.
    fn input_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get the type for an instruction's output.
    fn output_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass, ty: Type) -> Writable<Reg>;
    /// Get the number of block params.
    fn num_bb_params(&self, bb: Block) -> usize;
    /// Get the register for a block param.
    fn bb_param(&self, bb: Block, idx: usize) -> Reg;
    /// Get the register for a return value.
    fn retval(&self, idx: usize) -> Writable<Reg>;
}

/// A machine backend.
pub trait LowerBackend {
    /// The machine instruction type.
    type MInst: VCodeInst;

    /// Lower a single instruction. Instructions are lowered in reverse order.
    /// This function need not handle branches; those are always passed to
    /// `lower_branch_group` below.
    fn lower<C: LowerCtx<Self::MInst>>(&self, ctx: &mut C, inst: Inst);

    /// Lower a block-terminating group of branches (which together can be seen as one
    /// N-way branch), given a vcode BlockIndex for each target.
    fn lower_branch_group<C: LowerCtx<Self::MInst>>(
        &self,
        ctx: &mut C,
        insts: &[Inst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    );
}

/// Machine-independent lowering driver / machine-instruction container. Maintains a correspondence
/// from original Inst to MachInsts.
pub struct Lower<'a, I: VCodeInst> {
    // The function to lower.
    f: &'a Function,

    // Lowered machine instructions.
    vcode: VCodeBuilder<I>,

    // Number of active uses (minus `dec_use()` calls by backend) of each instruction.
    num_uses: SecondaryMap<Inst, u32>,

    // Mapping from `Value` (SSA value in IR) to virtual register.
    value_regs: SecondaryMap<Value, Reg>,

    // Return-value vregs.
    retval_regs: Vec<Reg>,

    // Next virtual register number to allocate.
    next_vreg: u32,
}

fn alloc_vreg(
    value_regs: &mut SecondaryMap<Value, Reg>,
    regclass: RegClass,
    value: Value,
    next_vreg: &mut u32,
) -> VirtualReg {
    if value_regs[value].get_index() == 0 {
        // default value in map.
        let v = *next_vreg;
        *next_vreg += 1;
        value_regs[value] = Reg::new_virtual(regclass, v);
    }
    value_regs[value].as_virtual_reg().unwrap()
}

impl<'a, I: VCodeInst> Lower<'a, I> {
    /// Prepare a new lowering context for the given IR function.
    pub fn new(f: &'a Function, abi: Box<dyn ABIBody<I>>) -> Lower<'a, I> {
        let mut vcode = VCodeBuilder::new(abi);

        let num_uses = NumUses::compute(f).take_uses();

        let mut next_vreg: u32 = 1;

        // Default register should never be seen, but the `value_regs` map needs a default and we
        // don't want to push `Option` everywhere. All values will be assigned registers by the
        // loops over block parameters and instruction results below.
        //
        // We do not use vreg 0 so that we can detect any unassigned register that leaks through.
        let default_register = Reg::new_virtual(RegClass::I32, 0);
        let mut value_regs = SecondaryMap::with_default(default_register);

        // Assign a vreg to each value.
        for bb in f.layout.blocks() {
            for param in f.dfg.block_params(bb) {
                let vreg = alloc_vreg(
                    &mut value_regs,
                    I::rc_for_type(f.dfg.value_type(*param)),
                    *param,
                    &mut next_vreg,
                );
                vcode.set_vreg_type(vreg, f.dfg.value_type(*param));
            }
            for inst in f.layout.block_insts(bb) {
                for result in f.dfg.inst_results(inst) {
                    let vreg = alloc_vreg(
                        &mut value_regs,
                        I::rc_for_type(f.dfg.value_type(*result)),
                        *result,
                        &mut next_vreg,
                    );
                    vcode.set_vreg_type(vreg, f.dfg.value_type(*result));
                }
            }
        }

        // Assign a vreg to each return value.
        let mut retval_regs = vec![];
        for ret in &f.signature.returns {
            let v = next_vreg;
            next_vreg += 1;
            let regclass = I::rc_for_type(ret.value_type);
            let vreg = Reg::new_virtual(regclass, v);
            retval_regs.push(vreg);
            vcode.set_vreg_type(vreg.as_virtual_reg().unwrap(), ret.value_type);
        }

        Lower {
            f,
            vcode,
            num_uses,
            value_regs,
            retval_regs,
            next_vreg,
        }
    }

    fn gen_arg_setup(&mut self) {
        if let Some(entry_bb) = self.f.layout.entry_block() {
            for (i, param) in self.f.dfg.block_params(entry_bb).iter().enumerate() {
                let reg = Writable::from_reg(self.value_regs[*param]);
                let insn = self.vcode.abi().gen_copy_arg_to_reg(i, reg);
                self.vcode.push(insn);
            }
        }
    }

    fn gen_retval_setup(&mut self) {
        for (i, reg) in self.retval_regs.iter().enumerate() {
            let insn = self.vcode.abi().gen_copy_reg_to_retval(i, *reg);
            self.vcode.push(insn);
        }
        let ret = self.vcode.abi().gen_ret();
        self.vcode.push(ret);
    }

    /// Lower the function.
    pub fn lower<B: LowerBackend<MInst = I>>(mut self, backend: &B) -> VCode<I> {
        // Work backward (reverse block order, reverse through each block), skipping insns with zero
        // uses.
        let mut bbs: SmallVec<[Block; 16]> = self.f.layout.blocks().collect();
        bbs.reverse();

        // This records a Block-to-BlockIndex map so that branch targets can be resolved.
        let mut next_bindex = self.vcode.init_bb_map(&bbs[..]);

        // Allocate a separate BlockIndex for each control-flow instruction so that we can create
        // the edge blocks later. Each entry for a control-flow inst is the edge block; the list
        // has (cf-inst, edge block, orig block) tuples.
        let mut edge_blocks_by_inst: SecondaryMap<Inst, Option<BlockIndex>> =
            SecondaryMap::with_default(None);
        let mut edge_blocks: Vec<(Inst, BlockIndex, Block)> = vec![];

        debug!("about to lower function: {:?}", self.f);
        debug!("bb map: {:?}", self.vcode.blocks_by_bb());

        for bb in bbs.iter() {
            for inst in self.f.layout.block_insts(*bb) {
                let op = self.f.dfg[inst].opcode();
                if op.is_branch() {
                    // Find the original target.
                    let instdata = &self.f.dfg[inst];
                    let next_bb = match op {
                        Opcode::Fallthrough | Opcode::FallthroughReturn => {
                            self.f.layout.next_block(*bb).unwrap()
                        }
                        Opcode::Trap | Opcode::IndirectJumpTableBr => unimplemented!(),
                        _ => branch_target(instdata).unwrap(),
                    };

                    // Allocate a new block number for the new target.
                    let edge_block = next_bindex;
                    next_bindex += 1;

                    edge_blocks_by_inst[inst] = Some(edge_block);
                    edge_blocks.push((inst, edge_block, next_bb));
                }
            }
        }

        for bb in bbs.iter() {
            debug!("lowering bb: {}", bb);

            // If this is a return block, produce the return value setup.
            let last_insn = self.f.layout.block_insts(*bb).last().unwrap();
            if self.f.dfg[last_insn].opcode().is_return() {
                self.gen_retval_setup();
                self.vcode.end_ir_inst();
            }

            // Find the branches at the end first, and process those, if any.
            let mut branches: SmallVec<[Inst; 2]> = SmallVec::new();
            let mut targets: SmallVec<[BlockIndex; 2]> = SmallVec::new();

            for inst in self.f.layout.block_insts(*bb).rev() {
                if edge_blocks_by_inst[inst].is_some() {
                    let target = edge_blocks_by_inst[inst].clone().unwrap();
                    branches.push(inst);
                    targets.push(target);
                } else {
                    // We've reached the end of the branches -- process all as a group, first.
                    if branches.len() > 0 {
                        let fallthrough = self.f.layout.next_block(*bb);
                        let fallthrough = fallthrough.map(|bb| self.vcode.bb_to_bindex(bb));
                        branches.reverse();
                        targets.reverse();
                        backend.lower_branch_group(
                            &mut self,
                            &branches[..],
                            &targets[..],
                            fallthrough,
                        );
                        self.vcode.end_ir_inst();
                        branches.clear();
                        targets.clear();
                    }

                    // Only codegen an instruction if it either has a side
                    // effect, or has at least one use of one of its results.
                    let num_uses = self.num_uses[inst];
                    let side_effect = has_side_effect(self.f, inst);
                    if side_effect || num_uses > 0 {
                        backend.lower(&mut self, inst);
                        self.vcode.end_ir_inst();
                    } else {
                        // If we're skipping the instruction, we need to dec-ref
                        // its arguments.
                        for arg in self.f.dfg.inst_args(inst) {
                            match self.f.dfg.value_def(*arg) {
                                ValueDef::Result(src_inst, _) => {
                                    self.dec_use(src_inst);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // There are possibly some branches left if the block contained only branches.
            if branches.len() > 0 {
                let fallthrough = self.f.layout.next_block(*bb);
                let fallthrough = fallthrough.map(|bb| self.vcode.bb_to_bindex(bb));
                branches.reverse();
                targets.reverse();
                backend.lower_branch_group(&mut self, &branches[..], &targets[..], fallthrough);
                self.vcode.end_ir_inst();
                branches.clear();
                targets.clear();
            }

            // If this is the entry block, produce the argument setup.
            if Some(*bb) == self.f.layout.entry_block() {
                self.gen_arg_setup();
                self.vcode.end_ir_inst();
            }

            let vcode_bb = self.vcode.end_bb();
            debug!("finished building bb: BlockIndex {}", vcode_bb);
            debug!("bb_to_bindex map says: {}", self.vcode.bb_to_bindex(*bb));
            assert!(vcode_bb == self.vcode.bb_to_bindex(*bb));
            if Some(*bb) == self.f.layout.entry_block() {
                self.vcode.set_entry(vcode_bb);
            }
        }

        // Now create the edge blocks, with phi lowering (block parameter copies).
        for (inst, edge_block, orig_block) in edge_blocks.into_iter() {
            debug!(
                "creating edge block: inst {}, edge_block {}, orig_block {}",
                inst, edge_block, orig_block
            );

            // Create a temporary for each block parameter.
            let phi_classes: Vec<(Type, RegClass)> = self
                .f
                .dfg
                .block_params(orig_block)
                .iter()
                .map(|p| self.f.dfg.value_type(*p))
                .map(|ty| (ty, I::rc_for_type(ty)))
                .collect();
            let phi_temps: Vec<Writable<Reg>> = phi_classes
                .into_iter()
                .map(|(ty, rc)| self.tmp(rc, ty)) // borrows `self` mutably.
                .collect();

            debug!("phi_temps = {:?}", phi_temps);

            // Create all of the phi uses (reads) from jump args to temps.
            for (i, arg) in self.f.dfg.inst_variable_args(inst).iter().enumerate() {
                debug!("jump arg {} is {}", i, arg);
                let src_reg = self.value_regs[*arg];
                let dst_reg = phi_temps[i];
                self.vcode.push(I::gen_move(dst_reg, src_reg));
            }

            // Create all of the phi defs (writes) from temps to block params.
            for (i, param) in self.f.dfg.block_params(orig_block).iter().enumerate() {
                debug!("bb arg {} is {}", i, param);
                let src_reg = phi_temps[i].to_reg();
                let dst_reg = Writable::from_reg(self.value_regs[*param]);
                self.vcode.push(I::gen_move(dst_reg, src_reg));
            }

            // Create the unconditional jump to the original target block.
            self.vcode
                .push(I::gen_jump(self.vcode.bb_to_bindex(orig_block)));

            // End the IR inst and block. (We lower this as if it were one IR instruction so that
            // we can emit machine instructions in forward order.)
            self.vcode.end_ir_inst();
            let blocknum = self.vcode.end_bb();
            assert!(blocknum == edge_block);
        }

        // Now that we've emitted all instructions into the VCodeBuilder, let's build the VCode.
        self.vcode.build()
    }

    /// Reduce the use-count of an IR instruction. Use this when, e.g., isel incorporates the
    /// computation of an input instruction directly, so that input instruction has one
    /// fewer use.
    fn dec_use(&mut self, ir_inst: Inst) {
        assert!(self.num_uses[ir_inst] > 0);
        self.num_uses[ir_inst] -= 1;
        debug!(
            "incref: ir_inst {} now has {} uses",
            ir_inst, self.num_uses[ir_inst]
        );
    }

    /// Increase the use-count of an IR instruction. Use this when, e.g., isel incorporates
    /// the computation of an input instruction directly, so that input instruction's
    /// inputs are now used directly by the merged instruction.
    fn inc_use(&mut self, ir_inst: Inst) {
        self.num_uses[ir_inst] += 1;
        debug!(
            "decref: ir_inst {} now has {} uses",
            ir_inst, self.num_uses[ir_inst]
        );
    }
}

impl<'a, I: VCodeInst> LowerCtx<I> for Lower<'a, I> {
    /// Get the instdata for a given IR instruction.
    fn data(&self, ir_inst: Inst) -> &InstructionData {
        &self.f.dfg[ir_inst]
    }

    /// Get the controlling type for a polymorphic IR instruction.
    fn ty(&self, ir_inst: Inst) -> Type {
        self.f.dfg.ctrl_typevar(ir_inst)
    }

    /// Emit a machine instruction.
    fn emit(&mut self, mach_inst: I) {
        self.vcode.push(mach_inst);
    }

    /// Indicate that a merge has occurred.
    fn merged(&mut self, from_inst: Inst) {
        debug!("merged: inst {}", from_inst);
        // First, inc-ref all inputs of `from_inst`, because they are now used
        // directly by `into_inst`.
        for arg in self.f.dfg.inst_args(from_inst) {
            match self.f.dfg.value_def(*arg) {
                ValueDef::Result(src_inst, _) => {
                    debug!(" -> inc-reffing src inst {}", src_inst);
                    self.inc_use(src_inst);
                }
                _ => {}
            }
        }
        // Then, dec-ref the merged instruction itself. It still retains references
        // to its arguments (inc-ref'd above). If its refcount has reached zero,
        // it will be skipped during emission and its args will be dec-ref'd at that
        // time.
        self.dec_use(from_inst);
    }

    /// Get the producing instruction, if any, and output number, for the `idx`th input to the
    /// given IR instruction.
    fn input_inst(&self, ir_inst: Inst, idx: usize) -> Option<(Inst, usize)> {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        match self.f.dfg.value_def(val) {
            ValueDef::Result(src_inst, result_idx) => Some((src_inst, result_idx)),
            _ => None,
        }
    }

    /// Get the `idx`th input to the given IR instruction as a virtual register.
    fn input(&self, ir_inst: Inst, idx: usize) -> Reg {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        self.value_regs[val]
    }

    /// Get the `idx`th output of the given IR instruction as a virtual register.
    fn output(&self, ir_inst: Inst, idx: usize) -> Writable<Reg> {
        let val = self.f.dfg.inst_results(ir_inst)[idx];
        Writable::from_reg(self.value_regs[val])
    }

    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass, ty: Type) -> Writable<Reg> {
        let v = self.next_vreg;
        self.next_vreg += 1;
        let vreg = Reg::new_virtual(rc, v);
        self.vcode.set_vreg_type(vreg.as_virtual_reg().unwrap(), ty);
        Writable::from_reg(vreg)
    }

    /// Get the number of inputs for the given IR instruction.
    fn num_inputs(&self, ir_inst: Inst) -> usize {
        self.f.dfg.inst_args(ir_inst).len()
    }

    /// Get the number of outputs for the given IR instruction.
    fn num_outputs(&self, ir_inst: Inst) -> usize {
        self.f.dfg.inst_results(ir_inst).len()
    }

    /// Get the type for an instruction's input.
    fn input_ty(&self, ir_inst: Inst, idx: usize) -> Type {
        self.f.dfg.value_type(self.f.dfg.inst_args(ir_inst)[idx])
    }

    /// Get the type for an instruction's output.
    fn output_ty(&self, ir_inst: Inst, idx: usize) -> Type {
        self.f.dfg.value_type(self.f.dfg.inst_results(ir_inst)[idx])
    }

    /// Get the number of block params.
    fn num_bb_params(&self, bb: Block) -> usize {
        self.f.dfg.block_params(bb).len()
    }

    /// Get the register for a block param.
    fn bb_param(&self, bb: Block, idx: usize) -> Reg {
        let val = self.f.dfg.block_params(bb)[idx];
        self.value_regs[val]
    }

    /// Get the register for a return value.
    fn retval(&self, idx: usize) -> Writable<Reg> {
        Writable::from_reg(self.retval_regs[idx])
    }
}

fn branch_target(inst: &InstructionData) -> Option<Block> {
    match inst {
        &InstructionData::Jump { destination, .. }
        | &InstructionData::Branch { destination, .. }
        | &InstructionData::BranchInt { destination, .. }
        | &InstructionData::BranchIcmp { destination, .. }
        | &InstructionData::BranchFloat { destination, .. } => Some(destination),
        &InstructionData::BranchTable { destination: _, .. } => unimplemented!(),
        _ => None,
    }
}
