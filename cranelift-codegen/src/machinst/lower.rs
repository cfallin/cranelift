//! This module implements lowering (instruction selection) from Cranelift IR to machine
//! instructions with virtual registers, with lookup tables as built by the backend. This is
//! *almost* the final machine code, except for register allocation.

use crate::binemit::CodeSink;
use crate::entity::SecondaryMap;
use crate::ir::{Ebb, Function, Inst, InstructionData, Opcode, Type, Value, ValueDef};
use crate::isa::registers::RegUnit;
use crate::machinst::{
    ABIBody, BlockIndex, MachInst, MachInstEmit, MachInstRegs, VCode, VCodeBuilder, VCodeInst,
};
use crate::num_uses::NumUses;

use regalloc::Function as RegallocFunction;
use regalloc::{RealReg, Reg, RegClass, VirtualReg};

use alloc::boxed::Box;
use alloc::vec::Vec;
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
    /// Reduce the use-count of an IR instruction. Use this when, e.g., isel incorporates the
    /// computation of an input instruction directly.
    fn dec_use(&mut self, ir_inst: Inst);
    /// Get the producing instruction, if any, and output number, for the `idx`th input to the
    /// given IR instruction
    fn input_inst(&self, ir_inst: Inst, idx: usize) -> Option<(Inst, usize)>;
    /// Get the `idx`th input to the given IR instruction as a virtual register.
    fn input(&self, ir_inst: Inst, idx: usize) -> Reg;
    /// Get the `idx`th output of the given IR instruction as a virtual register.
    fn output(&self, ir_inst: Inst, idx: usize) -> Reg;
    /// Get the number of inputs to the given IR instruction.
    fn num_inputs(&self, ir_inst: Inst) -> usize;
    /// Get the number of outputs to the given IR instruction.
    fn num_outputs(&self, ir_inst: Inst) -> usize;
    /// Get the type for an instruction's input.
    fn input_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get the type for an instruction's output.
    fn output_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass) -> Reg;
    /// Get the number of EBB params.
    fn num_ebb_params(&self, ebb: Ebb) -> usize;
    /// Get the register for an EBB param.
    fn ebb_param(&self, ebb: Ebb, idx: usize) -> Reg;
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

    /// Lower the entry point of the function. This allows the machine backend
    /// an opportunity to emit any setup instructions (aside from the prologue),
    /// for example initially assigning args from ABI regs and/or emitting
    /// ghost instructions that define special (zero?) regs.
    fn lower_entry<C: LowerCtx<Self::MInst>>(&self, ctx: &mut C, ebb: Ebb);
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

    // Next virtual register number to allocate.
    next_vreg: u32,

    // Current IR instruction which we are lowering.
    cur_inst: Option<Inst>,
}

fn alloc_vreg(
    value_regs: &mut SecondaryMap<Value, Reg>,
    regclass: RegClass,
    value: Value,
    next_vreg: &mut u32,
) {
    if value_regs[value].get_index() == 0 {
        // default value in map.
        let v = *next_vreg;
        *next_vreg += 1;
        value_regs[value] = Reg::new_virtual(regclass, v);
    }
}

impl<'a, I: VCodeInst> Lower<'a, I> {
    /// Prepare a new lowering context for the given IR function.
    pub fn new(f: &'a Function, abi: Box<dyn ABIBody<I>>) -> Lower<'a, I> {
        let num_uses = NumUses::compute(f).take_uses();

        let mut next_vreg: u32 = 1;

        // Default register should never be seen, but the `value_regs` map needs a default and we
        // don't want to push `Option` everywhere. All values will be assigned registers by the
        // loops over EBB parameters and instruction results below.
        //
        // We do not use vreg 0 so that we can detect any unassigned register that leaks through.
        let default_register = Reg::new_virtual(RegClass::I32, 0);
        let mut value_regs = SecondaryMap::with_default(default_register);

        // Assign a vreg to each value.
        for ebb in f.layout.ebbs() {
            for param in f.dfg.ebb_params(ebb) {
                alloc_vreg(
                    &mut value_regs,
                    I::rc_for_type(f.dfg.value_type(*param)),
                    *param,
                    &mut next_vreg,
                );
            }
            for inst in f.layout.ebb_insts(ebb) {
                for arg in f.dfg.inst_args(inst) {
                    alloc_vreg(
                        &mut value_regs,
                        I::rc_for_type(f.dfg.value_type(*arg)),
                        *arg,
                        &mut next_vreg,
                    );
                }
                for result in f.dfg.inst_results(inst) {
                    alloc_vreg(
                        &mut value_regs,
                        I::rc_for_type(f.dfg.value_type(*result)),
                        *result,
                        &mut next_vreg,
                    );
                }
            }
        }

        Lower {
            f,
            vcode: VCodeBuilder::new(abi),
            num_uses,
            value_regs,
            next_vreg,
            cur_inst: None,
        }
    }

    /// Lower the function.
    pub fn lower<B: LowerBackend<MInst = I>>(mut self, backend: &B) -> VCode<I> {
        // Work backward (reverse EBB order, reverse through each EBB), skipping insns with zero
        // uses.
        let mut ebbs: SmallVec<[Ebb; 16]> = self.f.layout.ebbs().collect();
        ebbs.reverse();

        // This records an Ebb-to-BlockIndex map so that branch targets can be resolved.
        let mut next_bindex = self.vcode.init_ebb_map(&ebbs[..]);

        // Allocate a separate BlockIndex for each control-flow instruction so that we can create
        // the edge blocks later. Each entry for a control-flow inst is the edge block; the list
        // has (cf-inst, edge block, orig block) tuples.
        let mut edge_blocks_by_inst: SecondaryMap<Inst, Option<BlockIndex>> =
            SecondaryMap::with_default(None);
        let mut edge_blocks: Vec<(Inst, BlockIndex, Ebb)> = vec![];

        println!("about to lower function: {:?}", self.f);
        println!("ebb map: {:?}", self.vcode.blocks_by_ebb());

        for ebb in ebbs.iter() {
            for inst in self.f.layout.ebb_insts(*ebb) {
                let op = self.f.dfg[inst].opcode();
                if op.is_branch() {
                    // Find the original target.
                    let instdata = &self.f.dfg[inst];
                    let next_ebb = match op {
                        Opcode::Fallthrough | Opcode::FallthroughReturn => {
                            self.f.layout.next_ebb(*ebb).unwrap()
                        }
                        Opcode::Trap | Opcode::IndirectJumpTableBr => unimplemented!(),
                        _ => branch_target(instdata).unwrap(),
                    };

                    // Allocate a new block number for the new target.
                    let edge_block = next_bindex;
                    next_bindex += 1;

                    edge_blocks_by_inst[inst] = Some(edge_block);
                    edge_blocks.push((inst, edge_block, next_ebb));
                }
            }
        }

        for ebb in ebbs.iter() {
            println!("lowering ebb: {}", ebb);
            // Find the branches at the end first, and process those, if any.
            let mut branches: SmallVec<[Inst; 2]> = SmallVec::new();
            let mut targets: SmallVec<[BlockIndex; 2]> = SmallVec::new();

            for inst in self.f.layout.ebb_insts(*ebb).rev() {
                if edge_blocks_by_inst[inst].is_some() {
                    let target = edge_blocks_by_inst[inst].clone().unwrap();
                    branches.push(inst);
                    targets.push(target);
                } else {
                    // We've reached the end of the branches -- process all as a group, first.
                    if branches.len() > 0 {
                        let fallthrough = self.f.layout.next_ebb(*ebb);
                        let fallthrough = fallthrough.map(|ebb| self.vcode.ebb_to_bindex(ebb));
                        branches.reverse();
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

                    // Of instructions that produce results, only lower instructions that have not
                    // been marked as unused by all of their consumers.
                    let num_results = self.f.dfg.inst_results(inst).len();
                    let num_uses = self.num_uses[inst];
                    if num_results == 0 || num_uses > 0 {
                        self.start_inst(inst);
                        backend.lower(&mut self, inst);
                        self.end_inst();
                        self.vcode.end_ir_inst();
                    }
                }
            }

            // There are possibly some branches left if the block contained only branches.
            if branches.len() > 0 {
                let fallthrough = self.f.layout.next_ebb(*ebb);
                let fallthrough = fallthrough.map(|ebb| self.vcode.ebb_to_bindex(ebb));
                branches.reverse();
                backend.lower_branch_group(&mut self, &branches[..], &targets[..], fallthrough);
                self.vcode.end_ir_inst();
                branches.clear();
                targets.clear();
            }

            // If this is the entry block, invoke the backend's lower_entry hook.
            if Some(*ebb) == self.f.layout.entry_block() {
                backend.lower_entry(&mut self, *ebb);
                self.vcode.end_ir_inst();
            }

            let bb = self.vcode.end_bb();
            println!("finished building bb: BlockIndex {}", bb);
            println!("ebb_to_bindex map says: {}", self.vcode.ebb_to_bindex(*ebb));
            assert!(bb == self.vcode.ebb_to_bindex(*ebb));
            if Some(*ebb) == self.f.layout.entry_block() {
                self.vcode.set_entry(bb);
            }
        }

        // Now create the edge blocks, with phi lowering (block parameter copies).
        for (inst, edge_block, orig_block) in edge_blocks.into_iter() {
            println!(
                "creating edge block: inst {}, edge_block {}, orig_block {}",
                inst, edge_block, orig_block
            );
            // Create a temporary for each block parameter.
            let phi_classes: Vec<RegClass> = self
                .f
                .dfg
                .ebb_params(orig_block)
                .iter()
                .map(|p| self.f.dfg.value_type(*p))
                .map(I::rc_for_type)
                .collect();
            let phi_temps: Vec<Reg> = phi_classes
                .into_iter()
                .map(|rc| self.tmp(rc)) // borrows `self` mutably.
                .collect();

            println!("phi_temps = {:?}", phi_temps);

            // Create all of the phi uses (reads) from jump args to temps.
            for (i, arg) in self.f.dfg.inst_variable_args(inst).iter().enumerate() {
                println!("jump arg {} is {}", i, arg);
                let src_reg = self.value_regs[*arg];
                let dst_reg = phi_temps[i];
                self.vcode.push(I::gen_move(dst_reg, src_reg));
            }

            // Create all of the phi defs (writes) from temps to block params.
            for (i, param) in self.f.dfg.ebb_params(orig_block).iter().enumerate() {
                println!("ebb arg {} is {}", i, param);
                let src_reg = phi_temps[i];
                let dst_reg = self.value_regs[*param];
                self.vcode.push(I::gen_move(dst_reg, src_reg));
            }

            // Create the unconditional jump to the original target block.
            self.vcode
                .push(I::gen_jump(self.vcode.ebb_to_bindex(orig_block)));

            // End the IR inst and block. (We lower this as if it were one IR instruction so that
            // we can emit machine instructions in forward order.)
            self.vcode.end_ir_inst();
            let blocknum = self.vcode.end_bb();
            assert!(blocknum == edge_block);
        }

        // Now that we've emitted all instructions into the VCodeBuilder, let's build the VCode.
        self.vcode.build()
    }

    fn start_inst(&mut self, inst: Inst) {
        self.cur_inst = Some(inst);
    }

    fn end_inst(&mut self) {
        self.cur_inst = None;
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

    /// Reduce the use-count of an IR instruction. Use this when, e.g., isel incorporates the
    /// computation of an input instruction directly.
    fn dec_use(&mut self, ir_inst: Inst) {
        assert!(self.num_uses[ir_inst] > 0);
        self.num_uses[ir_inst] -= 1;
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
    fn output(&self, ir_inst: Inst, idx: usize) -> Reg {
        let val = self.f.dfg.inst_results(ir_inst)[idx];
        self.value_regs[val]
    }

    /// Get a new temp.
    fn tmp(&mut self, rc: RegClass) -> Reg {
        let v = self.next_vreg;
        self.next_vreg += 1;
        Reg::new_virtual(rc, v)
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

    /// Get the number of EBB params.
    fn num_ebb_params(&self, ebb: Ebb) -> usize {
        self.f.dfg.ebb_params(ebb).len()
    }

    /// Get the register for an EBB param.
    fn ebb_param(&self, ebb: Ebb, idx: usize) -> Reg {
        let val = self.f.dfg.ebb_params(ebb)[idx];
        self.value_regs[val]
    }
}

fn branch_target(inst: &InstructionData) -> Option<Ebb> {
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
