//! Lowering rules for ARM64.

// TODO: handle multiple-output instructions properly:
// - add/sub: carry/borrow variants
// - isplit, vsplit
// - call with multiple results

use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{Ebb, InstructionData, Opcode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::*;

use smallvec::SmallVec;

pub struct Arm64LowerBackend {}

fn op_to_aluop(op: Opcode, ty: Type) -> Option<ALUOp> {
    match (op, ty) {
        (Opcode::Iadd, I32) => Some(ALUOp::Add32),
        (Opcode::Iadd, I64) => Some(ALUOp::Add64),
        (Opcode::Isub, I32) => Some(ALUOp::Sub32),
        (Opcode::Isub, I64) => Some(ALUOp::Sub64),
        _ => None,
    }
}

fn is_alu_op(op: Opcode, ctrl_typevar: Type) -> bool {
    op_to_aluop(op, ctrl_typevar).is_some()
}

/// A lowering result: register, register-shift, register-extend.  An SSA value can always be
/// lowered into one of these options; the register form is the fallback.
#[derive(Clone, Debug)]
enum ResultRSE {
    Reg(MachReg),
    RegShift(MachReg, ShiftOpAndAmt),
    RegExtend(MachReg, ExtendOp),
}

/// A lowering result: register, register-shift, register-extend, or 12-bit immediate form.
/// An SSA value can always be lowered into one of these options; the register form is the
/// fallback.
#[derive(Clone, Debug)]
enum ResultRSEImm12 {
    Reg(MachReg),
    RegShift(MachReg, ShiftOpAndAmt),
    RegExtend(MachReg, ExtendOp),
    Imm12(Imm12),
}

impl ResultRSEImm12 {
    fn from_rse(rse: ResultRSE) -> ResultRSEImm12 {
        match rse {
            ResultRSE::Reg(r) => ResultRSEImm12::Reg(r),
            ResultRSE::RegShift(r, s) => ResultRSEImm12::RegShift(r, s),
            ResultRSE::RegExtend(r, e) => ResultRSEImm12::RegExtend(r, e),
        }
    }
}

/// A lowering result: register, register-shift, register-extend, or logical immediate form.
/// An SSA value can always be lowered into one of these options; the register form is the
/// fallback.
#[derive(Clone, Debug)]
enum ResultRSEImmLogic {
    Reg(MachReg),
    RegShift(MachReg, ShiftOpAndAmt),
    RegExtend(MachReg, ExtendOp),
    ImmLogic(ImmLogic),
}

impl ResultRSEImmLogic {
    fn from_rse(rse: ResultRSE) -> ResultRSEImmLogic {
        match rse {
            ResultRSE::Reg(r) => ResultRSEImmLogic::Reg(r),
            ResultRSE::RegShift(r, s) => ResultRSEImmLogic::RegShift(r, s),
            ResultRSE::RegExtend(r, e) => ResultRSEImmLogic::RegExtend(r, e),
        }
    }
}

/// Identifier for a particular output of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InsnOutput {
    insn: IRInst,
    output: usize,
}

/// Identifier for a particular input of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InsnInput {
    insn: IRInst,
    input: usize,
}

/// Producer of a value: either a previous instruction's output, or a register that will be
/// codegen'd separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsnInputSource {
    Output(InsnOutput),
    Reg(MachReg),
}

impl InsnInputSource {
    fn as_output(self) -> Option<InsnOutput> {
        match self {
            InsnInputSource::Output(o) => Some(o),
            _ => None,
        }
    }
}

/// Context passed to all lowering functions.
type Ctx<'a> = &'a mut dyn LowerCtx<Inst>;

fn get_input<'a>(ctx: Ctx<'a>, output: InsnOutput, num: usize) -> InsnInput {
    assert!(num <= ctx.num_inputs(output.insn));
    InsnInput {
        insn: output.insn,
        input: num,
    }
}

/// Convert an instruction input to a producing instruction's output if possible (in same BB), or a
/// register otherwise.
fn input_source<'a>(ctx: Ctx<'a>, input: InsnInput) -> InsnInputSource {
    if let Some((input_inst, result_num)) = ctx.input_inst(input.insn, input.input) {
        let out = InsnOutput {
            insn: input_inst,
            output: result_num,
        };
        InsnInputSource::Output(out)
    } else {
        let reg = ctx.input(input.insn, input.input);
        InsnInputSource::Reg(reg)
    }
}

/// Lower an instruction output to a 64-bit constant, if possible.
fn output_to_const<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<u64> {
    if out.output > 0 {
        None
    } else {
        let inst_data = ctx.data(out.insn);
        match inst_data {
            &InstructionData::UnaryImm { opcode, imm } => {
                // Only has Into for i64; we use u64 elsewhere, so we cast.
                let imm: i64 = imm.into();
                Some(imm as u64)
            }
            &InstructionData::UnaryIeee32 { opcode, imm } => Some(imm.bits() as u64),
            &InstructionData::UnaryIeee64 { opcode, imm } => Some(imm.bits()),
            _ => None,
        }
    }
}

/// Lower an instruction output to a constant register-shift amount, if possible.
fn output_to_shiftimm<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<ShiftOpShiftImm> {
    output_to_const(ctx, out).and_then(ShiftOpShiftImm::maybe_from_shift)
}

/// Lower an instruction input to a reg.
fn input_to_reg<'a>(ctx: Ctx<'a>, input: InsnInput) -> MachReg {
    ctx.input(input.insn, input.input)
}

/// Lower an instruction output to a reg.
fn output_to_reg<'a>(ctx: Ctx<'a>, out: InsnOutput) -> MachReg {
    ctx.output(out.insn, out.output)
}

/// Lower an instruction output to a reg, reg/shift, or reg/extend operand.  This does not actually
/// codegen the source instruction; it just uses the vreg into which the source instruction will
/// generate its value.
fn output_to_rse<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSE {
    assert!(out.output <= ctx.num_outputs(out.insn));
    let insn = out.insn;
    let op = ctx.data(out.insn).opcode();
    let out_ty = ctx.output_ty(out.insn, out.output);

    if op == Opcode::Ishl {
        let shiftee = get_input(ctx, out, 0);
        let shift_amt = get_input(ctx, out, 1);

        // Can we get the shift amount as an immediate?
        if let Some(out) = input_source(ctx, shiftee).as_output() {
            if let Some(shiftimm) = output_to_shiftimm(ctx, out) {
                ctx.dec_use(insn);
                let reg = input_to_reg(ctx, shiftee);
                return ResultRSE::RegShift(reg, ShiftOpAndAmt::new(ShiftOp::LSL, shiftimm));
            }
        }
    }

    // Is this a zero-extend or sign-extend and can we handle that with a register-mode operator?
    if (op == Opcode::Uextend || op == Opcode::Sextend) && out_ty == I64 {
        let sign_extend = op == Opcode::Sextend;
        let extendee = get_input(ctx, out, 0);
        if let Some(out) = input_source(ctx, extendee).as_output() {
            let inner_ty = ctx.output_ty(out.insn, out.output);
            if inner_ty == I32 || inner_ty == I16 || inner_ty == I8 {
                let extendop = match (sign_extend, inner_ty) {
                    (true, I8) => ExtendOp::SXTB,
                    (false, I8) => ExtendOp::UXTB,
                    (true, I16) => ExtendOp::SXTH,
                    (false, I16) => ExtendOp::UXTH,
                    (true, I32) => ExtendOp::SXTW,
                    (false, I32) => ExtendOp::UXTW,
                    _ => unreachable!(),
                };
                let reg = input_to_reg(ctx, extendee);
                ctx.dec_use(insn);
                return ResultRSE::RegExtend(reg, extendop);
            }
        }
    }

    // Otherwise, just return the register corresponding to the output.
    ResultRSE::Reg(output_to_reg(ctx, out))
}

/// Lower an instruction output to a reg, reg/shift, reg/extend, or 12-bit immediate operand.
fn output_to_rse_imm12<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSEImm12 {
    if let Some(imm_value) = output_to_const(ctx, out) {
        if let Some(i) = Imm12::maybe_from_u64(imm_value) {
            ctx.dec_use(out.insn);
            return ResultRSEImm12::Imm12(i);
        }
    }

    ResultRSEImm12::from_rse(output_to_rse(ctx, out))
}

/// Lower an instruction output to a reg, reg/shift, reg/extend, or logic-immediate operand.
fn output_to_rse_immlogic<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSEImmLogic {
    if let Some(imm_value) = output_to_const(ctx, out) {
        if let Some(i) = ImmLogic::maybe_from_u64(imm_value) {
            ctx.dec_use(out.insn);
            return ResultRSEImmLogic::ImmLogic(i);
        }
    }

    ResultRSEImmLogic::from_rse(output_to_rse(ctx, out))
}

fn input_to_rse<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSE {
    match input_source(ctx, input) {
        InsnInputSource::Output(out) => output_to_rse(ctx, out),
        InsnInputSource::Reg(reg) => ResultRSE::Reg(reg),
    }
}

fn input_to_rse_imm12<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSEImm12 {
    match input_source(ctx, input) {
        InsnInputSource::Output(out) => output_to_rse_imm12(ctx, out),
        InsnInputSource::Reg(reg) => ResultRSEImm12::Reg(reg),
    }
}

fn input_to_rse_immlogic<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSEImmLogic {
    match input_source(ctx, input) {
        InsnInputSource::Output(out) => output_to_rse_immlogic(ctx, out),
        InsnInputSource::Reg(reg) => ResultRSEImmLogic::Reg(reg),
    }
}

fn alu_inst_imm12(op: ALUOp, rd: MachReg, rn: MachReg, rm: ResultRSEImm12) -> Inst {
    match rm {
        ResultRSEImm12::Imm12(imm12) => Inst::AluRRImm12 {
            alu_op: op,
            rd,
            rn,
            imm12,
        },
        ResultRSEImm12::Reg(rm) => Inst::AluRRR {
            alu_op: op,
            rd,
            rn,
            rm,
        },
        ResultRSEImm12::RegShift(rm, shiftop) => Inst::AluRRRShift {
            alu_op: op,
            rd,
            rn,
            rm,
            shiftop,
        },
        ResultRSEImm12::RegExtend(rm, extendop) => Inst::AluRRRExtend {
            alu_op: op,
            rd,
            rn,
            rm,
            extendop,
        },
    }
}

/// Lower the address of a load or store.
fn lower_address<'a>(ctx: Ctx<'a>, elem_ty: Type, addends: &[InsnInput], offset: i32) -> MemArg {
    // Handle one reg and offset that fits in immediate, if possible.
    if addends.len() == 1 {
        let reg = input_to_reg(ctx, addends[0]);
        if let Some(memarg) = MemArg::reg_maybe_offset(reg, offset as i64, elem_ty) {
            return memarg;
        }
    }

    // Handle two regs and a zero offset, if possible.
    if addends.len() == 2 && offset == 0 {
        let ra = input_to_reg(ctx, addends[0]);
        let rb = input_to_reg(ctx, addends[1]);
        return MemArg::BasePlusReg(ra, rb);
    }

    // Otherwise, generate add instructions.
    let addr = ctx.tmp(GPR);

    let const_data = u64_constant(offset as u64);
    ctx.emit(Inst::ULoad64 {
        rd: addr.clone(),
        mem: MemArg::label(MemLabel::ConstantData(const_data)),
    });

    for addend in addends {
        let reg = input_to_reg(ctx, *addend);
        ctx.emit(Inst::AluRRR {
            alu_op: ALUOp::Add64,
            rd: addr.clone(),
            rn: addr.clone(),
            rm: reg.clone(),
        });
    }

    MemArg::Base(addr)
}

/// Actually codegen an instruction's results into registers.
fn lower_insn_to_regs<'a>(ctx: Ctx<'a>, insn: IRInst) {
    let op = ctx.data(insn).opcode();
    let inputs: SmallVec<[InsnInput; 4]> = (0..ctx.num_inputs(insn))
        .map(|i| InsnInput { insn, input: i })
        .collect();
    let outputs: SmallVec<[InsnOutput; 2]> = (0..ctx.num_outputs(insn))
        .map(|i| InsnOutput { insn, output: i })
        .collect();
    let ty = if outputs.len() > 0 {
        Some(ctx.output_ty(insn, 0))
    } else {
        None
    };

    match op {
        Opcode::Iadd => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0]);
            let rm = input_to_rse_imm12(ctx, inputs[1]);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Add32, ALUOp::Add64);
            ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
        }
        Opcode::Isub => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0]);
            let rm = input_to_rse_imm12(ctx, inputs[1]);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
        }

        Opcode::Load
        | Opcode::Uload8
        | Opcode::Sload8
        | Opcode::Uload16
        | Opcode::Sload16
        | Opcode::Uload32
        | Opcode::Sload32
        | Opcode::LoadComplex
        | Opcode::Uload8Complex
        | Opcode::Sload8Complex
        | Opcode::Uload16Complex
        | Opcode::Sload16Complex
        | Opcode::Uload32Complex
        | Opcode::Sload32Complex => {
            let off = ldst_offset(ctx.data(insn)).unwrap();
            let elem_ty = match op {
                Opcode::Sload8 | Opcode::Uload8 => I8,
                Opcode::Sload16 | Opcode::Uload16 => I16,
                Opcode::Sload32 | Opcode::Uload32 => I32,
                Opcode::Load => I64,
                _ => unreachable!(),
            };

            let mem = lower_address(ctx, elem_ty, &inputs[..], off);
            let rd = output_to_reg(ctx, outputs[0]);

            ctx.emit(match op {
                Opcode::Uload8 | Opcode::Uload8Complex => Inst::ULoad8 { rd, mem },
                Opcode::Sload8 | Opcode::Sload8Complex => Inst::SLoad8 { rd, mem },
                Opcode::Uload16 | Opcode::Uload16Complex => Inst::ULoad16 { rd, mem },
                Opcode::Sload16 | Opcode::Sload16Complex => Inst::SLoad16 { rd, mem },
                Opcode::Uload32 | Opcode::Uload32Complex => Inst::ULoad32 { rd, mem },
                Opcode::Sload32 | Opcode::Sload32Complex => Inst::SLoad32 { rd, mem },
                Opcode::Load | Opcode::LoadComplex => Inst::ULoad64 { rd, mem },
                _ => unreachable!(),
            });
        }

        Opcode::Jump => {
            lower_ebb_param_moves(ctx, insn);
            let dest = branch_target(ctx.data(insn)).unwrap();
            ctx.emit(Inst::Jump { dest });
        }
        Opcode::Fallthrough => {
            lower_ebb_param_moves(ctx, insn);
        }
        Opcode::Brz => {
            let rt = input_to_reg(ctx, inputs[0]);
            let dest = branch_target(ctx.data(insn)).unwrap();
            ctx.emit(Inst::CondBrZ { dest, rt });
        }
        Opcode::Brnz => {
            let rt = input_to_reg(ctx, inputs[0]);
            let dest = branch_target(ctx.data(insn)).unwrap();
            ctx.emit(Inst::CondBrNZ { dest, rt });
        }

        // TODO: BrIcmp, Brif/icmp, Brff/icmp, jump tables, call, ret

        // TODO: cmp
        // TODO: more alu ops
        _ => unimplemented!(),
    }
}

fn choose_32_64(ty: Type, op32: ALUOp, op64: ALUOp) -> ALUOp {
    if ty == I32 {
        op32
    } else if ty == I64 {
        op64
    } else {
        panic!("type {} is not I32 or I64", ty);
    }
}

fn branch_target(data: &InstructionData) -> Option<Ebb> {
    match data {
        &InstructionData::BranchIcmp { destination, .. }
        | &InstructionData::Branch { destination, .. }
        | &InstructionData::BranchInt { destination, .. }
        | &InstructionData::Jump { destination, .. }
        | &InstructionData::BranchTable { destination, .. }
        | &InstructionData::BranchFloat { destination, .. } => Some(destination),
        _ => {
            assert!(!data.opcode().is_branch());
            None
        }
    }
}

fn ldst_offset(data: &InstructionData) -> Option<i32> {
    match data {
        &InstructionData::Load { offset, .. }
        | &InstructionData::StackLoad { offset, .. }
        | &InstructionData::LoadComplex { offset, .. }
        | &InstructionData::Store { offset, .. }
        | &InstructionData::StackStore { offset, .. }
        | &InstructionData::StoreComplex { offset, .. } => Some(offset.into()),
        _ => None,
    }
}

fn lower_ebb_param_moves<'a>(ctx: Ctx<'a>, ir_inst: IRInst) {
    assert!(ctx.data(ir_inst).opcode().is_branch());
    let target = branch_target(ctx.data(ir_inst)).unwrap();

    for i in 0..ctx.num_inputs(ir_inst) {
        let src = ctx.input(ir_inst, i);
        let dst = ctx.ebb_param(target, i);
        ctx.emit(Inst::RegallocMove { dst, src });
    }
}

impl LowerBackend for Arm64LowerBackend {
    type MInst = Inst;

    fn lower(&mut self, ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst) {
        lower_insn_to_regs(ctx, ir_inst);
    }
}
