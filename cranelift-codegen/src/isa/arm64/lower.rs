//! Lowering rules for ARM64.

// TODO: handle multiple-output instructions properly:
// - add/sub: carry/borrow variants
// - isplit, vsplit
// - call with multiple results

use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::*;

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

fn is_cflow_op(op: Opcode) -> bool {
    false
}

fn lower_cflow(ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst) {
    unimplemented!()
}

fn lower_store(ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst) {
    unimplemented!()
}

bitflags! {
    /// The possible kind of result of an integer operation. It may be some type of immediate, or a
    /// register processed through a register shift or extend operator, or simply a register.
    /// `lower_value` takes a set of these kinds to indicate which are acceptable results.
    struct ResultKind: u32 {
        const Reg = 0x01;
        const RegShift = 0x02;
        const RegExtend = 0x04;
        const Imm12 = 0x08;
        const ImmLogic = 0x10;
        const ImmShift = 0x20;
        const ImmRegShift = 0x40;
    }
}

#[derive(Clone, Debug)]
enum ResultValue {
    Reg(MachReg),
    RegShift(MachReg, ShiftOpAndAmt),
    RegExtend(MachReg, ExtendOpAndAmt),
    Imm12(Imm12),
    ImmLogic(ImmLogic),
    ImmShift(ImmShift),
    ImmRegShift(usize), // value up to ShiftOpAndAmt::MAX_SHIFT
    None,               // If none of the requested value modes could be used.
}

impl ResultValue {
    fn kind(&self) -> ResultKind {
        match self {
            &ResultValue::Reg(..) => ResultKind::Reg,
            &ResultValue::RegShift(..) => ResultKind::RegShift,
            &ResultValue::RegExtend(..) => ResultKind::RegExtend,
            &ResultValue::Imm12(..) => ResultKind::Imm12,
            &ResultValue::ImmLogic(..) => ResultKind::ImmLogic,
            &ResultValue::ImmShift(..) => ResultKind::ImmShift,
            &ResultValue::ImmRegShift(..) => ResultKind::ImmRegShift,
            &ResultValue::None => ResultKind::empty(),
        }
    }

    fn as_reg(&self) -> Option<MachReg> {
        match self {
            &ResultValue::Reg(ref r) => Some(r.clone()),
            _ => None,
        }
    }

    fn ok(&self) -> bool {
        match self {
            &ResultValue::None => false,
            _ => true,
        }
    }
}

fn maybe_immediate_value(inst_data: &InstructionData) -> Option<u64> {
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

fn lower_value(
    ctx: &mut dyn LowerCtx<Inst>,
    ir_inst: IRInst,
    result_num: usize,
    into: Option<MachReg>,
    kinds: ResultKind,
) -> ResultValue {
    let op = ctx.data(ir_inst).opcode();
    let out_ty = ctx.output_ty(ir_inst, result_num);

    // Can we produce an immediate?
    if let Some(imm64) = maybe_immediate_value(ctx.data(ir_inst)) {
        if kinds.contains(ResultKind::Imm12) {
            if let Some(i) = Imm12::maybe_from_u64(imm64) {
                return ResultValue::Imm12(i);
            }
        }
        if kinds.contains(ResultKind::ImmLogic) {
            if let Some(i) = ImmLogic::maybe_from_u64(imm64) {
                return ResultValue::ImmLogic(i);
            }
        }
        if kinds.contains(ResultKind::ImmShift) {
            if let Some(i) = ImmShift::maybe_from_u64(imm64) {
                return ResultValue::ImmShift(i);
            }
        }
        if kinds.contains(ResultKind::ImmRegShift) {
            if imm64 <= ShiftOpAndAmt::MAX_SHIFT as u64 {
                return ResultValue::ImmRegShift(imm64 as usize);
            }
        }
    }

    // Is this a shift and can we handle that with a register-mode operator?
    if kinds.contains(ResultKind::RegShift) && op == Opcode::Ishl {
        assert!(kinds.contains(ResultKind::Reg));
        // Can we get the shift amount as an immediate?
        match (ctx.input_inst(ir_inst, 0), ctx.input_inst(ir_inst, 1)) {
            (Some((lhs_inst, lhs_result)), Some((rhs_inst, rhs_result))) => {
                if let ResultValue::ImmRegShift(amt) =
                    lower_value(ctx, rhs_inst, rhs_result, None, ResultKind::ImmRegShift)
                {
                    let out_reg = lower_value(ctx, lhs_inst, lhs_result, into, ResultKind::Reg)
                        .as_reg()
                        .unwrap();
                    return ResultValue::RegShift(out_reg, ShiftOpAndAmt::new(ShiftOp::LSL, amt));
                }
            }
            _ => {}
        }
    }

    // Is this a zero-extend or sign-extend and can we handle that with a register-mode operator?
    if kinds.contains(ResultKind::RegExtend)
        && (op == Opcode::Uextend || op == Opcode::Sextend)
        && out_ty == I64
    {
        assert!(kinds.contains(ResultKind::Reg));
        let sign_extend = op == Opcode::Sextend;
        if let Some((inner_inst, inner_result)) = ctx.input_inst(ir_inst, 0) {
            let inner_ty = ctx.output_ty(inner_inst, inner_result);
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
                let out_reg = lower_value(ctx, inner_inst, inner_result, into, ResultKind::Reg)
                    .as_reg()
                    .unwrap();
                return ResultValue::RegExtend(out_reg, ExtendOpAndAmt::new(extendop, 0));
            }
        }
    }

    // Otherwise, generate into a register.
    if kinds.contains(ResultKind::Reg) {
        let dest_reg = into.unwrap_or_else(|| ctx.tmp(GPR));
        lower_value_reg(ctx, ir_inst, result_num, dest_reg);
        return ResultValue::Reg(dest_reg);
    }

    // No requested value-mode worked.
    ResultValue::None
}

fn lower_value_reg(
    ctx: &mut dyn LowerCtx<Inst>,
    ir_inst: IRInst,
    result_num: usize,
    dest_reg: MachReg,
) {
    // TODO
    unimplemented!()
}

fn lower_address(ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst, into: Option<MachReg>) -> MemArg {
    // TODO: allow for constant offsets and two-register forms.
    let result = lower_value(ctx, ir_inst, 0, into, ResultKind::Reg)
        .as_reg()
        .unwrap();
    MemArg::Base(result)
}

impl LowerBackend for Arm64LowerBackend {
    type MInst = Inst;

    fn lower(&mut self, ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst) {
        let op = ctx.data(ir_inst).opcode();
        if op.is_branch() || op.is_call() || op.is_return() {
            lower_cflow(ctx, ir_inst);
        } else if op.can_store() {
            lower_store(ctx, ir_inst);
        } else {
            assert!(ctx.num_outputs(ir_inst) > 0);
            for i in 0..ctx.num_outputs(ir_inst) {
                let out = ctx.output(ir_inst, i);
                lower_value(ctx, ir_inst, i, Some(out), ResultKind::Reg);
            }
        }
    }
}
