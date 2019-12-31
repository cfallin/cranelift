//! Lowering rules for ARM64.

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

fn lower_cflow(ctx: &mut dyn LowerCtx<Inst>, op: Opcode, ir_inst: IRInst) {
    unimplemented!()
}

fn lower_store(ctx: &mut dyn LowerCtx<Inst>, op: Opcode, ir_inst: IRInst) {
    unimplemented!()
}

bitflags! {
    /// The possible kind of result of an integer operation. It may be some type of immediate, or a
    /// register processed through a register shift or extend operator, or simply a register.
    /// `lower_value` takes a set of these kinds to indicate which are acceptable results.
    struct ResultKind: u8 {
        const Reg = 0x01;
        const RegShift = 0x02;
        const RegExtend = 0x04;
        const Imm12 = 0x08;
        const ImmLogic = 0x10;
        const ImmShift = 0x20;
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
        }
    }
}

fn lower_value(
    ctx: &mut dyn LowerCtx<Inst>,
    op: Opcode,
    ty: Type,
    ir_inst: IRInst,
    into: Option<MachReg>,
    kinds: ResultKind,
) -> ResultValue {
    unimplemented!()
}

fn lower_address(ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst, into: Option<MachReg>) -> MemArg {
    unimplemented!()
}

impl LowerBackend for Arm64LowerBackend {
    type MInst = Inst;

    fn lower(&mut self, ctx: &mut dyn LowerCtx<Inst>, ir_inst: IRInst, ctrl_typevar: Type) {
        let op = ctx.data(ir_inst).opcode();
        if op.is_branch() || op.is_call() || op.is_return() {
            lower_cflow(ctx, op, ir_inst);
        } else if op.can_store() {
            lower_store(ctx, op, ir_inst);
        } else {
            assert!(ctx.num_outputs(ir_inst) > 0);
            let out = ctx.output(ir_inst, 0);
            lower_value(ctx, op, ctrl_typevar, ir_inst, Some(out), ResultKind::Reg);
        }
    }
}
