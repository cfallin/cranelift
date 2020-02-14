//! Lowering rules for ARM64.

#![allow(dead_code)]

use crate::ir::condcodes::IntCC;
use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{Block, InstructionData, Opcode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm64::inst::*;
use crate::isa::arm64::Arm64Backend;

use regalloc::{RealReg, Reg, RegClass, VirtualReg, Writable};

use smallvec::SmallVec;

//============================================================================
// Helpers: opcode conversions

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

//============================================================================
// Result enum types.
//
// Lowering of a given value results in one of these enums, depending on the
// modes in which we can accept the value.

/// A lowering result: register, register-shift, register-extend.  An SSA value can always be
/// lowered into one of these options; the register form is the fallback.
#[derive(Clone, Debug)]
enum ResultRSE {
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
    RegExtend(Reg, ExtendOp),
}

/// A lowering result: register, register-shift, register-extend, or 12-bit immediate form.
/// An SSA value can always be lowered into one of these options; the register form is the
/// fallback.
#[derive(Clone, Debug)]
enum ResultRSEImm12 {
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
    RegExtend(Reg, ExtendOp),
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
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
    RegExtend(Reg, ExtendOp),
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

//============================================================================
// Instruction input and output "slots".
//
// We use these types to refer to operand numbers, and result numbers, together
// with the associated instruction, in a type-safe way.

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
    Reg(Reg),
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

//============================================================================
// Lowering: convert instruction outputs to result types.

/// Lower an instruction output to a 64-bit constant, if possible.
fn output_to_const<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<u64> {
    if out.output > 0 {
        None
    } else {
        let inst_data = ctx.data(out.insn);
        if inst_data.opcode() == Opcode::Null {
            Some(0)
        } else {
            match inst_data {
                &InstructionData::UnaryImm { opcode: _, imm } => {
                    // Only has Into for i64; we use u64 elsewhere, so we cast.
                    let imm: i64 = imm.into();
                    Some(imm as u64)
                }
                &InstructionData::UnaryIeee32 { opcode: _, imm } => Some(imm.bits() as u64),
                &InstructionData::UnaryIeee64 { opcode: _, imm } => Some(imm.bits()),
                _ => None,
            }
        }
    }
}

/// Lower an instruction output to a constant register-shift amount, if possible.
fn output_to_shiftimm<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<ShiftOpShiftImm> {
    output_to_const(ctx, out).and_then(ShiftOpShiftImm::maybe_from_shift)
}

/// Lower an instruction input to a reg.
fn input_to_reg<'a>(ctx: Ctx<'a>, input: InsnInput) -> Reg {
    ctx.input(input.insn, input.input)
}

/// Lower an instruction output to a reg.
fn output_to_reg<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Writable<Reg> {
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
        let _shift_amt = get_input(ctx, out, 1);

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
        // TODO: This won't extend a value that comes from a block parameter,
        // since it has no associated instruction.
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
    ResultRSE::Reg(output_to_reg(ctx, out).to_reg())
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

fn alu_inst_imm12(op: ALUOp, rd: Writable<Reg>, rn: Reg, rm: ResultRSEImm12) -> Inst {
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

//============================================================================
// Lowering: addressing mode support. Takes instruction directly, rather
// than an `InsnInput`, to do more introspection.

/// Lower the address of a load or store.
fn lower_address<'a>(ctx: Ctx<'a>, elem_ty: Type, addends: &[InsnInput], offset: i32) -> MemArg {
    // TODO: support base_reg + scale * index_reg. For this, we would need to pattern-match shl or
    // mul instructions (Load/StoreComplex don't include scale factors).

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
    let addr = ctx.tmp(RegClass::I64, I64);

    // Get the const into a reg.
    lower_constant(ctx, addr.clone(), offset as u64);

    // Add each addend to the address.
    for addend in addends {
        let reg = input_to_reg(ctx, *addend);
        ctx.emit(Inst::AluRRR {
            alu_op: ALUOp::Add64,
            rd: addr.clone(),
            rn: addr.to_reg(),
            rm: reg.clone(),
        });
    }

    MemArg::Base(addr.to_reg())
}

fn lower_constant<'a>(ctx: Ctx<'a>, rd: Writable<Reg>, value: u64) {
    if let Some(imm) = MoveWideConst::maybe_from_u64(value) {
        // 16-bit immediate (shifted by 0, 16, 32 or 48 bits) in MOVZ
        ctx.emit(Inst::MovZ { rd, imm });
    } else if let Some(imm) = MoveWideConst::maybe_from_u64(!value) {
        // 16-bit immediate (shifted by 0, 16, 32 or 48 bits) in MOVN
        ctx.emit(Inst::MovN { rd, imm });
    } else if let Some(imml) = ImmLogic::maybe_from_u64(value) {
        // Weird logical-instruction immediate in ORI using zero register
        ctx.emit(Inst::AluRRImmLogic {
            alu_op: ALUOp::Orr64,
            rd,
            rn: zero_reg(),
            imml,
        });
    } else {
        // 64-bit constant in constant pool
        let const_data = u64_constant(value);
        ctx.emit(Inst::ULoad64 {
            rd,
            mem: MemArg::label(MemLabel::ConstantData(const_data)),
        });
    }
}

fn lower_condcode(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::SignedLessThan => Cond::Lt,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi,
        IntCC::UnsignedLessThanOrEqual => Cond::Ls,
        IntCC::UnsignedLessThan => Cond::Lo,
        IntCC::Overflow => Cond::Vs,
        IntCC::NotOverflow => Cond::Vc,
    }
}

//=============================================================================
// Top-level instruction lowering entry point, for one instruction.

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

    // TODO: legalize smaller integer types to larger ones before lowering.
    if let Some(ty) = ty {
        assert!(ty == I32 || ty == I64 || ty == B1);
        // TODO: FP and vector types.
        // TODO: I8, I16
        // TODO: B8, B16, B32, B64
    }

    match op {
        Opcode::Iconst | Opcode::Bconst | Opcode::F32const | Opcode::F64const | Opcode::Null => {
            let value = output_to_const(ctx, outputs[0]).unwrap();
            let rd = output_to_reg(ctx, outputs[0]);
            lower_constant(ctx, rd, value);
        }
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

        Opcode::UaddSat | Opcode::SaddSat => {
            // TODO: open-code a sequence: adds, then branch-on-no-overflow
            // over a load of the saturated value.
        }

        Opcode::UsubSat | Opcode::SsubSat => {
            // TODO
        }

        Opcode::Ineg => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = zero_reg();
            let rm = input_to_reg(ctx, inputs[0]);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
        }

        Opcode::Imul => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0]);
            let rm = input_to_reg(ctx, inputs[1]);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::MAdd32, ALUOp::MAdd64);
            ctx.emit(Inst::AluRRRR {
                alu_op,
                rd,
                rn,
                rm,
                ra: zero_reg(),
            });
        }

        Opcode::Umulhi | Opcode::Smulhi => {
            let _ty = ty.unwrap();
            // TODO
        }

        Opcode::Udiv | Opcode::Sdiv | Opcode::Urem | Opcode::Srem => {
            // TODO
        }

        Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::Bnot
        | Opcode::BandNot
        | Opcode::BorNot
        | Opcode::BxorNot => {
            // TODO
        }

        Opcode::Rotl | Opcode::Rotr => {
            // TODO
        }

        Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => {
            // TODO
        }

        Opcode::Bitrev => {
            // TODO
        }

        Opcode::Clz | Opcode::Cls | Opcode::Ctz | Opcode::Popcnt => {
            // TODO
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
                Opcode::Sload8 | Opcode::Uload8 | Opcode::Sload8Complex | Opcode::Uload8Complex => {
                    I8
                }
                Opcode::Sload16
                | Opcode::Uload16
                | Opcode::Sload16Complex
                | Opcode::Uload16Complex => I16,
                Opcode::Sload32
                | Opcode::Uload32
                | Opcode::Sload32Complex
                | Opcode::Uload32Complex => I32,
                Opcode::Load | Opcode::LoadComplex => I64,
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

        Opcode::Store
        | Opcode::Istore8
        | Opcode::Istore16
        | Opcode::Istore32
        | Opcode::StoreComplex
        | Opcode::Istore8Complex
        | Opcode::Istore16Complex
        | Opcode::Istore32Complex => {
            let off = ldst_offset(ctx.data(insn)).unwrap();
            let elem_ty = match op {
                Opcode::Istore8 | Opcode::Istore8Complex => I8,
                Opcode::Istore16 | Opcode::Istore16Complex => I16,
                Opcode::Istore32 | Opcode::Istore32Complex => I32,
                Opcode::Store | Opcode::StoreComplex => I64,
                _ => unreachable!(),
            };

            let mem = lower_address(ctx, elem_ty, &inputs[1..], off);
            let rd = input_to_reg(ctx, inputs[0]);

            ctx.emit(match op {
                Opcode::Istore8 | Opcode::Istore8Complex => Inst::Store8 { rd, mem },
                Opcode::Istore16 | Opcode::Istore16Complex => Inst::Store16 { rd, mem },
                Opcode::Istore32 | Opcode::Istore32Complex => Inst::Store32 { rd, mem },
                Opcode::Store | Opcode::StoreComplex => Inst::Store64 { rd, mem },
                _ => unreachable!(),
            });
        }

        Opcode::StackLoad => {
            // TODO
        }

        Opcode::StackStore => {
            // TODO
        }

        Opcode::StackAddr => {
            // TODO
        }

        Opcode::GlobalValue => {
            // TODO
        }

        Opcode::SymbolValue => {
            // TODO
        }

        Opcode::HeapAddr => {
            // TODO
        }

        Opcode::TableAddr => {
            // TODO
        }

        Opcode::Nop => {
            // Nothing.
        }

        Opcode::Select | Opcode::Selectif => {
            // TODO.
        }

        Opcode::Bitselect => {
            // TODO.
        }

        Opcode::IsNull | Opcode::IsInvalid | Opcode::Trueif | Opcode::Trueff => {
            // TODO.
        }

        Opcode::Copy => {
            // TODO
        }

        Opcode::Breduce | Opcode::Bextend | Opcode::Bint | Opcode::Bmask => {
            // TODO
        }

        Opcode::Ireduce | Opcode::Uextend | Opcode::Sextend | Opcode::Isplit | Opcode::Iconcat => {
            // TODO
        }

        Opcode::FallthroughReturn => {
            // What is this? The definition says it's a "special
            // instruction" meant to allow falling through into an
            // epilogue that will then return; that just sounds like a
            // normal fallthrough. TODO: Do we need to handle this
            // differently?
            unimplemented!();
        }

        Opcode::Return => {
            for (i, input) in inputs.iter().enumerate() {
                let reg = input_to_reg(ctx, *input);
                let retval_reg = ctx.retval(i);
                ctx.emit(Inst::gen_move(retval_reg, reg));
            }
            ctx.emit(Inst::Ret {});
        }

        Opcode::Icmp | Opcode::IcmpImm | Opcode::Ifcmp | Opcode::IfcmpImm => {
            // TODO
        }

        Opcode::JumpTableEntry => {
            // TODO
        }

        Opcode::JumpTableBase => {
            // TODO
        }

        Opcode::Debugtrap => {}

        Opcode::Trap => {}

        Opcode::Trapz | Opcode::Trapnz | Opcode::Trapif | Opcode::Trapff => {}

        Opcode::ResumableTrap => {}

        Opcode::Safepoint => {}

        Opcode::FuncAddr => {
            // TODO
        }

        Opcode::Call | Opcode::CallIndirect => {
            // TODO
        }

        Opcode::GetPinnedReg
        | Opcode::SetPinnedReg
        | Opcode::Spill
        | Opcode::Fill
        | Opcode::FillNop
        | Opcode::Regmove
        | Opcode::CopySpecial
        | Opcode::CopyToSsa
        | Opcode::CopyNop
        | Opcode::AdjustSpDown
        | Opcode::AdjustSpUpImm
        | Opcode::AdjustSpDownImm
        | Opcode::IfcmpSp
        | Opcode::Regspill
        | Opcode::Regfill => {
            panic!("Unused opcode should not be encountered.");
        }

        // TODO: cmp
        // TODO: more alu ops
        Opcode::Jump
        | Opcode::Fallthrough
        | Opcode::Brz
        | Opcode::Brnz
        | Opcode::BrIcmp
        | Opcode::Brif
        | Opcode::Brff
        | Opcode::IndirectJumpTableBr
        | Opcode::BrTable => {
            panic!("Branch opcode reached non-branch lowering logic!");
        }

        Opcode::Vconst
        | Opcode::Shuffle
        | Opcode::Vsplit
        | Opcode::Vconcat
        | Opcode::Vselect
        | Opcode::VanyTrue
        | Opcode::VallTrue
        | Opcode::Splat
        | Opcode::Insertlane
        | Opcode::Extractlane
        | Opcode::Bitcast
        | Opcode::RawBitcast
        | Opcode::ScalarToVector => {
            // TODO
            panic!("Vector ops not implemented.");
        }

        Opcode::Fcmp
        | Opcode::Ffcmp
        | Opcode::Fadd
        | Opcode::Fsub
        | Opcode::Fmul
        | Opcode::Fdiv
        | Opcode::Sqrt
        | Opcode::Fma
        | Opcode::Fneg
        | Opcode::Fabs
        | Opcode::Fcopysign
        | Opcode::Fmin
        | Opcode::Fmax
        | Opcode::Ceil
        | Opcode::Floor
        | Opcode::Trunc
        | Opcode::Nearest
        | Opcode::Fpromote
        | Opcode::Fdemote
        | Opcode::FcvtToUint
        | Opcode::FcvtToUintSat
        | Opcode::FcvtToSint
        | Opcode::FcvtToSintSat
        | Opcode::FcvtFromUint
        | Opcode::FcvtFromSint => {
            panic!("Floating point ops not implemented.");
        }

        Opcode::IaddImm
        | Opcode::ImulImm
        | Opcode::UdivImm
        | Opcode::SdivImm
        | Opcode::UremImm
        | Opcode::SremImm
        | Opcode::IrsubImm
        | Opcode::IaddCin
        | Opcode::IaddIfcin
        | Opcode::IaddCout
        | Opcode::IaddIfcout
        | Opcode::IaddCarry
        | Opcode::IaddIfcarry
        | Opcode::IsubBin
        | Opcode::IsubIfbin
        | Opcode::IsubBout
        | Opcode::IsubIfbout
        | Opcode::IsubBorrow
        | Opcode::IsubIfborrow
        | Opcode::BandImm
        | Opcode::BorImm
        | Opcode::BxorImm
        | Opcode::RotlImm
        | Opcode::RotrImm
        | Opcode::IshlImm
        | Opcode::UshrImm
        | Opcode::SshrImm => {
            panic!("ALU+imm and ALU+carry ops should not appear here!");
        }

        Opcode::X86Udivmodx
        | Opcode::X86Sdivmodx
        | Opcode::X86Umulx
        | Opcode::X86Smulx
        | Opcode::X86Cvtt2si
        | Opcode::X86Fmin
        | Opcode::X86Fmax
        | Opcode::X86Push
        | Opcode::X86Pop
        | Opcode::X86Bsr
        | Opcode::X86Bsf
        | Opcode::X86Pshufd
        | Opcode::X86Pshufb
        | Opcode::X86Pextr
        | Opcode::X86Pinsr
        | Opcode::X86Insertps
        | Opcode::X86Movsd
        | Opcode::X86Movlhps
        | Opcode::X86Psll
        | Opcode::X86Psrl
        | Opcode::X86Psra
        | Opcode::X86Ptest
        | Opcode::X86Pmaxs
        | Opcode::X86Pmaxu
        | Opcode::X86Pmins
        | Opcode::X86Pminu => {
            panic!("x86-specific opcode in ARM64 backend!");
        }
    }
}

//=============================================================================
// Helpers for instruction lowering.

fn choose_32_64(ty: Type, op32: ALUOp, op64: ALUOp) -> ALUOp {
    if ty == I32 {
        op32
    } else if ty == I64 {
        op64
    } else {
        panic!("type {} is not I32 or I64", ty);
    }
}

fn branch_target(data: &InstructionData) -> Option<Block> {
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

fn inst_condcode(data: &InstructionData) -> Option<IntCC> {
    match data {
        &InstructionData::IntCond { cond, .. }
        | &InstructionData::BranchIcmp { cond, .. }
        | &InstructionData::IntCompare { cond, .. }
        | &InstructionData::IntCondTrap { cond, .. }
        | &InstructionData::BranchInt { cond, .. }
        | &InstructionData::IntSelect { cond, .. }
        | &InstructionData::IntCompareImm { cond, .. } => Some(cond),
        _ => None,
    }
}

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for Arm64Backend {
    type MInst = Inst;

    fn lower<C: LowerCtx<Inst>>(&self, ctx: &mut C, ir_inst: IRInst) {
        lower_insn_to_regs(ctx, ir_inst);
    }

    fn lower_branch_group<C: LowerCtx<Inst>>(
        &self,
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    ) {
        // A block should end with at most two branches. The first may be a
        // conditional branch; a conditional branch can be followed only by an
        // unconditional branch or fallthrough. Otherwise, if only one branch,
        // it may be an unconditional branch, a fallthrough, a return, or a
        // trap. These conditions are verified by `is_ebb_basic()` during the
        // verifier pass.
        assert!(branches.len() <= 2);

        if branches.len() == 2 {
            // Must be a conditional branch followed by an unconditional branch.
            let op1 = ctx.data(branches[0]).opcode();
            let op2 = ctx.data(branches[1]).opcode();

            //println!(
            //    "lowering two-branch group: opcodes are {:?} and {:?}",
            //    op1, op2
            //);

            assert!(op2 == Opcode::Jump || op2 == Opcode::Fallthrough);
            let taken = BranchTarget::Block(targets[0]);
            let not_taken = match op2 {
                Opcode::Jump => BranchTarget::Block(targets[1]),
                Opcode::Fallthrough => BranchTarget::Block(fallthrough.unwrap()),
                _ => unreachable!(), // assert above.
            };
            match op1 {
                Opcode::Brz | Opcode::Brnz => {
                    let rt = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 0,
                        },
                    );
                    let kind = match op1 {
                        Opcode::Brz => CondBrKind::Zero(rt),
                        Opcode::Brnz => CondBrKind::NotZero(rt),
                        _ => unreachable!(),
                    };
                    ctx.emit(Inst::CondBr {
                        taken,
                        not_taken,
                        kind,
                    });
                }
                Opcode::BrIcmp => {
                    let rn = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 0,
                        },
                    );
                    let rm = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 1,
                        },
                    );
                    let ty = ctx.input_ty(branches[0], 0);
                    let alu_op = choose_32_64(ty, ALUOp::SubS32, ALUOp::SubS64);
                    let rd = writable_zero_reg();
                    ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
                    let cond = lower_condcode(inst_condcode(ctx.data(branches[0])).unwrap());
                    ctx.emit(Inst::CondBr {
                        taken,
                        not_taken,
                        kind: CondBrKind::Cond(cond),
                    });
                }

                // TODO: Brif/icmp, Brff/icmp, jump tables
                _ => unimplemented!(),
            }
        } else {
            assert!(branches.len() == 1);

            // Must be an unconditional branch or trap.
            let op = ctx.data(branches[0]).opcode();
            match op {
                Opcode::Jump => {
                    ctx.emit(Inst::Jump {
                        dest: BranchTarget::Block(targets[0]),
                    });
                }
                Opcode::Fallthrough => {
                    ctx.emit(Inst::Jump {
                        dest: BranchTarget::Block(targets[0]),
                    });
                }

                Opcode::Trap => unimplemented!(),

                _ => panic!("Unknown branch type!"),
            }
        }
    }
}
