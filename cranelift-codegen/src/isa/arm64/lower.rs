//! Lowering rules for ARM64.

#![allow(dead_code)]

use crate::ir::condcodes::IntCC;
use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{Block, InstructionData, Opcode, SourceLoc, TrapCode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm64::abi::*;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::Arm64Backend;

use regalloc::{RealReg, Reg, RegClass, VirtualReg, Writable};

use alloc::vec::Vec;
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

/// A lowering result: register, register-shift.  An SSA value can always be
/// lowered into one of these options; the register form is the fallback.
#[derive(Clone, Debug)]
enum ResultRS {
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
}

/// A lowering result: register, register-shift, register-extend.  An SSA value can always be
/// lowered into one of these options; the register form is the fallback.
#[derive(Clone, Debug)]
enum ResultRSE {
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
    RegExtend(Reg, ExtendOp),
}

impl ResultRSE {
    fn from_rs(rs: ResultRS) -> ResultRSE {
        match rs {
            ResultRS::Reg(r) => ResultRSE::Reg(r),
            ResultRS::RegShift(r, s) => ResultRSE::RegShift(r, s),
        }
    }
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

/// A lowering result: register, register-shift, or logical immediate form.
/// An SSA value can always be lowered into one of these options; the register form is the
/// fallback.
#[derive(Clone, Debug)]
enum ResultRSImmLogic {
    Reg(Reg),
    RegShift(Reg, ShiftOpAndAmt),
    ImmLogic(ImmLogic),
}

impl ResultRSImmLogic {
    fn from_rs(rse: ResultRS) -> ResultRSImmLogic {
        match rse {
            ResultRS::Reg(r) => ResultRSImmLogic::Reg(r),
            ResultRS::RegShift(r, s) => ResultRSImmLogic::RegShift(r, s),
        }
    }
}

/// A lowering result: register or immediate shift amount (arg to a shift op).
/// An SSA value can always be lowered into one of these options; the register form is the
/// fallback.
#[derive(Clone, Debug)]
enum ResultRegImmShift {
    Reg(Reg),
    ImmShift(ImmShift),
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

fn get_input<C: LowerCtx<Inst>>(ctx: &mut C, output: InsnOutput, num: usize) -> InsnInput {
    assert!(num <= ctx.num_inputs(output.insn));
    InsnInput {
        insn: output.insn,
        input: num,
    }
}

/// Convert an instruction input to a producing instruction's output if possible (in same BB), or a
/// register otherwise.
fn input_source<C: LowerCtx<Inst>>(ctx: &mut C, input: InsnInput) -> InsnInputSource {
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
fn output_to_const<C: LowerCtx<Inst>>(ctx: &mut C, out: InsnOutput) -> Option<u64> {
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
fn output_to_shiftimm<C: LowerCtx<Inst>>(ctx: &mut C, out: InsnOutput) -> Option<ShiftOpShiftImm> {
    output_to_const(ctx, out).and_then(ShiftOpShiftImm::maybe_from_shift)
}

/// How to handle narrow values loaded into registers; see note on `narrow_mode`
/// parameter to `input_to_*` below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NarrowValueMode {
    None,
    /// Zero-extend to 32 bits if original is < 32 bits.
    ZeroExtend32,
    /// Sign-extend to 32 bits if original is < 32 bits.
    SignExtend32,
    /// Zero-extend to 64 bits if original is < 64 bits.
    ZeroExtend64,
    /// Sign-extend to 64 bits if original is < 64 bits.
    SignExtend64,
}

impl NarrowValueMode {
    fn is_32bit(&self) -> bool {
        match self {
            NarrowValueMode::None => false,
            NarrowValueMode::ZeroExtend32 | NarrowValueMode::SignExtend32 => true,
            NarrowValueMode::ZeroExtend64 | NarrowValueMode::SignExtend64 => false,
        }
    }
}

/// Lower an instruction output to a reg.
fn output_to_reg<C: LowerCtx<Inst>>(ctx: &mut C, out: InsnOutput) -> Writable<Reg> {
    ctx.output(out.insn, out.output)
}

/// Lower an instruction input to a reg.
///
/// The given register will be extended appropriately, according to
/// `narrow_mode` and the input's type. If extended, the value is
/// always extended to 64 bits, for simplicity.
fn input_to_reg<C: LowerCtx<Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> Reg {
    let ty = ctx.input_ty(input.insn, input.input);
    let from_bits = ty_bits(ty) as u8;
    let in_reg = ctx.input(input.insn, input.input);
    match (narrow_mode, from_bits) {
        (NarrowValueMode::None, _) => in_reg,
        (NarrowValueMode::ZeroExtend32, n) if n < 32 => {
            let tmp = ctx.tmp(RegClass::I64, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rn: in_reg,
                signed: false,
                from_bits,
                to_bits: 32,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::SignExtend32, n) if n < 32 => {
            let tmp = ctx.tmp(RegClass::I64, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rn: in_reg,
                signed: true,
                from_bits,
                to_bits: 32,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::ZeroExtend32, n) | (NarrowValueMode::SignExtend32, n) if n == 32 => {
            in_reg
        }

        (NarrowValueMode::ZeroExtend64, n) if n < 64 => {
            let tmp = ctx.tmp(RegClass::I64, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rn: in_reg,
                signed: false,
                from_bits,
                to_bits: 64,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::SignExtend64, n) if n < 64 => {
            let tmp = ctx.tmp(RegClass::I64, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rn: in_reg,
                signed: true,
                from_bits,
                to_bits: 64,
            });
            tmp.to_reg()
        }
        (_, n) if n == 64 => in_reg,

        _ => panic!(
            "Unsupported input width: input ty {} bits {} mode {:?}",
            ty, from_bits, narrow_mode
        ),
    }
}

/// Lower an instruction input to a reg or reg/shift, or reg/extend operand.
/// This does not actually codegen the source instruction; it just uses the
/// vreg into which the source instruction will generate its value.
///
/// The `narrow_mode` flag indicates whether the consumer of this value needs
/// the high bits clear. For many operations, such as an add/sub/mul or any
/// bitwise logical operation, the low-bit results depend only on the low-bit
/// inputs, so e.g. we can do an 8 bit add on 32 bit registers where the 8-bit
/// value is stored in the low 8 bits of the register and the high 24 bits are
/// undefined. If the op truly needs the high N bits clear (such as for a
/// divide or a right-shift or a compare-to-zero), `narrow_mode` should be
/// set to `ZeroExtend` or `SignExtend` as appropriate, and the resulting
/// register will be provided the extended value.
fn input_to_rs<C: LowerCtx<Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> ResultRS {
    if let InsnInputSource::Output(out) = input_source(ctx, input) {
        let insn = out.insn;
        assert!(out.output <= ctx.num_outputs(insn));
        let op = ctx.data(insn).opcode();

        if op == Opcode::Ishl {
            let shiftee = get_input(ctx, out, 0);
            let shift_amt = get_input(ctx, out, 1);

            // Can we get the shift amount as an immediate?
            if let Some(shift_amt_out) = input_source(ctx, shift_amt).as_output() {
                if let Some(shiftimm) = output_to_shiftimm(ctx, shift_amt_out) {
                    let reg = input_to_reg(ctx, shiftee, narrow_mode);
                    ctx.merged(insn);
                    ctx.merged(shift_amt_out.insn);
                    return ResultRS::RegShift(reg, ShiftOpAndAmt::new(ShiftOp::LSL, shiftimm));
                }
            }
        }
    }

    ResultRS::Reg(input_to_reg(ctx, input, narrow_mode))
}

/// Lower an instruction input to a reg or reg/shift, or reg/extend operand.
/// This does not actually codegen the source instruction; it just uses the
/// vreg into which the source instruction will generate its value.
///
/// See note on `input_to_rs` for a description of `narrow_mode`.
fn input_to_rse<C: LowerCtx<Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> ResultRSE {
    if let InsnInputSource::Output(out) = input_source(ctx, input) {
        let insn = out.insn;
        assert!(out.output <= ctx.num_outputs(insn));
        let op = ctx.data(insn).opcode();
        let out_ty = ctx.output_ty(insn, out.output);
        let out_bits = ty_bits(out_ty);

        // If `out_ty` is smaller than 32 bits and we need to zero- or sign-extend,
        // then get the result into a register and return an Extend-mode operand on
        // that register.
        if narrow_mode != NarrowValueMode::None
            && ((narrow_mode.is_32bit() && out_bits < 32)
                || (!narrow_mode.is_32bit() && out_bits < 64))
        {
            let reg = output_to_reg(ctx, out);
            let extendop = match (narrow_mode, out_bits) {
                (NarrowValueMode::SignExtend32, 1) | (NarrowValueMode::SignExtend64, 1) => {
                    ExtendOp::SXTB
                }
                (NarrowValueMode::ZeroExtend32, 1) | (NarrowValueMode::ZeroExtend64, 1) => {
                    ExtendOp::UXTB
                }
                (NarrowValueMode::SignExtend32, 8) | (NarrowValueMode::SignExtend64, 8) => {
                    ExtendOp::SXTB
                }
                (NarrowValueMode::ZeroExtend32, 8) | (NarrowValueMode::ZeroExtend64, 8) => {
                    ExtendOp::UXTB
                }
                (NarrowValueMode::SignExtend32, 16) | (NarrowValueMode::SignExtend64, 16) => {
                    ExtendOp::SXTH
                }
                (NarrowValueMode::ZeroExtend32, 16) | (NarrowValueMode::ZeroExtend64, 16) => {
                    ExtendOp::UXTH
                }
                (NarrowValueMode::SignExtend64, 32) => ExtendOp::SXTW,
                (NarrowValueMode::ZeroExtend64, 32) => ExtendOp::UXTW,
                _ => unreachable!(),
            };
            return ResultRSE::RegExtend(reg.to_reg(), extendop);
        }

        // Is this a zero-extend or sign-extend and can we handle that with a register-mode operator?
        if op == Opcode::Uextend || op == Opcode::Sextend {
            assert!(out_bits == 32 || out_bits == 64);
            let sign_extend = op == Opcode::Sextend;
            let extendee = get_input(ctx, out, 0);
            let inner_ty = ctx.input_ty(extendee.insn, extendee.input);
            let inner_bits = ty_bits(inner_ty);
            assert!(inner_bits < out_bits);
            let extendop = match (sign_extend, inner_bits) {
                (true, 1) => ExtendOp::SXTB,
                (false, 1) => ExtendOp::UXTB,
                (true, 8) => ExtendOp::SXTB,
                (false, 8) => ExtendOp::UXTB,
                (true, 16) => ExtendOp::SXTH,
                (false, 16) => ExtendOp::UXTH,
                (true, 32) => ExtendOp::SXTW,
                (false, 32) => ExtendOp::UXTW,
                _ => unreachable!(),
            };
            let reg = input_to_reg(ctx, extendee, NarrowValueMode::None);
            ctx.merged(insn);
            return ResultRSE::RegExtend(reg, extendop);
        }
    }

    ResultRSE::from_rs(input_to_rs(ctx, input, narrow_mode))
}

fn input_to_rse_imm12<C: LowerCtx<Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> ResultRSEImm12 {
    if let InsnInputSource::Output(out) = input_source(ctx, input) {
        if let Some(imm_value) = output_to_const(ctx, out) {
            if let Some(i) = Imm12::maybe_from_u64(imm_value) {
                ctx.merged(out.insn);
                return ResultRSEImm12::Imm12(i);
            }
        }
    }

    ResultRSEImm12::from_rse(input_to_rse(ctx, input, narrow_mode))
}

fn input_to_rs_immlogic<C: LowerCtx<Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> ResultRSImmLogic {
    if let InsnInputSource::Output(out) = input_source(ctx, input) {
        if let Some(imm_value) = output_to_const(ctx, out) {
            if let Some(i) = ImmLogic::maybe_from_u64(imm_value) {
                ctx.merged(out.insn);
                return ResultRSImmLogic::ImmLogic(i);
            }
        }
    }

    ResultRSImmLogic::from_rs(input_to_rs(ctx, input, narrow_mode))
}

fn input_to_reg_immshift<C: LowerCtx<Inst>>(ctx: &mut C, input: InsnInput) -> ResultRegImmShift {
    if let InsnInputSource::Output(out) = input_source(ctx, input) {
        if let Some(imm_value) = output_to_const(ctx, out) {
            if let Some(immshift) = ImmShift::maybe_from_u64(imm_value) {
                ctx.merged(out.insn);
                return ResultRegImmShift::ImmShift(immshift);
            }
        }
    }

    ResultRegImmShift::Reg(input_to_reg(ctx, input, NarrowValueMode::None))
}

//============================================================================
// ALU instruction constructors.

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

fn alu_inst_immlogic(op: ALUOp, rd: Writable<Reg>, rn: Reg, rm: ResultRSImmLogic) -> Inst {
    match rm {
        ResultRSImmLogic::ImmLogic(imml) => Inst::AluRRImmLogic {
            alu_op: op,
            rd,
            rn,
            imml,
        },
        ResultRSImmLogic::Reg(rm) => Inst::AluRRR {
            alu_op: op,
            rd,
            rn,
            rm,
        },
        ResultRSImmLogic::RegShift(rm, shiftop) => Inst::AluRRRShift {
            alu_op: op,
            rd,
            rn,
            rm,
            shiftop,
        },
    }
}

fn alu_inst_immshift(op: ALUOp, rd: Writable<Reg>, rn: Reg, rm: ResultRegImmShift) -> Inst {
    match rm {
        ResultRegImmShift::ImmShift(immshift) => Inst::AluRRImmShift {
            alu_op: op,
            rd,
            rn,
            immshift,
        },
        ResultRegImmShift::Reg(rm) => Inst::AluRRR {
            alu_op: op,
            rd,
            rn,
            rm,
        },
    }
}

//============================================================================
// Lowering: addressing mode support. Takes instruction directly, rather
// than an `InsnInput`, to do more introspection.

/// Lower the address of a load or store.
fn lower_address<C: LowerCtx<Inst>>(
    ctx: &mut C,
    elem_ty: Type,
    addends: &[InsnInput],
    offset: i32,
) -> MemArg {
    // TODO: support base_reg + scale * index_reg. For this, we would need to pattern-match shl or
    // mul instructions (Load/StoreComplex don't include scale factors).

    // Handle one reg and offset that fits in immediate, if possible.
    if addends.len() == 1 {
        let reg = input_to_reg(ctx, addends[0], NarrowValueMode::ZeroExtend64);
        if let Some(memarg) = MemArg::reg_maybe_offset(reg, offset as i64, elem_ty) {
            return memarg;
        }
    }

    // Handle two regs and a zero offset, if possible.
    if addends.len() == 2 && offset == 0 {
        let ra = input_to_reg(ctx, addends[0], NarrowValueMode::ZeroExtend64);
        let rb = input_to_reg(ctx, addends[1], NarrowValueMode::ZeroExtend64);
        return MemArg::reg_reg(ra, rb);
    }

    // Otherwise, generate add instructions.
    let addr = ctx.tmp(RegClass::I64, I64);

    // Get the const into a reg.
    lower_constant(ctx, addr.clone(), offset as u64);

    // Add each addend to the address.
    for addend in addends {
        let reg = input_to_reg(ctx, *addend, NarrowValueMode::ZeroExtend64);
        ctx.emit(Inst::AluRRR {
            alu_op: ALUOp::Add64,
            rd: addr.clone(),
            rn: addr.to_reg(),
            rm: reg.clone(),
        });
    }

    MemArg::reg(addr.to_reg())
}

fn lower_constant<C: LowerCtx<Inst>>(ctx: &mut C, rd: Writable<Reg>, value: u64) {
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

/// Determines whether this condcode interprets inputs as signed or
/// unsigned.  See the documentation for the `icmp` instruction in
/// cranelift-codegen/meta/src/shared/instructions.rs for further insights
/// into this.
pub fn condcode_is_signed(cc: IntCC) -> bool {
    match cc {
        IntCC::Equal => false,
        IntCC::NotEqual => false,
        IntCC::SignedGreaterThanOrEqual => true,
        IntCC::SignedGreaterThan => true,
        IntCC::SignedLessThanOrEqual => true,
        IntCC::SignedLessThan => true,
        IntCC::UnsignedGreaterThanOrEqual => false,
        IntCC::UnsignedGreaterThan => false,
        IntCC::UnsignedLessThanOrEqual => false,
        IntCC::UnsignedLessThan => false,
        IntCC::Overflow => true,
        IntCC::NotOverflow => true,
    }
}

//=============================================================================
// Top-level instruction lowering entry point, for one instruction.

/// Actually codegen an instruction's results into registers.
fn lower_insn_to_regs<C: LowerCtx<Inst>>(ctx: &mut C, insn: IRInst) {
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
        Opcode::Iconst | Opcode::Bconst | Opcode::F32const | Opcode::F64const | Opcode::Null => {
            let value = output_to_const(ctx, outputs[0]).unwrap();
            let rd = output_to_reg(ctx, outputs[0]);
            lower_constant(ctx, rd, value);
        }
        Opcode::Iadd => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_rse_imm12(ctx, inputs[1], NarrowValueMode::None);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Add32, ALUOp::Add64);
            ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
        }
        Opcode::Isub => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_rse_imm12(ctx, inputs[1], NarrowValueMode::None);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
        }
        Opcode::Imax | Opcode::Imin | Opcode::Umin | Opcode::Umax => {
            // TODO
            unimplemented!()
        }

        Opcode::UaddSat | Opcode::SaddSat => {
            // We use the vector instruction set's saturating adds (UQADD /
            // SQADD), which require vector registers.
            let is_signed = op == Opcode::SaddSat;
            let narrow_mode = if is_signed {
                NarrowValueMode::SignExtend64
            } else {
                NarrowValueMode::ZeroExtend64
            };
            let alu_op = if is_signed {
                VecALUOp::SQAddScalar
            } else {
                VecALUOp::UQAddScalar
            };
            let va = ctx.tmp(RegClass::V128, I128);
            let vb = ctx.tmp(RegClass::V128, I128);
            let ra = input_to_reg(ctx, inputs[0], narrow_mode);
            let rb = input_to_reg(ctx, inputs[1], narrow_mode);
            let rd = output_to_reg(ctx, outputs[0]);
            ctx.emit(Inst::MovToVec64 { rd: va, rn: ra });
            ctx.emit(Inst::MovToVec64 { rd: vb, rn: rb });
            ctx.emit(Inst::VecRRR {
                rd: va,
                rn: va.to_reg(),
                rm: vb.to_reg(),
                alu_op,
            });
            ctx.emit(Inst::MovFromVec64 {
                rd,
                rn: va.to_reg(),
            });
        }

        Opcode::UsubSat | Opcode::SsubSat => {
            let is_signed = op == Opcode::SsubSat;
            let narrow_mode = if is_signed {
                NarrowValueMode::SignExtend64
            } else {
                NarrowValueMode::ZeroExtend64
            };
            let alu_op = if is_signed {
                VecALUOp::SQSubScalar
            } else {
                VecALUOp::UQSubScalar
            };
            let va = ctx.tmp(RegClass::V128, I128);
            let vb = ctx.tmp(RegClass::V128, I128);
            let ra = input_to_reg(ctx, inputs[0], narrow_mode);
            let rb = input_to_reg(ctx, inputs[1], narrow_mode);
            let rd = output_to_reg(ctx, outputs[0]);
            ctx.emit(Inst::MovToVec64 { rd: va, rn: ra });
            ctx.emit(Inst::MovToVec64 { rd: vb, rn: rb });
            ctx.emit(Inst::VecRRR {
                rd: va,
                rn: va.to_reg(),
                rm: vb.to_reg(),
                alu_op,
            });
            ctx.emit(Inst::MovFromVec64 {
                rd,
                rn: va.to_reg(),
            });
        }

        Opcode::Ineg => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = zero_reg();
            let rm = input_to_rse_imm12(ctx, inputs[0], NarrowValueMode::None);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
        }

        Opcode::Imul => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_reg(ctx, inputs[1], NarrowValueMode::None);
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
            let rd = output_to_reg(ctx, outputs[0]);
            let is_signed = op == Opcode::Smulhi;
            let narrow_mode = if is_signed {
                NarrowValueMode::SignExtend64
            } else {
                NarrowValueMode::ZeroExtend64
            };
            let alu_op = if is_signed {
                ALUOp::SMulH
            } else {
                ALUOp::UMulH
            };
            let rn = input_to_reg(ctx, inputs[0], narrow_mode);
            let rm = input_to_reg(ctx, inputs[1], narrow_mode);
            let ra = zero_reg();
            ctx.emit(Inst::AluRRRR {
                alu_op,
                rd,
                rn,
                rm,
                ra,
            });
        }

        Opcode::Udiv | Opcode::Sdiv | Opcode::Urem | Opcode::Srem => {
            let is_signed = match op {
                Opcode::Udiv | Opcode::Urem => false,
                Opcode::Sdiv | Opcode::Srem => true,
                _ => unreachable!(),
            };
            let is_rem = match op {
                Opcode::Udiv | Opcode::Sdiv => false,
                Opcode::Urem | Opcode::Srem => true,
                _ => unreachable!(),
            };
            let narrow_mode = if is_signed {
                NarrowValueMode::SignExtend64
            } else {
                NarrowValueMode::ZeroExtend64
            };
            let div_op = if is_signed {
                ALUOp::SDiv64
            } else {
                ALUOp::UDiv64
            };

            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], narrow_mode);
            if !is_rem {
                let rm = input_to_rse_imm12(ctx, inputs[1], narrow_mode);
                ctx.emit(alu_inst_imm12(div_op, rd, rn, rm));
            } else {
                let rm = input_to_reg(ctx, inputs[1], narrow_mode);
                // Remainder (rn % rm) is implemented as:
                //
                //   tmp = rn / rm
                //   rd = rn - (tmp*rm)
                //
                // use 'rd' for tmp and you have:
                //
                //   div rd, rn, rm       ; rd = rn / rm
                //   msub rd, rd, rm, rn  ; rd = rn - rd * rm
                ctx.emit(Inst::AluRRR {
                    alu_op: div_op,
                    rd,
                    rn,
                    rm,
                });
                ctx.emit(Inst::AluRRRR {
                    alu_op: ALUOp::MSub64,
                    rd: rd,
                    rn: rd.to_reg(),
                    rm: rm,
                    ra: rn,
                });
            }
        }

        Opcode::Uextend | Opcode::Sextend => {
            let output_ty = ty.unwrap();
            let input_ty = ctx.input_ty(insn, 0);
            let from_bits = ty_bits(input_ty) as u8;
            let to_bits = ty_bits(output_ty) as u8;
            let to_bits = std::cmp::max(32, to_bits);
            assert!(from_bits <= to_bits);
            if from_bits < to_bits {
                let signed = op == Opcode::Sextend;
                // If we reach this point, we weren't able to incorporate the extend as
                // a register-mode on another instruction, so we have a 'None'
                // narrow-value/extend mode here, and we emit the explicit instruction.
                let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
                let rd = output_to_reg(ctx, outputs[0]);
                ctx.emit(Inst::Extend {
                    rd,
                    rn,
                    signed,
                    from_bits,
                    to_bits,
                });
            }
        }

        Opcode::Bnot => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rm = input_to_rs_immlogic(ctx, inputs[0], NarrowValueMode::None);
            let ty = ty.unwrap();
            let alu_op = choose_32_64(ty, ALUOp::OrrNot32, ALUOp::OrrNot64);
            // NOT rd, rm ==> ORR_NOT rd, zero, rm
            ctx.emit(alu_inst_immlogic(alu_op, rd, zero_reg(), rm));
        }

        Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::BandNot
        | Opcode::BorNot
        | Opcode::BxorNot => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_rs_immlogic(ctx, inputs[1], NarrowValueMode::None);
            let ty = ty.unwrap();
            let alu_op = match op {
                Opcode::Band => choose_32_64(ty, ALUOp::And32, ALUOp::And64),
                Opcode::Bor => choose_32_64(ty, ALUOp::Orr32, ALUOp::Orr64),
                Opcode::Bxor => choose_32_64(ty, ALUOp::Eor32, ALUOp::Eor64),
                Opcode::BandNot => choose_32_64(ty, ALUOp::AndNot32, ALUOp::AndNot64),
                Opcode::BorNot => choose_32_64(ty, ALUOp::OrrNot32, ALUOp::OrrNot64),
                Opcode::BxorNot => choose_32_64(ty, ALUOp::EorNot32, ALUOp::EorNot64),
                _ => unreachable!(),
            };
            ctx.emit(alu_inst_immlogic(alu_op, rd, rn, rm));
        }

        Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => {
            let ty = ty.unwrap();
            let is32 = ty_bits(ty) <= 32;
            let narrow_mode = match (op, is32) {
                (Opcode::Ishl, _) => NarrowValueMode::None,
                (Opcode::Ushr, false) => NarrowValueMode::ZeroExtend64,
                (Opcode::Ushr, true) => NarrowValueMode::ZeroExtend32,
                (Opcode::Sshr, false) => NarrowValueMode::SignExtend64,
                (Opcode::Sshr, true) => NarrowValueMode::SignExtend32,
                _ => unreachable!(),
            };
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], narrow_mode);
            let rm = input_to_reg_immshift(ctx, inputs[1]);
            let alu_op = match op {
                Opcode::Ishl => choose_32_64(ty, ALUOp::Lsl32, ALUOp::Lsl64),
                Opcode::Ushr => choose_32_64(ty, ALUOp::Lsr32, ALUOp::Lsr64),
                Opcode::Sshr => choose_32_64(ty, ALUOp::Asr32, ALUOp::Asr64),
                _ => unreachable!(),
            };
            ctx.emit(alu_inst_immshift(alu_op, rd, rn, rm));
        }

        Opcode::Rotr => {
            // For a 32-bit or 64-bit rotate-right, we can use the ROR
            // instruction directly.
            //
            // For a < 32-bit rotate-right, we synthesize this as:
            //
            //    rotr rd, rn, rm
            //
            //       =>
            //
            //    zero-extend rn, <32-or-64>
            //    sub tmp1, rm, <bitwidth>
            //    sub tmp1, zero, tmp1  ; neg
            //    lsr tmp2, rn, rm
            //    lsl rd, rn, tmp1
            //    orr rd, rd, tmp2
            //
            // For a constant amount, we can instead do:
            //
            //    zero-extend rn, <32-or-64>
            //    lsr tmp2, rn, #<shiftimm>
            //    lsl rd, rn, <bitwidth - shiftimm>
            //    orr rd, rd, tmp2

            let ty = ty.unwrap();
            let bits = ty_bits(ty);
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(
                ctx,
                inputs[0],
                if bits <= 32 {
                    NarrowValueMode::ZeroExtend32
                } else {
                    NarrowValueMode::ZeroExtend64
                },
            );
            let rm = input_to_reg_immshift(ctx, inputs[1]);

            if bits == 32 || bits == 64 {
                let alu_op = choose_32_64(ty, ALUOp::RotR32, ALUOp::RotR64);
                ctx.emit(alu_inst_immshift(alu_op, rd, rn, rm));
            } else {
                assert!(bits < 32);
                match rm {
                    ResultRegImmShift::Reg(reg) => {
                        let tmp1 = ctx.tmp(RegClass::I64, I32);
                        let tmp2 = ctx.tmp(RegClass::I64, I32);
                        ctx.emit(Inst::AluRRImm12 {
                            alu_op: ALUOp::Sub32,
                            rd: tmp1,
                            rn: reg,
                            imm12: Imm12::maybe_from_u64(bits as u64).unwrap(),
                        });
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::Sub32,
                            rd: tmp1,
                            rn: zero_reg(),
                            rm: tmp1.to_reg(),
                        });
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::Lsr32,
                            rd: tmp2,
                            rn: rn,
                            rm: reg,
                        });
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::Lsl32,
                            rd: rd,
                            rn: rn,
                            rm: tmp1.to_reg(),
                        });
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::Orr32,
                            rd: rd,
                            rn: rd.to_reg(),
                            rm: tmp2.to_reg(),
                        });
                    }
                    ResultRegImmShift::ImmShift(immshift) => {
                        let tmp1 = ctx.tmp(RegClass::I64, I32);
                        let amt = immshift.value();
                        assert!(amt <= bits as u8);
                        let opp_shift = ImmShift::maybe_from_u64(bits as u64 - amt as u64).unwrap();
                        ctx.emit(Inst::AluRRImmShift {
                            alu_op: ALUOp::Lsr32,
                            rd: tmp1,
                            rn: rn,
                            immshift: immshift,
                        });
                        ctx.emit(Inst::AluRRImmShift {
                            alu_op: ALUOp::Lsl32,
                            rd: rd,
                            rn: rn,
                            immshift: opp_shift,
                        });
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::Orr32,
                            rd: rd,
                            rn: rd.to_reg(),
                            rm: tmp1.to_reg(),
                        });
                    }
                }
            }
        }

        Opcode::Rotl => {
            // ARM64 does not have a ROL instruction, so we always synthesize
            // this as:
            //
            //    rotl rd, rn, rm
            //
            //       =>
            //
            //    zero-extend rn, <32-or-64>
            //    sub tmp1, rm, <bitwidth>
            //    sub tmp1, zero, tmp1  ; neg
            //    lsl tmp2, rn, rm
            //    lsr rd, rn, tmp1
            //    orr rd, rd, tmp2
            //
            // For a constant amount, we can instead do:
            //
            //    zero-extend rn, <32-or-64>
            //    lsl tmp2, rn, #<shiftimm>
            //    lsr rd, rn, #<bitwidth - shiftimm>
            //    orr rd, rd, tmp2

            let ty = ty.unwrap();
            let bits = ty_bits(ty);
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(
                ctx,
                inputs[0],
                if bits <= 32 {
                    NarrowValueMode::ZeroExtend32
                } else {
                    NarrowValueMode::ZeroExtend64
                },
            );
            let rm = input_to_reg_immshift(ctx, inputs[1]);

            match rm {
                ResultRegImmShift::Reg(reg) => {
                    let tmp1 = ctx.tmp(RegClass::I64, I32);
                    let tmp2 = ctx.tmp(RegClass::I64, I64);
                    ctx.emit(Inst::AluRRImm12 {
                        alu_op: ALUOp::Sub32,
                        rd: tmp1,
                        rn: reg,
                        imm12: Imm12::maybe_from_u64(bits as u64).unwrap(),
                    });
                    ctx.emit(Inst::AluRRR {
                        alu_op: ALUOp::Sub32,
                        rd: tmp1,
                        rn: zero_reg(),
                        rm: tmp1.to_reg(),
                    });
                    ctx.emit(Inst::AluRRR {
                        alu_op: choose_32_64(ty, ALUOp::Lsl32, ALUOp::Lsl64),
                        rd: tmp2,
                        rn: rn,
                        rm: reg,
                    });
                    ctx.emit(Inst::AluRRR {
                        alu_op: choose_32_64(ty, ALUOp::Lsr32, ALUOp::Lsr64),
                        rd: rd,
                        rn: rn,
                        rm: tmp1.to_reg(),
                    });
                    ctx.emit(Inst::AluRRR {
                        alu_op: choose_32_64(ty, ALUOp::Orr32, ALUOp::Orr64),
                        rd: rd,
                        rn: rd.to_reg(),
                        rm: tmp2.to_reg(),
                    });
                }
                ResultRegImmShift::ImmShift(immshift) => {
                    let tmp1 = ctx.tmp(RegClass::I64, I64);
                    let amt = immshift.value();
                    assert!(amt <= bits as u8);
                    let opp_shift = ImmShift::maybe_from_u64(bits as u64 - amt as u64).unwrap();
                    ctx.emit(Inst::AluRRImmShift {
                        alu_op: choose_32_64(ty, ALUOp::Lsl32, ALUOp::Lsl64),
                        rd: tmp1,
                        rn: rn,
                        immshift: immshift,
                    });
                    ctx.emit(Inst::AluRRImmShift {
                        alu_op: choose_32_64(ty, ALUOp::Lsr32, ALUOp::Lsr64),
                        rd: rd,
                        rn: rn,
                        immshift: opp_shift,
                    });
                    ctx.emit(Inst::AluRRR {
                        alu_op: choose_32_64(ty, ALUOp::Orr32, ALUOp::Orr64),
                        rd: rd,
                        rn: rd.to_reg(),
                        rm: tmp1.to_reg(),
                    });
                }
            }
        }

        Opcode::Bitrev | Opcode::Clz | Opcode::Cls => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let op = BitOp::from((op, ty.unwrap()));
            ctx.emit(Inst::BitRR { rd, rn, op });
        }

        Opcode::Ctz => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let op = BitOp::from((Opcode::Bitrev, ty.unwrap()));
            ctx.emit(Inst::BitRR { rd, rn, op });
            let op = BitOp::from((Opcode::Clz, ty.unwrap()));
            ctx.emit(Inst::BitRR {
                rd,
                rn: rd.to_reg(),
                op,
            });
        }

        Opcode::Popcnt => {
            // TODO
            unimplemented!()
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
            let rd = input_to_reg(ctx, inputs[0], NarrowValueMode::None);

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
            unimplemented!()
        }

        Opcode::StackStore => {
            // TODO
            unimplemented!()
        }

        Opcode::StackAddr => {
            // TODO
            unimplemented!()
        }

        Opcode::HeapAddr => {
            panic!("heap_addr should have been removed by legalization!");
        }

        Opcode::TableAddr => {
            panic!("table_addr should have been removed by legalization!");
        }

        Opcode::Nop => {
            // Nothing.
        }

        Opcode::Select | Opcode::Selectif => {
            // TODO.
            unimplemented!()
        }

        Opcode::Bitselect => {
            // TODO.
            unimplemented!()
        }

        Opcode::IsNull | Opcode::IsInvalid | Opcode::Trueif | Opcode::Trueff => {
            // TODO.
            unimplemented!()
        }

        Opcode::Copy => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            ctx.emit(Inst::gen_move(rd, rn));
        }

        Opcode::Bint | Opcode::Breduce | Opcode::Bextend | Opcode::Ireduce => {
            // All of these ops are simply a move from a zero-extended source.
            // Here is why this works, in each case:
            //
            // - Bint: Bool-to-int. We always represent a bool as a 0 or 1, so we
            //   merely need to zero-extend here.
            //
            // - Breduce, Bextend: changing width of a boolean. We represent a
            //   bool as a 0 or 1, so again, this is a zero-extend / no-op.
            //
            // - Ireduce: changing width of an integer. Smaller ints are stored
            //   with undefined high-order bits, so we can simply do a copy.

            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend64);
            let rd = output_to_reg(ctx, outputs[0]);
            ctx.emit(Inst::gen_move(rd, rn));
        }

        Opcode::Bmask => {
            // Bool is {0, 1}, so we can subtract from 0 to get all-1s.
            let rd = output_to_reg(ctx, outputs[0]);
            let rm = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend64);
            ctx.emit(Inst::AluRRR {
                alu_op: ALUOp::Sub64,
                rd,
                rn: zero_reg(),
                rm,
            });
        }

        Opcode::Isplit | Opcode::Iconcat => {
            // TODO
            unimplemented!()
        }

        Opcode::FallthroughReturn => {
            // What is this? The definition says it's a "special
            // instruction" meant to allow falling through into an
            // epilogue that will then return; that just sounds like a
            // normal fallthrough. TODO: Do we need to handle this
            // differently?
            panic!("FallthroughReturn should not appear!");
        }

        Opcode::Return => {
            for (i, input) in inputs.iter().enumerate() {
                // N.B.: according to the AArch64 ABI, the top bits of a register
                // (above the bits for the value's type) are undefined, so we
                // need not extend the return values.
                let reg = input_to_reg(ctx, *input, NarrowValueMode::None);
                let retval_reg = ctx.retval(i);
                ctx.emit(Inst::gen_move(retval_reg, reg));
            }
            // N.B.: the Ret itself is generated by the ABI.
        }

        Opcode::Ifcmp => {
            // An Ifcmp must always be seen as a use of a brif instruction. This
            // will always be the case as long as the IR uses an Ifcmp from the
            // same block, or a dominating block. In other words, it cannot pass
            // through a BB param (phi). The flags pass of the verifier will
            // ensure this.
            panic!("Should never reach ifcmp as isel root!");
        }

        Opcode::Icmp => {
            let condcode = inst_condcode(ctx.data(insn)).unwrap();
            let cond = lower_condcode(condcode);
            let is_signed = condcode_is_signed(condcode);
            let ty = ctx.input_ty(insn, 0);
            let bits = ty_bits(ty);
            let narrow_mode = match (bits <= 32, is_signed) {
                (true, true) => NarrowValueMode::SignExtend32,
                (true, false) => NarrowValueMode::ZeroExtend32,
                (false, true) => NarrowValueMode::SignExtend64,
                (false, false) => NarrowValueMode::ZeroExtend64,
            };
            let alu_op = choose_32_64(ty, ALUOp::SubS32, ALUOp::SubS64);
            let rn = input_to_reg(ctx, inputs[0], narrow_mode);
            let rm = input_to_rse_imm12(ctx, inputs[1], narrow_mode);
            let rd = output_to_reg(ctx, outputs[0]);
            ctx.emit(alu_inst_imm12(alu_op, writable_zero_reg(), rn, rm));
            ctx.emit(Inst::CondSet { cond, rd });
        }

        Opcode::JumpTableEntry => {
            // TODO
            unimplemented!()
        }

        Opcode::JumpTableBase => {
            // TODO
            unimplemented!()
        }

        Opcode::Debugtrap | Opcode::Trap | Opcode::Trapif => {
            let is_cond = op == Opcode::Trapif;
            let maybe_trapinfo = inst_trapcode(ctx.data(insn)).map(|code| (ctx.srcloc(insn), code));

            if is_cond {
                let condcode = inst_condcode(ctx.data(insn)).unwrap();
                let cond = lower_condcode(condcode);
                let is_signed = condcode_is_signed(condcode);
                // Verification ensures that the input is always a
                // single-def ifcmp.
                let ifcmp_insn = maybe_input_insn(ctx, inputs[0], Opcode::Ifcmp).unwrap();
                lower_ifcmp_to_flags(ctx, ifcmp_insn, is_signed);

                // Branch around the break instruction with inverted cond. Go straight
                // to lowered one-target form; this is logically part of a single-in
                // single-out template lowering.
                let cond = cond.invert();
                ctx.emit(Inst::CondBrLowered {
                    target: BranchTarget::ResolvedOffset(8),
                    kind: CondBrKind::Cond(cond),
                });
            }
            ctx.emit(Inst::Brk {
                trap_info: maybe_trapinfo,
            });
        }

        Opcode::Safepoint => {
            panic!("trap support not implemented!");
        }

        Opcode::Trapz | Opcode::Trapnz => {
            panic!("trapz / trapnz should have been removed by legalization!");
        }

        Opcode::Trapff => {
            panic!("trapff requires floating point support");
        }

        Opcode::ResumableTrap => {
            panic!("Resumable traps not supported");
        }

        Opcode::FuncAddr => {
            let rd = output_to_reg(ctx, outputs[0]);
            let extname = ctx.call_target(insn).unwrap().clone();
            ctx.emit(Inst::ULoad64 {
                rd,
                mem: MemArg::Label(MemLabel::ExtName(extname, 0)),
            });
        }

        Opcode::GlobalValue => {
            panic!("global_value should have been removed by legalization!");
        }

        Opcode::SymbolValue => {
            let rd = output_to_reg(ctx, outputs[0]);
            let (extname, offset) = ctx.symbol_value(insn).unwrap();
            let extname = extname.clone();
            ctx.emit(Inst::ULoad64 {
                rd,
                mem: MemArg::Label(MemLabel::ExtName(extname, offset)),
            });
        }

        Opcode::Call | Opcode::CallIndirect => {
            let (abi, inputs) = match op {
                Opcode::Call => {
                    let extname = ctx.call_target(insn).unwrap();
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (ARM64ABICall::from_func(sig, extname), &inputs[..])
                }
                Opcode::CallIndirect => {
                    let ptr = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend64);
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() - 1 == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (ARM64ABICall::from_ptr(sig, ptr), &inputs[1..])
                }
                _ => unreachable!(),
            };
            for (i, input) in inputs.iter().enumerate() {
                let arg_reg = input_to_reg(ctx, *input, NarrowValueMode::None);
                ctx.emit(abi.gen_copy_reg_to_arg(i, arg_reg));
            }
            ctx.emit(abi.gen_call());
            for (i, output) in outputs.iter().enumerate() {
                let retval_reg = output_to_reg(ctx, *output);
                ctx.emit(abi.gen_copy_retval_to_reg(i, retval_reg));
            }
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
        | Opcode::SshrImm
        | Opcode::IcmpImm
        | Opcode::IfcmpImm => {
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
            panic!("x86-specific opcode in supposedly arch-neutral IR!");
        }
    }
}

//=============================================================================
// Helpers for instruction lowering.
fn ty_bits(ty: Type) -> usize {
    match ty {
        B1 => 1,
        B8 | I8 => 8,
        B16 | I16 => 16,
        B32 | I32 | F32 => 32,
        B64 | I64 | F64 => 64,
        B128 | I128 => 128,
        IFLAGS | FFLAGS => 32,
        _ => panic!("ty_bits() on unknown type: {:?}", ty),
    }
}

fn choose_32_64(ty: Type, op32: ALUOp, op64: ALUOp) -> ALUOp {
    let bits = ty_bits(ty);
    if bits <= 32 {
        op32
    } else if bits == 64 {
        op64
    } else {
        panic!("choose_32_64 on > 64 bits!")
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

fn inst_trapcode(data: &InstructionData) -> Option<TrapCode> {
    match data {
        &InstructionData::Trap { code, .. }
        | &InstructionData::CondTrap { code, .. }
        | &InstructionData::IntCondTrap { code, .. }
        | &InstructionData::FloatCondTrap { code, .. } => Some(code),
        _ => None,
    }
}

fn maybe_input_insn<C: LowerCtx<Inst>>(c: &mut C, input: InsnInput, op: Opcode) -> Option<IRInst> {
    if let InsnInputSource::Output(out) = input_source(c, input) {
        let data = c.data(out.insn);
        if data.opcode() == op {
            return Some(out.insn);
        }
    }
    None
}

fn lower_ifcmp_to_flags<C: LowerCtx<Inst>>(ctx: &mut C, ifcmp_insn: IRInst, is_signed: bool) {
    // Get the condcode and the args, and treat this like a BrIcmp.
    let ty = ctx.input_ty(ifcmp_insn, 0);
    let bits = ty_bits(ty);
    let narrow_mode = match (bits <= 32, is_signed) {
        (true, true) => NarrowValueMode::SignExtend32,
        (true, false) => NarrowValueMode::ZeroExtend32,
        (false, true) => NarrowValueMode::SignExtend64,
        (false, false) => NarrowValueMode::ZeroExtend64,
    };
    let ifcmp_inputs = [
        InsnInput {
            insn: ifcmp_insn,
            input: 0,
        },
        InsnInput {
            insn: ifcmp_insn,
            input: 1,
        },
    ];
    let ty = ctx.input_ty(ifcmp_insn, 0);
    let rn = input_to_reg(ctx, ifcmp_inputs[0], narrow_mode);
    let rm = input_to_rse_imm12(ctx, ifcmp_inputs[1], narrow_mode);
    let alu_op = choose_32_64(ty, ALUOp::SubS32, ALUOp::SubS64);
    let rd = writable_zero_reg();
    ctx.merged(ifcmp_insn);
    ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
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
            let op0 = ctx.data(branches[0]).opcode();
            let op1 = ctx.data(branches[1]).opcode();

            //println!(
            //    "lowering two-branch group: opcodes are {:?} and {:?}",
            //    op0, op1
            //);

            assert!(op1 == Opcode::Jump || op1 == Opcode::Fallthrough);
            let taken = BranchTarget::Block(targets[0]);
            let not_taken = match op1 {
                Opcode::Jump => BranchTarget::Block(targets[1]),
                Opcode::Fallthrough => BranchTarget::Block(fallthrough.unwrap()),
                _ => unreachable!(), // assert above.
            };
            match op0 {
                Opcode::Brz | Opcode::Brnz => {
                    let rt = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 0,
                        },
                        NarrowValueMode::ZeroExtend64,
                    );
                    let kind = match op0 {
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
                    let condcode = inst_condcode(ctx.data(branches[0])).unwrap();
                    let cond = lower_condcode(condcode);
                    let is_signed = condcode_is_signed(condcode);
                    let ty = ctx.input_ty(branches[0], 0);
                    let bits = ty_bits(ty);
                    let narrow_mode = match (bits <= 32, is_signed) {
                        (true, true) => NarrowValueMode::SignExtend32,
                        (true, false) => NarrowValueMode::ZeroExtend32,
                        (false, true) => NarrowValueMode::SignExtend64,
                        (false, false) => NarrowValueMode::ZeroExtend64,
                    };
                    let rn = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 0,
                        },
                        narrow_mode,
                    );
                    let rm = input_to_rse_imm12(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 1,
                        },
                        narrow_mode,
                    );

                    let alu_op = choose_32_64(ty, ALUOp::SubS32, ALUOp::SubS64);
                    let rd = writable_zero_reg();
                    ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
                    ctx.emit(Inst::CondBr {
                        taken,
                        not_taken,
                        kind: CondBrKind::Cond(cond),
                    });
                }

                Opcode::Brif => {
                    let condcode = inst_condcode(ctx.data(branches[0])).unwrap();
                    let cond = lower_condcode(condcode);
                    let is_signed = condcode_is_signed(condcode);
                    let flag_input = InsnInput {
                        insn: branches[0],
                        input: 0,
                    };
                    if let Some(ifcmp_insn) = maybe_input_insn(ctx, flag_input, Opcode::Ifcmp) {
                        lower_ifcmp_to_flags(ctx, ifcmp_insn, is_signed);
                        ctx.emit(Inst::CondBr {
                            taken,
                            not_taken,
                            kind: CondBrKind::Cond(cond),
                        });
                    } else {
                        // If the ifcmp result is actually placed in a
                        // register, we need to move it back into the flags.
                        let rn = input_to_reg(ctx, flag_input, NarrowValueMode::None);
                        ctx.emit(Inst::MovToNZCV { rn });
                        ctx.emit(Inst::CondBr {
                            taken,
                            not_taken,
                            kind: CondBrKind::Cond(cond),
                        });
                    }
                }

                // TODO: Brff
                _ => unimplemented!(),
            }
        } else {
            // Must be an unconditional branch or an indirect branch.
            let op = ctx.data(branches[0]).opcode();
            match op {
                Opcode::Jump | Opcode::Fallthrough => {
                    assert!(branches.len() == 1);
                    // In the Fallthrough case, the machine-independent driver
                    // fills in `targets[0]` with our fallthrough block, so this
                    // is valid for both Jump and Fallthrough.
                    ctx.emit(Inst::Jump {
                        dest: BranchTarget::Block(targets[0]),
                    });
                }
                Opcode::BrTable => {
                    // Expand `br_table index, default, JT` to:
                    //
                    //   subs idx, #jt_size
                    //   b.hs default
                    //   adr vTmp1, JT_addr@pcrel
                    //   ldr vTmp1, [vTmp1, idx, lsl #2]
                    //   adr vTmp2, start_of_code@pcrel
                    //   add vTmp2, vTmp2, vTmp1
                    //   br vTmp2
                    let jt = match ctx.data(branches[0]) {
                        &InstructionData::BranchTable { table, .. } => table,
                        _ => panic!("Unexpected instruction format for BrTable op"),
                    };

                    let jt_size = targets.len() - 1;
                    assert!(jt_size <= std::u32::MAX as usize);
                    let ridx = input_to_reg(
                        ctx,
                        InsnInput {
                            insn: branches[0],
                            input: 0,
                        },
                        NarrowValueMode::ZeroExtend32,
                    );

                    let rtmp1 = ctx.tmp(RegClass::I64, I32);
                    let rtmp2 = ctx.tmp(RegClass::I64, I32);

                    // Bounds-check and branch to default.
                    if let Some(imm12) = Imm12::maybe_from_u64(jt_size as u64) {
                        ctx.emit(Inst::AluRRImm12 {
                            alu_op: ALUOp::SubS32,
                            rd: writable_zero_reg(),
                            rn: ridx,
                            imm12,
                        });
                    } else {
                        lower_constant(ctx, rtmp1, jt_size as u64);
                        ctx.emit(Inst::AluRRR {
                            alu_op: ALUOp::SubS32,
                            rd: writable_zero_reg(),
                            rn: ridx,
                            rm: rtmp1.to_reg(),
                        });
                    }
                    ctx.emit(Inst::CondBrLowered {
                        kind: CondBrKind::Cond(Cond::Hs), // unsigned >=
                        target: BranchTarget::Block(targets[0]),
                    });

                    // Load address of jump table
                    ctx.emit(Inst::Adr {
                        rd: rtmp1,
                        label: MemLabel::JumpTable(jt),
                    });
                    // Load value out of jump table
                    ctx.emit(Inst::ULoad32 {
                        rd: rtmp1,
                        mem: MemArg::reg_reg_scaled(rtmp1.to_reg(), ridx, I32),
                    });
                    // Get base of code segment (using PC-rel reference)
                    ctx.emit(Inst::Adr {
                        rd: rtmp2,
                        label: MemLabel::CodeOffset(0),
                    });
                    // Add base to jump-table-sourced block offset
                    ctx.emit(Inst::AluRRR {
                        alu_op: ALUOp::Add64,
                        rd: rtmp2,
                        rn: rtmp1.to_reg(),
                        rm: rtmp2.to_reg(),
                    });
                    // Jump to it!
                    let jt_targets: Vec<BlockIndex> = targets.iter().skip(1).cloned().collect();
                    ctx.emit(Inst::IndirectBr {
                        rn: rtmp2.to_reg(),
                        targets: jt_targets,
                    });
                }

                Opcode::Trap => unimplemented!(),

                _ => panic!("Unknown branch type!"),
            }
        }
    }
}
