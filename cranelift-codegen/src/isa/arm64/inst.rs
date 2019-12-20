//! This module defines `Op` and `Arg` and friends for Arm64, along with constructors/helpers for
//! lowering code that constructs `MachInst<Op, Arg>` values.

/*
 * TODO:
 * - Support lowering table keying on "controlling typevar".
 *
 * - Support (Iadd (Uextend ...) ...) and (Iadd (Ishl_imm ...) ...) using extended-register and
 *   shifted-register forms.
 */

use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{FuncRef, GlobalValue, Inst, InstructionData, Type};
use crate::ir::types::*;
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::lower::LowerCtx;
use crate::machinst::*;
use crate::{mach_args, mach_ops};

mach_ops!(Op, {
    Add32,
    Add64,
    AddI32,
    AddI64,
    Sub32,
    Sub64,
    SubI32,
    SubI64,
    Neg32,
    Neg64,
    SMulH,
    SMulL,
    UMulH,
    UMulL,
    SMSubL,
    UMSubL,
    UDiv,
    SDiv,
    LdrImm,
    LdrLit32,
    LdrLit64
});

mach_args!(Arg, ArgKind, {
    Imm(ShiftedImm),
    Reg(RegRef),
    ShiftedReg(RegRef, ShiftOp, usize),
    ExtendedReg(RegRef, ExtendOp, usize),
    Mem(MemArg)
});

impl Arg {
    /// Get the embedded register reference, if any, in this arg.
    pub fn reg(&self) -> &RegRef {
        match self {
            &Arg::Reg(ref r) => r,
            &Arg::ShiftedReg(ref r, ..) => r,
            &Arg::ExtendedReg(ref r, ..) => r,
            _ => panic!("reg() on arg without register"),
        }
    }
}

/// A shifted immediate value in 'imm12' format: supports 12 bits, shifted left by 0 or 12 places.
#[derive(Clone, Debug)]
pub struct ShiftedImm {
    /// The immediate bits.
    pub bits: usize,
    /// Whether the immediate bits are shifted left by 12 or not.
    pub shift12: bool,
}

impl ShiftedImm {
    /// Compute a ShiftedImm from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<ShiftedImm> {
        if val == 0 {
            Some(ShiftedImm { bits: 0, shift12: false })
        } else if val < 0xfff {
            Some(ShiftedImm { bits: val as usize, shift12: false })
        } else if val < 0xfff_000 && (val & 0xfff == 0) {
            Some(ShiftedImm { bits: (val as usize) >> 12, shift12: true })
        } else {
            None
        }
    }

    /// Bits for 2-bit "shift" field in e.g. AddI.
    pub fn shift_bits(&self) -> u8 {
        if self.shift12 {
            0b01
        } else {
            0b00
        }
    }
}

/// A shift operator for a register or immediate.
#[derive(Clone, Debug)]
pub enum ShiftOp {
    ASR,
    LSR,
    LSL,
    ROR,
}

impl ShiftOp {
    /// Get the encoding of this shift op.
    pub fn bits(&self) -> u8 {
        match self {
            &ShiftOp::LSL => 0b00,
            &ShiftOp::LSR => 0b01,
            &ShiftOp::ASR => 0b10,
            &ShiftOp::ROR => 0b11,
        }
    }
}

/// An extend operator for a register.
#[derive(Clone, Debug)]
pub enum ExtendOp {
    SXTB,
    SXTH,
    SXTW,
    SXTX,
    UXTB,
    UXTH,
    UXTW,
    UXTX,
}

impl ExtendOp {
    /// Encoding of this op.
    pub fn bits(&self) -> u8 {
        match self {
            &ExtendOp::UXTB => 0b000,
            &ExtendOp::UXTH => 0b001,
            &ExtendOp::UXTW => 0b010,
            &ExtendOp::UXTX => 0b011,
            &ExtendOp::SXTB => 0b100,
            &ExtendOp::SXTH => 0b101,
            &ExtendOp::SXTW => 0b110,
            &ExtendOp::SXTX => 0b111,
        }
    }
}

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    Base(RegRef),
    BaseImm(RegRef, usize),
    BaseOffsetShifted(RegRef, usize),
    BaseImmPreIndexed(RegRef, usize),
    BaseImmPostIndexed(RegRef, usize),
    Label(MemLabel),
}

/// A reference to some memory address.
#[derive(Clone, Debug)]
pub enum MemLabel {
    /// A value in a constant pool, already emitted.
    ConstantPool(ConstantOffset),
    /// A value in a constant pool, to be emitted during binemit.
    ConstantData(ConstantData),
    Function(FuncRef),
    GlobalValue(GlobalValue),
}

#[derive(Clone, Debug)]
enum Cond {
    Eq,
    Ne,
    Hs,
    Lo,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

// -------------------- instruction constructors -------------------

/// Make a reg / reg / reg inst.
pub fn make_reg_reg(op: Op, rd: RegRef, rm: RegRef) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rm))
}

/// Make a reg / reg / reg inst.
pub fn make_reg_reg_reg(op: Op, rd: RegRef, rn: RegRef, rm: RegRef) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rn))
        .with_arg(Arg::Reg(rm))
}

/// Make a reg / reg / reg / reg inst.
pub fn make_reg_reg_reg_reg(
    op: Op,
    rd: RegRef,
    rn: RegRef,
    rm: RegRef,
    ra: RegRef,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rn))
        .with_arg(Arg::Reg(rm))
        .with_arg(Arg::Reg(ra))
}

/// Make a reg / reg / immediate inst.
pub fn make_reg_reg_imm(op: Op, rd: RegRef, rn: RegRef, imm: ShiftedImm) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rn))
        .with_arg(Arg::Imm(imm))
}

/// Make a reg / reg / rshift inst.
pub fn make_reg_reg_rshift(
    op: Op,
    rd: RegRef,
    rn: RegRef,
    rm: RegRef,
    shift: ShiftOp,
    amt: usize,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rn))
        .with_arg(Arg::ShiftedReg(rm, shift, amt))
}

/// Make a reg / reg / rextend inst.
pub fn make_reg_reg_rextend(
    op: Op,
    rd: RegRef,
    rn: RegRef,
    rm: RegRef,
    ext: ExtendOp,
    shift_amt: usize,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Reg(rn))
        .with_arg(Arg::ExtendedReg(rm, ext, shift_amt))
}

/// Make a reg / memory-label inst.
pub fn make_reg_memlabel(op: Op, rd: RegRef, mem: MemArg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Mem(mem))
}

/// Make a reg / memory inst.
pub fn make_reg_mem(op: Op, rd: RegRef, mem: MemArg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Reg(rd))
        .with_arg(Arg::Mem(mem))
}

/// Make a memory / reg inst.
pub fn make_mem_reg(op: Op, mem: MemArg, rn: RegRef) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg(Arg::Mem(mem))
        .with_arg(Arg::Reg(rn))
}

/// Helper: load an arbitrary immediate (up to 64 bits large) into a register, using the smallest
/// possible encoding or constant pool entry.
pub fn load_imm<'a>(ctx: &mut LowerCtx<Op, Arg>, value: u64, dest: RegRef) {
    if let Some(imm) = ShiftedImm::maybe_from_u64(value) {
        let xzr = ctx.fixed_tmp(I64, 31);
        ctx.emit(make_reg_reg_imm(Op::AddI64, dest, xzr, imm));
    } else if value <= std::u32::MAX as u64 {
        ctx.emit(make_reg_memlabel(
            Op::LdrLit32,
            dest,
            MemArg::Label(MemLabel::ConstantData(u32_constant(value as u32))),
        ));
    } else {
        ctx.emit(make_reg_memlabel(
            Op::LdrLit64,
            dest,
            MemArg::Label(MemLabel::ConstantData(u64_constant(value))),
        ));
    }
}

/// Helper: maybe return a `ShiftedImm` if an immediate value fits.
pub fn with_imm12<'a, F>(ctx: &mut LowerCtx<Op, Arg>, value: u64, f: F) -> bool
where
    F: FnOnce(&mut LowerCtx<Op, Arg>, ShiftedImm),
{
    if let Some(imm) = ShiftedImm::maybe_from_u64(value) {
        f(ctx, imm);
        true
    } else {
        false
    }
}

/// Helper: get a ConstantData from a u32.
pub fn u32_constant(bits: u32) -> ConstantData {
    let data = [
        (bits & 0xff) as u8,
        ((bits >> 8) & 0xff) as u8,
        ((bits >> 16) & 0xff) as u8,
        ((bits >> 24) & 0xff) as u8,
    ];
    ConstantData::from(&data[..])
}

/// Helper: get a ConstantData from a u64.
pub fn u64_constant(bits: u64) -> ConstantData {
    let data = [
        (bits & 0xff) as u8,
        ((bits >> 8) & 0xff) as u8,
        ((bits >> 16) & 0xff) as u8,
        ((bits >> 24) & 0xff) as u8,
        ((bits >> 32) & 0xff) as u8,
        ((bits >> 40) & 0xff) as u8,
        ((bits >> 48) & 0xff) as u8,
        ((bits >> 56) & 0xff) as u8,
    ];
    ConstantData::from(&data[..])
}

/// Helper: pick one of two opcodes based on size.
pub fn choose_32_64(ty: Type, op32: Op, op64: Op) -> Op {
    match ty {
        I32 | B32 | F32 => op32,
        I64 | B64 | F64 => op64,
        _ => panic!("Bad type: {}", ty),
    }
}
