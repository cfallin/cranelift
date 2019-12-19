//! This module defines `Arm64Inst` and friends, which implement `MachInst`.

use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{FuncRef, GlobalValue, Inst, InstructionData, Type};
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::lower::LowerCtx;
use crate::machinst::*;
use crate::{mach_args, mach_ops};

use std::slice;

mach_ops!(Op, {
    Add,
    AddI,
    Sub,
    SubI,
    Neg,
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
    RegDef(Value),
    Imm(ShiftedImm),
    RegUse(Value),
    ShiftedReg(Value, ShiftOp, usize),
    ExtendedReg(Value, ExtendOp, usize),
    Mem(MemArg)
});

impl MachInstArg for Arg {
    fn defs(&self) -> &[Value] {
        match self {
            &Arg::RegDef(ref v) => slice::from_ref(v),
            &Arg::RegUse(..) |
            &Arg::Imm(..) |
            &Arg::ShiftedReg(..) |
            &Arg::ExtendedReg(..) => &[],
            &Arg::Mem(ref m) => m.defs(),
        }
    }

    fn uses(&self) -> &[Value] {
        match self {
            &Arg::RegDef(ref v) => &[],
            &Arg::RegUse(ref v) => slice::from_ref(v),
            &Arg::Imm(..) => &[],
            &Arg::ShiftedReg(ref v, ..) => slice::from_ref(v),
            &Arg::ExtendedReg(ref v, ..) => slice::from_reg(v),
            &Arg::Mem(ref m) => m.uses(),
        }
    }

    fn regclass_for_type(ty: Type) -> RegClass {
        if ty.is_int() || ty.is_bool() {
            GPR
        } else {
            FPR
        }
    }
}

/// A shifted immediate value.
#[derive(Clone, Debug)]
pub struct ShiftedImm {
    bits: usize,
    shift: usize,
}

impl ShiftedImm {
    pub fn maybe_from_u64(mut val: u64) -> Option<ShiftedImm> {
        if val == 0 {
            Some(ShiftedImm { bits: 0, shift: 0 })
        } else {
            let mut shift = 0;
            while (val & 1) == 0 {
                shift += 1;
                val >>= 1;
            }
            if val < 0x1000 {
                // 12 bits
                Some(ShiftedImm {
                    bits: val as usize,
                    shift,
                })
            } else {
                None
            }
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

/// An extend operator for a register.
#[derive(Clone, Debug)]
pub enum ExtendOp {
    SXTB,
    SXTH,
    SXTW,
    XSTX,
    UXTB,
    UXTH,
    UXTW,
    UXTX,
}

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    Base(Value),
    BaseImm(Value, usize),
    BaseOffsetShifted(Value, usize),
    BaseImmPreIndexed(Value, usize),
    BaseImmPostIndexed(Value, usize),
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

impl MemArg {
    fn uses(&self) -> &[Value] {
        match self {
            &MemArg::Base(ref v) => slice::from_ref(v),
            &MemArg::BaseImm(ref v, ..) => slice::from_ref(v),
            &MemArg::BaseOffsetShifted(ref v, ..) => slice::from_ref(v),
            &MemArg::BaseImmPreIndexed(ref v, ..) => slice::from_ref(v),
            &MemArg::BaseImmPostIndexed(ref v, ..) => slice::from_ref(v),
            &MemArg::Label(..) => &[],
        }
    }

    fn defs(&self) -> &[Value] {
        match self {
            &MemArg::Base(..) |
            &MemArg::BaseImm(..) |
            &MemArg::BaseOffsetShifted(..) => &[],
            &MemArg::BaseImmPreIndexed(ref v, ..) => slice::from_ref(v),
            &MemArg::BaseImmPostIndexed(ref v, ..) => slice::from_ref(v),
            &MemArg::Label(..) => &[],
        }
    }
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
pub fn make_reg_reg(op: Op, rd: MachReg, rm: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rm)
}

/// Make a reg / reg / reg inst.
pub fn make_reg_reg_reg(op: Op, rd: MachReg, rn: MachReg, rm: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg_reg_use(Arg::Reg(), rm)
}

/// Make a reg / reg / reg / reg inst.
pub fn make_reg_reg_reg_reg(
    op: Op,
    rd: MachReg,
    rn: MachReg,
    rm: MachReg,
    ra: MachReg,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg_reg_use(Arg::Reg(), rm)
        .with_arg_reg_use(Arg::Reg(), ra)
}

/// Make a reg / reg / immediate inst.
pub fn make_reg_reg_imm(op: Op, rd: MachReg, rn: MachReg, imm: ShiftedImm) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg(Arg::Imm(imm))
}

/// Make a reg / reg / rshift inst.
pub fn make_reg_reg_rshift(
    op: Op,
    rd: MachReg,
    rn: MachReg,
    rm: MachReg,
    shift: ShiftOp,
    amt: usize,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg_reg_use(Arg::ShiftedReg(shift, amt), rm)
}

/// Make a reg / reg / rextend inst.
pub fn make_reg_reg_rextend(
    op: Op,
    rd: MachReg,
    rn: MachReg,
    rm: MachReg,
    ext: ExtendOp,
    shift_amt: usize,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg_reg_use(Arg::ExtendedReg(ext, shift_amt), rm)
}

/// Make a reg / memory-label inst.
pub fn make_reg_memlabel(op: Op, rd: MachReg, mem: MemArg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg(Arg::Mem(mem))
}

/// Make a reg / memory inst.
pub fn make_reg_mem(op: Op, rd: MachReg, mem: MemArg, rn: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Mem(mem), rn)
}

/// Make a reg / memory-update-addr inst.
pub fn make_reg_memupd(op: Op, rd: MachReg, mem: MemArg, rn: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use_def(Arg::Mem(mem), rn)
}

/// Make a reg / memory-2-reg-amode inst.
pub fn make_reg_mem2reg(
    op: Op,
    rd: MachReg,
    mem: MemArg,
    rn: MachReg,
    rm: MachReg,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_2reg(Arg::Mem(mem), rn, rm)
}

/// Make a memory / reg inst.
pub fn make_mem_reg(op: Op, mem: MemArg, rn: MachReg, rd: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_use(Arg::Mem(mem), rn)
        .with_arg_reg_use(Arg::Reg(), rd)
}

/// Make a memory-update-addr / reg inst.
pub fn make_memupd_reg(op: Op, mem: MemArg, rn: MachReg, rd: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_use_def(Arg::Mem(mem), rn)
        .with_arg_reg_use(Arg::Reg(), rd)
}

/// Make a memory-2-reg-amode / reg inst.
pub fn make_mem2reg_reg(
    op: Op,
    mem: MemArg,
    rn: MachReg,
    rm: MachReg,
    rd: MachReg,
) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_2reg(Arg::Mem(mem), rn, rm)
        .with_arg_reg_use(Arg::Reg(), rd)
}

/// Helper: load an arbitrary immediate (up to 64 bits large) into a register, using the smallest
/// possible encoding or constant pool entry.
pub fn load_imm<'a>(ctx: &mut LowerCtx<'a, Op, Arg>, value: u64, dest: MachReg) {
    if let Some(imm) = ShiftedImm::maybe_from_u64(value) {
        let xzr = ctx.fixed(31);
        ctx.emit(make_reg_reg_imm(Op::AddI, dest, xzr, imm));
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
pub fn with_imm12<'a, F>(ctx: &mut LowerCtx<'a, Op, Arg>, value: u64, f: F) -> bool
where
    F: FnOnce(&mut LowerCtx<'a, Op, Arg>, ShiftedImm),
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
