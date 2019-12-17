//! This module defines `Arm64Inst` and friends, which implement `MachInst`.

use crate::ir::{Inst, InstructionData, Type};
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::lower::LowerCtx;
use crate::machinst::*;
use crate::{mach_args, mach_ops};

mach_ops!(Op, {
    Add,
    AddS,
    AddI,
    Sub,
    SubS,
    SubI,
    Cmp,
    Cmn,
    Neg,
    NegS,
    Mov,
    MovI,
    And,
    AndS,
    Orr,
    Orn,
    Eor,
    Eon,
    Bic,
    BicS,
    Tst,
    Asr,
    Lsl,
    Lsr,
    Ror,
    Asrv,
    Lslv,
    Lsrv,
    Rorv,
    Cls,
    Clz,
    Adc,
    AdcS,
    Csel
});

mach_args!(Arg, ArgKind, {
    Imm(ShiftedImm),
    Reg(),
    ShiftedReg(ShiftOp, usize),
    ExtendedReg(ExtendOp, usize),
    Mem(MemArg)
});

impl MachInstArg for Arg {
    fn num_regs(&self) -> usize {
        match self {
            &Arg::Imm(..) => 0,
            &Arg::Reg(..) | &Arg::ShiftedReg(..) | &Arg::ExtendedReg(..) => 1,
            &Arg::Mem(ref m) => m.num_regs(),
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
    pub fn maybe_from_i64(mut val: i64) -> Option<ShiftedImm> {
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

    pub fn maybe_from_iconst(inst: &InstructionData) -> Option<ShiftedImm> {
        let imm: i64 = if let &InstructionData::UnaryImm { ref imm, .. } = inst {
            imm.clone().into()
        } else {
            return None;
        };
        ShiftedImm::maybe_from_i64(imm)
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
    Base,
    BaseImm(usize),
    BaseOffsetShifted(usize),
    BaseImmPreIndexed(usize),
    BaseImmPostIndexed(usize),
    PCRel(usize), // TODO: what is the right type for a label reference?
}

impl MemArg {
    fn num_regs(&self) -> usize {
        match self {
            &MemArg::Base => 1,
            &MemArg::BaseImm(..) => 1,
            &MemArg::BaseOffsetShifted(..) => 2,
            &MemArg::BaseImmPreIndexed(..) => 1,
            &MemArg::BaseImmPostIndexed(..) => 1,
            &MemArg::PCRel(..) => 0,
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
pub fn make_reg_reg_reg(op: Op, rd: MachReg, rn: MachReg, rm: MachReg) -> MachInst<Op, Arg> {
    MachInst::new(op)
        .with_arg_reg_def(Arg::Reg(), rd)
        .with_arg_reg_use(Arg::Reg(), rn)
        .with_arg_reg_use(Arg::Reg(), rm)
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

/// Helper: in a lowering action, check if an Iconst can become an Imm12; invoke a thunk with it if
/// so.
pub fn with_imm12<'a, F>(ctx: &mut LowerCtx<'a, Op, Arg>, inst: Inst, f: F) -> bool
where
    F: FnOnce(ShiftedImm),
{
    if let Some(imm) = ShiftedImm::maybe_from_iconst(ctx.inst(inst)) {
        f(imm);
        true
    } else {
        false
    }
}
