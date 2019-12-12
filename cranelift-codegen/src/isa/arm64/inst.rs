//! This module defines `Arm64Inst` and friends, which implement `MachInst`.

use crate::binemit::CodeSink;
use crate::machinst::*;
use smallvec::SmallVec;
use std::fmt::Debug;

type Arm64InstArgs = SmallVec<[Arm64Arg<Arm64InstRegIdx>; 3]>;
type Arm64InstRegs = SmallVec<[MachReg; 3]>;
type Arm64InstRegIdx = u8;

#[derive(Clone, Debug)]
struct Arm64Inst {
    pub op: Arm64Op,
    pub args: Arm64InstArgs,
    pub regs: Arm64InstRegs,
    pub cond: Option<Arm64Cond>,
}

macro_rules! arm64_ops {
    ($($op:ident),*) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        enum Arm64Op {
            $($op),*
        }

        impl Arm64Op {
            pub fn name(&self) -> &'static str {
                match self {
                    $(Arm64Op::$op => stringify!($op)),*
                }
            }
        }
    };
}

arm64_ops! {
    Add,
    AddS,
    Sub,
    SubS,
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
}

#[derive(Clone, Debug)]
enum Arm64Arg<Reg: Clone + Debug> {
    Imm(Arm64ShiftedImm),
    Reg(Reg),
    ShiftedReg(Reg, Arm64ShiftOp, usize),
    ExtendedReg(Reg, Arm64ExtendOp, usize),
    Mem(Arm64MemArg<Reg>),
}

impl<Reg: Clone + Debug> Arm64Arg<Reg> {
    fn map_reg<T, F>(&self, f: &mut F) -> Arm64Arg<T>
    where
        F: FnMut(&Reg) -> T,
        T: Clone + Debug,
    {
        match self {
            &Arm64Arg::Imm(ref imm) => Arm64Arg::Imm(imm.clone()),
            &Arm64Arg::Reg(ref r) => Arm64Arg::Reg(f(r)),
            &Arm64Arg::ShiftedReg(ref r, ref shiftop, shift) => {
                Arm64Arg::ShiftedReg(f(r), shiftop.clone(), shift)
            }
            &Arm64Arg::ExtendedReg(ref r, ref extendop, shift) => {
                Arm64Arg::ExtendedReg(f(r), extendop.clone(), shift)
            }
            &Arm64Arg::Mem(ref m) => Arm64Arg::Mem(m.map_reg(f)),
        }
    }
}

#[derive(Clone, Debug)]
struct Arm64ShiftedImm {
    bits: usize,
    shift: usize,
}

#[derive(Clone, Debug)]
enum Arm64ShiftOp {
    ASR,
    LSR,
    LSL,
    ROR,
}

#[derive(Clone, Debug)]
enum Arm64ExtendOp {
    SXTB,
    SXTH,
    SXTW,
    XSTX,
    UXTB,
    UXTH,
    UXTW,
    UXTX,
}

#[derive(Clone, Debug)]
enum Arm64MemArg<Reg: Clone + Debug> {
    Base(Reg),
    BaseImm(Reg, usize),
    BaseOffsetShifted(Reg, Reg, usize),
    BaseImmPreIndexed(Reg, usize),
    BaseImmPostIndexed(Reg, usize),
    PCRel(usize), // TODO: what is the right type for a label reference?
}

#[derive(Clone, Debug)]
enum Arm64Cond {
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

impl<Reg: Clone + Debug> Arm64MemArg<Reg> {
    fn map_reg<T, F>(&self, f: &mut F) -> Arm64MemArg<T>
    where
        F: FnMut(&Reg) -> T,
        T: Clone + Debug,
    {
        match self {
            &Arm64MemArg::Base(ref r) => Arm64MemArg::Base(f(r)),
            &Arm64MemArg::BaseImm(ref r, imm) => Arm64MemArg::BaseImm(f(r), imm),
            &Arm64MemArg::BaseOffsetShifted(ref r0, ref r1, shift) => {
                Arm64MemArg::BaseOffsetShifted(f(r0), f(r1), shift)
            }
            &Arm64MemArg::BaseImmPreIndexed(ref r, imm) => {
                Arm64MemArg::BaseImmPreIndexed(f(r), imm)
            }
            &Arm64MemArg::BaseImmPostIndexed(ref r, imm) => {
                Arm64MemArg::BaseImmPostIndexed(f(r), imm)
            }
            &Arm64MemArg::PCRel(off) => Arm64MemArg::PCRel(off),
        }
    }
}

// TODO: implement machine-instruction bits here. The MachInst infra is essentially:
//
// - A normalized interface to machine code (opaque insts with register slots) for regalloc and
//   binemit.
// - An API to the lowering patterns to allow principled emission of arm64 code.

impl MachInst for Arm64Inst {
    fn name(&self) -> &'static str {
        self.op.name()
    }
    fn num_regs(&self) -> usize {
        self.regs.len()
    }
    fn reg(&self, idx: usize) -> &MachReg {
        &self.regs[idx]
    }
    fn reg_mut(&mut self, idx: usize) -> &mut MachReg {
        &mut self.regs[idx]
    }
    fn size(&self) -> usize {
        // The joys of RISC: every ARM instruction is 4 bytes long.
        4
    }
    fn emit(&self, sink: &mut dyn CodeSink) {
        // TODO
    }
}

impl Arm64Inst {
    pub fn new(op: Arm64Op) -> Arm64Inst {
        Arm64Inst {
            op,
            args: SmallVec::new(),
            regs: SmallVec::new(),
            cond: None,
        }
    }

    pub fn add_arg(&mut self, arg: Arm64Arg<MachReg>) {
        let arg = arg.map_reg(&mut |r| {
            let idx = self.regs.len();
            self.regs.push(r.clone());
            idx as Arm64InstRegIdx
        });
        self.args.push(arg);
    }

    pub fn new_with_args(op: Arm64Op, args: &[Arm64Arg<MachReg>]) -> Arm64Inst {
        let mut inst = Arm64Inst::new(op);
        for arg in args {
            inst.add_arg(arg.clone());
        }
        inst
    }
    pub fn new_with_args_cc(op: Arm64Op, cc: Arm64Cond, args: &[Arm64Arg<MachReg>]) -> Arm64Inst {
        let mut inst = Arm64Inst::new_with_args(op, args);
        inst.cond = Some(cc);
        inst
    }
}
