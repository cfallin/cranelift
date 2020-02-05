//! This module defines x86_64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{CodeOffset, CodeSink};
//zz use crate::ir::constant::{ConstantData, ConstantOffset};
//zz use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{Ebb, FuncRef, GlobalValue, Type};
use crate::machinst::*;

use regalloc::Map as RegallocMap;
use regalloc::Set;
use regalloc::{RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, NUM_REG_CLASSES};

use std::fmt;
use std::string::{String, ToString};
use std::vec::Vec;

use smallvec::SmallVec;
//zz use std::mem;
//zz use std::sync::Once;
//zz

//=============================================================================
// Registers and the Universe thereof

// These are ordered by sequence number, as required in the Universe.  The
// strange ordering is intended to make callee-save registers available before
// caller-saved ones.  This is a net win provided that each function makes at
// least one onward call.  It'll be a net loss for leaf functions, and we
// should change the ordering in that case, so as to make caller-save regs
// available first.

fn info_R12() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 12, /*index=*/ 0).to_real_reg(),
        "r12".to_string(),
    )
}
fn info_R13() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 13, /*index=*/ 1).to_real_reg(),
        "r13".to_string(),
    )
}
fn info_R14() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 14, /*index=*/ 2).to_real_reg(),
        "r14".to_string(),
    )
}
fn info_R15() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 15, /*index=*/ 3).to_real_reg(),
        "r15".to_string(),
    )
}
fn info_RBX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 3, /*index=*/ 4).to_real_reg(),
        "rbx".to_string(),
    )
}

fn info_RSI() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 6, /*index=*/ 5).to_real_reg(),
        "rsi".to_string(),
    )
}
fn info_RDI() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 7, /*index=*/ 6).to_real_reg(),
        "rdi".to_string(),
    )
}
fn info_RAX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 0, /*index=*/ 7).to_real_reg(),
        "rax".to_string(),
    )
}
fn info_RCX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 1, /*index=*/ 8).to_real_reg(),
        "rcx".to_string(),
    )
}
fn info_RDX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 2, /*index=*/ 9).to_real_reg(),
        "rdx".to_string(),
    )
}

fn info_R8() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 8, /*index=*/ 10).to_real_reg(),
        "r8".to_string(),
    )
}
fn info_R9() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 9, /*index=*/ 11).to_real_reg(),
        "r9".to_string(),
    )
}
fn info_R10() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 10, /*index=*/ 12).to_real_reg(),
        "r10".to_string(),
    )
}
fn info_R11() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 11, /*index=*/ 13).to_real_reg(),
        "r11".to_string(),
    )
}

fn info_XMM0() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 0, /*index=*/ 14).to_real_reg(),
        "xmm0".to_string(),
    )
}
fn info_XMM1() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 1, /*index=*/ 15).to_real_reg(),
        "xmm1".to_string(),
    )
}
fn info_XMM2() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 2, /*index=*/ 16).to_real_reg(),
        "xmm2".to_string(),
    )
}
fn info_XMM3() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 3, /*index=*/ 17).to_real_reg(),
        "xmm3".to_string(),
    )
}
fn info_XMM4() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 4, /*index=*/ 18).to_real_reg(),
        "xmm4".to_string(),
    )
}
fn info_XMM5() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 5, /*index=*/ 19).to_real_reg(),
        "xmm5".to_string(),
    )
}
fn info_XMM6() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 6, /*index=*/ 20).to_real_reg(),
        "xmm6".to_string(),
    )
}
fn info_XMM7() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 7, /*index=*/ 21).to_real_reg(),
        "xmm7".to_string(),
    )
}
fn info_XMM8() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 8, /*index=*/ 22).to_real_reg(),
        "xmm8".to_string(),
    )
}
fn info_XMM9() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 9, /*index=*/ 23).to_real_reg(),
        "xmm9".to_string(),
    )
}
fn info_XMM10() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 10, /*index=*/ 24).to_real_reg(),
        "xmm10".to_string(),
    )
}
fn info_XMM11() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 11, /*index=*/ 25).to_real_reg(),
        "xmm11".to_string(),
    )
}
fn info_XMM12() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 12, /*index=*/ 26).to_real_reg(),
        "xmm12".to_string(),
    )
}
fn info_XMM13() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 13, /*index=*/ 27).to_real_reg(),
        "xmm13".to_string(),
    )
}
fn info_XMM14() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 14, /*index=*/ 28).to_real_reg(),
        "xmm14".to_string(),
    )
}
fn info_XMM15() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 15, /*index=*/ 29).to_real_reg(),
        "xmm15".to_string(),
    )
}

fn info_RSP() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 4, /*index=*/ 30).to_real_reg(),
        "rsp".to_string(),
    )
}
fn info_RBP() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 5, /*index=*/ 31).to_real_reg(),
        "rbp".to_string(),
    )
}

// For external consumption.  It's probably important that LLVM optimises
// these into a constant.
pub fn reg_RCX() -> Reg {
    info_RCX().0.to_reg()
}

/// Create the register universe for X64.
pub fn get_reg_universe() -> RealRegUniverse {
    let mut regs = Vec::<(RealReg, String)>::new();
    let mut allocable_by_class = [None; NUM_REG_CLASSES];

    // Integer regs
    let mut base = regs.len();
    // Callee-saved, in the ELF x86_64 ABI
    regs.push(info_R12());
    regs.push(info_R13());
    regs.push(info_R14());
    regs.push(info_R15());
    regs.push(info_RBX());
    // Caller-saved, in the ELF x86_64 ABI
    regs.push(info_RSI());
    regs.push(info_RDI());
    regs.push(info_RAX());
    regs.push(info_RCX());
    regs.push(info_RDX());
    regs.push(info_R8());
    regs.push(info_R9());
    regs.push(info_R10());
    regs.push(info_R11());
    allocable_by_class[RegClass::I64.rc_to_usize()] = Some((base, regs.len() - 1));

    // XMM registers
    base = regs.len();
    regs.push(info_XMM0());
    regs.push(info_XMM1());
    regs.push(info_XMM2());
    regs.push(info_XMM3());
    regs.push(info_XMM4());
    regs.push(info_XMM5());
    regs.push(info_XMM6());
    regs.push(info_XMM7());
    regs.push(info_XMM8());
    regs.push(info_XMM9());
    regs.push(info_XMM10());
    regs.push(info_XMM11());
    regs.push(info_XMM12());
    regs.push(info_XMM13());
    regs.push(info_XMM14());
    regs.push(info_XMM15());
    allocable_by_class[RegClass::V128.rc_to_usize()] = Some((base, regs.len() - 1));

    // Other regs, not available to the allocator.
    let allocable = regs.len();
    regs.push(info_RSP());
    regs.push(info_RBP());

    RealRegUniverse {
        regs,
        allocable,
        allocable_by_class,
    }
}

//=============================================================================
// Instructions: definitions, also of supporting types

#[derive(Clone)]
pub enum AMode {
    IR {
        simm32: u32,
        base: Reg,
    },
    IRRS {
        simm32: u32,
        base: Reg,
        index: Reg,
        shift: u8, /* 0 .. 3 only */
    },
}
impl fmt::Debug for AMode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AMode::IR { simm32, base } => write!(fmt, "{}({:?})", *simm32 as i32, base),
            AMode::IRRS {
                simm32,
                base,
                index,
                shift,
            } => write!(
                fmt,
                "{}({:?},{:?},{})",
                *simm32 as i32,
                base,
                index,
                1 << shift
            ),
        }
    }
}

#[derive(Clone)]
pub enum RMI {
    R { reg: Reg },
    M { amode: AMode },
    I { simm32: u32 },
}
impl fmt::Debug for RMI {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RMI::R { reg } => reg.fmt(fmt),
            RMI::M { amode } => amode.fmt(fmt),
            RMI::I { simm32 } => write!(fmt, "{}", *simm32 as i32),
        }
    }
}

#[derive(Clone)]
pub enum RM {
    R { reg: Reg },
    M { amode: AMode },
}
impl fmt::Debug for RM {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RM::R { reg } => reg.fmt(fmt),
            RM::M { amode } => amode.fmt(fmt),
        }
    }
}

#[derive(Clone)]
pub enum RMI_R_Op {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Mul,
} // Also Adc, Sbb ?
impl RMI_R_Op {
    fn to_string(&self) -> String {
        match self {
            RMI_R_Op::Add => "add".to_string(),
            RMI_R_Op::Sub => "sub".to_string(),
            RMI_R_Op::And => "and".to_string(),
            RMI_R_Op::Or => "or".to_string(),
            RMI_R_Op::Xor => "xor".to_string(),
            RMI_R_Op::Mul => "mul".to_string(),
        }
    }
}
impl fmt::Debug for RMI_R_Op {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

#[derive(Clone)]
pub enum ExtMode {
    BL,
    BQ,
    WL,
    WQ,
    LQ,
}
impl ExtMode {
    fn to_string(&self) -> String {
        match self {
            ExtMode::BL => "bl".to_string(),
            ExtMode::BQ => "bq".to_string(),
            ExtMode::WL => "wl".to_string(),
            ExtMode::WQ => "wq".to_string(),
            ExtMode::LQ => "lq".to_string(),
        }
    }
}
impl fmt::Debug for ExtMode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

#[derive(Clone)]
pub enum ShiftKind {
    Left,
    RightS,
    RightZ,
}
impl ShiftKind {
    fn to_string(&self) -> String {
        match self {
            ShiftKind::Left => "shl".to_string(),
            ShiftKind::RightS => "sar".to_string(),
            ShiftKind::RightZ => "shr".to_string(),
        }
    }
}
impl fmt::Debug for ShiftKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

#[derive(Clone)]
pub enum CC {
    Z,
    NZ,
} // add more as needed
impl CC {
    fn to_string(&self) -> String {
        match self {
            CC::Z => "z".to_string(),
            CC::NZ => "nz".to_string(),
        }
    }
}
impl fmt::Debug for CC {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

/// Instructions.  Destinations are on the RIGHT (a la AT&T syntax).
#[derive(Clone)]
pub enum Inst {
    /// (add sub and or xor mul adc? sbb?) (32 64) (reg amode imm) reg
    Alu_RMI_R {
        is64: bool,
        op: RMI_R_Op,
        src: RMI,
        dst: Reg,
    },

    /// (imm32 imm64) reg
    Imm_R {
        dstIs64: bool,
        simm64: u64,
        dst: Reg,
    },

    /// mov (64 32) reg reg
    Mov_R_R {
        is64: bool,
        src: Reg,
        dst: Reg,
    },

    /// movz (bl bq wl wq lq) amode reg (good for all ZX loads except 64->64)
    MovZX_M_R {
        extMode: ExtMode,
        addr: AMode,
        dst: Reg,
    },

    /// A plain 64-bit integer load, since MovXZ_M_R can't represent that
    Mov64_M_R {
        addr: AMode,
        dst: Reg,
    },

    /// movs (bl bq wl wq lq) amode reg (good for all SX loads)
    MovSX_M_R {
        extMode: ExtMode,
        addr: AMode,
        dst: Reg,
    },

    /// mov (b w l q) reg amode (good for all integer stores)
    Mov_R_M {
        size: u8, // 1, 2, 4 or 8
        src: Reg,
        addr: AMode,
    },

    /// (shl shr sar) (l q) imm reg
    Shift_R {
        is64: bool,
        kind: ShiftKind,
        nBits: u8, // 1 .. #bits-in-type - 1, or 0 to mean "%cl"
        dst: Reg,
    },

    /// cmp (b w l q) (reg amode imm) reg
    Cmp_RMI_R {
        size: u8, // 1, 2, 4 or 8
        src: RMI,
        dst: Reg,
    },

    /// push (l q) (reg amode imm)
    Push {
        is64: bool,
        src: RMI,
    },

    /// jmp simm32
    JmpKnown {
        simm32: u32,
    },

    /// jmpq (reg mem)
    JmpUnknown {
        target: RM,
    },

    /// jcond cond simm32 simm32
    JmpCond {
        cc: CC,
        tsimm32: u32,
        fsimm32: u32,
    },

    /// call simm32
    CallKnown {
        target: FuncRef,
    },

    // callq (reg mem)
    CallUnknown {
        target: RM,
    },
    //zz
    //zz     /// A machine call instruction.
    //zz     Call { dest: FuncRef },
    //zz     /// A machine indirect-call instruction.
    //zz     CallInd { rn: Reg },
    //zz
    //zz     // ---- branches (exactly one must appear at end of BB) ----
    //zz     /// A machine return instruction.
    //zz     Ret {},
    //zz     /// An unconditional branch.
    //zz     Jump { dest: BranchTarget },
    //zz
    //zz     /// A conditional branch.
    //zz     CondBr {
    //zz         taken: BranchTarget,
    //zz         not_taken: BranchTarget,
    //zz         kind: CondBrKind,
    //zz     },
    //zz
    //zz     /// Lowered conditional branch: contains the original instruction, and a
    //zz     /// flag indicating whether to invert the taken-condition or not. Only one
    //zz     /// BranchTarget is retained, and the other is implicitly the next
    //zz     /// instruction, given the final basic-block layout.
    //zz     CondBrLowered {
    //zz         target: BranchTarget,
    //zz         inverted: bool,
    //zz         kind: CondBrKind,
    //zz     },
    //zz
    //zz     /// As for `CondBrLowered`, but represents a condbr/uncond-br sequence (two
    //zz     /// actual machine instructions). Needed when the final block layout implies
    //zz     /// that both arms of a conditional branch are not the fallthrough block.
    //zz     CondBrLoweredCompound {
    //zz         taken: BranchTarget,
    //zz         not_taken: BranchTarget,
    //zz         kind: CondBrKind,
    //zz     },
}

//=============================================================================
// Instructions: printing

impl fmt::Debug for Inst {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fn ljustify(s: String) -> String {
            let w = 7;
            if s.len() >= w {
                s
            } else {
                // BEGIN hack
                let mut need = w - s.len();
                if need > w {
                    need = w;
                }
                let extra = [" ", "  ", "   ", "    ", "     ", "      ", "       "][need - 1];
                // END hack
                s + &extra.to_string()
            }
        }
        fn ljustify2(s1: String, s2: String) -> String {
            ljustify(s1 + &s2)
        }
        fn suffixLQ(is64: bool) -> String {
            (if is64 { "l" } else { "q" }).to_string()
        }
        fn suffixBWLQ(size: u8) -> String {
            match size {
                1 => "1".to_string(),
                2 => "2".to_string(),
                4 => "4".to_string(),
                8 => "8".to_string(),
                _ => panic!("Inst(x64).fmt.suffixBWLQ"),
            }
        }

        match self {
            Inst::Alu_RMI_R { is64, op, src, dst } => write!(
                fmt,
                "{} {:?}, {:?}",
                ljustify2(op.to_string(), suffixLQ(*is64)),
                src,
                dst
            ),
            Inst::Imm_R {
                dstIs64,
                simm64,
                dst,
            } => {
                if *dstIs64 {
                    write!(
                        fmt,
                        "{} ${:?},{:?}",
                        ljustify("movabsq".to_string()),
                        simm64,
                        dst
                    )
                } else {
                    write!(
                        fmt,
                        "{} ${:?},{:?}",
                        ljustify("movl".to_string()),
                        simm64,
                        dst
                    )
                }
            }
            Inst::Mov_R_R { is64, src, dst } => write!(
                fmt,
                "{} {:?}, {:?}",
                ljustify2("mov".to_string(), suffixLQ(*is64)),
                src,
                dst
            ),
            Inst::MovZX_M_R { extMode, addr, dst } => write!(
                fmt,
                "{} {:?}, {:?}",
                ljustify2("movz".to_string(), extMode.to_string()),
                addr,
                dst
            ),
            Inst::Mov64_M_R { addr, dst } => write!(
                fmt, "{} {:?}, {:?}", ljustify("movq".to_string()), addr, dst),
            Inst::MovSX_M_R { extMode, addr, dst } => write!(
                fmt,
                "{} {:?}, {:?}",
                ljustify2("movs".to_string(), extMode.to_string()),
                addr,
                dst
            ),
            Inst::Mov_R_M { size, src, addr } => write!(
                fmt,
                "{} {:?}. {:?}",
                ljustify2("mov".to_string(), suffixBWLQ(*size)),
                src,
                addr
            ),
            Inst::Shift_R {
                is64,
                kind,
                nBits,
                dst,
            } => {
                if *nBits == 0 {
                    write!(
                        fmt,
                        "{} %cl, {:?}",
                        ljustify2(kind.to_string(), suffixLQ(*is64)),
                        dst
                    )
                } else {
                    write!(
                        fmt,
                        "{} ${}, {:?}",
                        ljustify2(kind.to_string(), suffixLQ(*is64)),
                        nBits,
                        dst
                    )
                }
            }
            Inst::Cmp_RMI_R { size, src, dst } => write!(
                fmt,
                "{} {:?}, {:?}",
                ljustify2("cmp".to_string(), suffixBWLQ(*size)),
                src,
                dst
            ),
            Inst::Push { is64, src } => write!(
                fmt,
                "{} {:?}",
                ljustify2("push".to_string(), suffixLQ(*is64)),
                src
            ),
            Inst::JmpKnown { simm32 } => write!(
                fmt,
                "{} simm32={}",
                ljustify("jmp".to_string()),
                *simm32 as i32
            ),
            Inst::JmpUnknown { target } => {
                write!(fmt, "{} {:?}", ljustify("jmp".to_string()), target)
            }
            Inst::JmpCond {
                cc,
                tsimm32,
                fsimm32,
            } => write!(
                fmt,
                "{} tsimm32={} fsimm32={}",
                ljustify2("j".to_string(), cc.to_string()),
                *tsimm32 as i32,
                *fsimm32 as i32
            ),
            Inst::CallKnown { target } =>
                write!(fmt, "{} {:?}", ljustify("call".to_string()), target),
            Inst::CallUnknown { target } =>
                write!(fmt, "{} {:?}", ljustify("call".to_string()), target)
        }
    }
}

//=============================================================================
// Instructions: get and map regs

impl AMode {
    // Add the regs mentioned by |self| to |set|.  The role in which they
    // appear (def/mod/use) is meaningless here, hence the use of plain |set|.
    fn get_regs(&self, set: &mut Set<Reg>) {
        match self {
            AMode::IR { simm32:_, base } => {
                set.insert(*base);
            },
            AMode::IRRS { simm32:_, base, index, shift:_ } => {
                set.insert(*base);
                set.insert(*index);
            }
        }
    }
}

impl RMI {
    // Add the regs mentioned by |self| to |set|.  Same comment as above.
    fn get_regs(&self, set: &mut Set<Reg>) {
        match self {
            RMI::R { reg } => set.insert(*reg),
            RMI::M { amode } => amode.get_regs(set),
            RMI::I { simm32:_ } => {}
        }
    }
}

impl RM {
    // Add the regs mentioned by |self| to |set|.  Same comment as above.
    fn get_regs(&self, set: &mut Set<Reg>) {
        match self {
            RM::R { reg } => set.insert(*reg),
            RM::M { amode } => amode.get_regs(set),
        }
    }
}

impl MachInst for Inst {
    fn regs(&self) -> MachInstRegs {
        // This is a bit subtle.  If some register is in the modified set,
        // then it may not be in either the use or def sets.  However,
        // enforcing that directly is somewhat difficult.  Hence we
        // postprocess the sets at the end of this function.
        let mut def = Set::<Reg>::empty();
        let mut m0d = Set::<Reg>::empty();
        let mut uce = Set::<Reg>::empty();

        match self {
            Inst::Alu_RMI_R { is64:_, op:_, src, dst } => {
                src.get_regs(&mut uce);
                m0d.insert(*dst);
            },
            Inst::Imm_R { dstIs64:_, simm64:_, dst } => {
                def.insert(*dst);
            },
            Inst::Mov_R_R { is64:_, src, dst } => {
                uce.insert(*src);
                def.insert(*dst);
            },
            Inst::MovZX_M_R { extMode:_, addr, dst } => {
                addr.get_regs(&mut uce);
                def.insert(*dst);
            },
            Inst::Mov64_M_R { addr, dst } => {
                addr.get_regs(&mut uce);
                def.insert(*dst);
            },
            Inst::MovSX_M_R { extMode:_, addr, dst } => {
                addr.get_regs(&mut uce);
                def.insert(*dst);
            },
            Inst::Mov_R_M { size:_, src, addr } => {
                uce.insert(*src);
                addr.get_regs(&mut uce);
            },
            Inst::Shift_R { is64:_, kind:_, nBits, dst } => {
                if *nBits == 0 {
                    uce.insert(reg_RCX());
                }
                m0d.insert(*dst);
            },
            Inst::Cmp_RMI_R { size:_, src, dst } => {
                src.get_regs(&mut uce);
                uce.insert(*dst); // yes, really |uce|
            },
            Inst::Push { is64:_, src } => {
                src.get_regs(&mut uce);
            },
            Inst::JmpKnown { simm32:_ } => {
            },
            Inst::JmpUnknown { target } => {
                target.get_regs(&mut uce);
            }
            Inst::JmpCond { cc:_, tsimm32:_, fsimm32:_ } => {
            },
            Inst::CallKnown { target:_ } => {
                // FIXME add arg regs (uce) and caller-saved regs (def)
                unimplemented!();
            },
            Inst::CallUnknown { target } => {
                target.get_regs(&mut uce);
            }
        }

        // Enforce invariants described above.
        def.remove(&m0d);
        uce.remove(&m0d);

        // (Interim) translate to the expected format
        let mut res = SmallVec::new();
        for r in def.iter() {
            res.push((*r, RegMode::Def));
        }
        for r in m0d.iter() {
            res.push((*r, RegMode::Modify));
        }
        for r in uce.iter() {
            res.push((*r, RegMode::Use));
        }

        res
    }

/*
        match self {
            Inst::Alu_RMI_R { is64, op, src, dst } => {
            },
            Inst::Imm_R { dstIs64, simm64, dst } => {
            },
            Inst::Mov_R_R { is64, src, dst } => {
                ret.push((src, RegMode::Use));
                ret.push((dst, RegMode::Def));
            },
            Inst::MovZX_M_R { extMode, addr, dst } => {
            },
            Inst::Mov64_M_R { addr, dst } => {
            },
            Inst::MovSX_M_R { extMode, addr, dst } => {
            },
            Inst::Mov_R_M { size, src, addr } => {
            },
            Inst::Shift_R { is64, kind, nBits, dst } => {
            },
            Inst::Cmp_RMI_R { size, src, dst } => {
            },
            Inst::Push { is64, src } => {
            },
            Inst::JmpKnown { simm32 } => {
            },
            Inst::JmpUnknown { target } => {
            }
            Inst::JmpCond { cc, tsimm32, fsimm32 } => {
            },
            Inst::CallKnown { target } => {
            },
            Inst::CallUnknown { target } => {
            }
        }
*/

    fn map_regs(
        &mut self,
        _pre_map: &RegallocMap<VirtualReg, RealReg>,
        _post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        unimplemented!()
        //zz         fn map(m: &RegallocMap<VirtualReg, RealReg>, r: Reg) -> Reg {
        //zz             if r.is_virtual() {
        //zz                 m.get(&r.to_virtual_reg()).cloned().unwrap().to_reg()
        //zz             } else {
        //zz                 r
        //zz             }
        //zz         }
        //zz
        //zz         fn map_mem(u: &RegallocMap<VirtualReg, RealReg>, mem: &MemArg) -> MemArg {
        //zz             // N.B.: we take only the pre-map here, but this is OK because the
        //zz             // only addressing modes that update registers (pre/post-increment on
        //zz             // ARM64, which we don't use yet but we may someday) both read and
        //zz             // write registers, so they are "mods" rather than "defs", so must be
        //zz             // the same in both the pre- and post-map.
        //zz             match mem {
        //zz                 &MemArg::Base(reg) => MemArg::Base(map(u, reg)),
        //zz                 &MemArg::BaseSImm9(reg, simm9) => MemArg::BaseSImm9(map(u, reg), simm9),
        //zz                 &MemArg::BaseUImm12Scaled(reg, uimm12) => {
        //zz                     MemArg::BaseUImm12Scaled(map(u, reg), uimm12)
        //zz                 }
        //zz                 &MemArg::BasePlusReg(r1, r2) => MemArg::BasePlusReg(map(u, r1), map(u, r2)),
        //zz                 &MemArg::BasePlusRegScaled(r1, r2, ty) => {
        //zz                     MemArg::BasePlusRegScaled(map(u, r1), map(u, r2), ty)
        //zz                 }
        //zz                 &MemArg::Label(ref l) => MemArg::Label(l.clone()),
        //zz             }
        //zz         }
        //zz
        //zz         fn map_br(u: &RegallocMap<VirtualReg, RealReg>, br: &CondBrKind) -> CondBrKind {
        //zz             match br {
        //zz                 &CondBrKind::Zero(reg) => CondBrKind::Zero(map(u, reg)),
        //zz                 &CondBrKind::NotZero(reg) => CondBrKind::NotZero(map(u, reg)),
        //zz                 &CondBrKind::Cond(c) => CondBrKind::Cond(c),
        //zz             }
        //zz         }
        //zz
        //zz         let u = pre_map; // For brevity below.
        //zz         let d = post_map;
        //zz
        //zz         let newval = match self {
        //zz             &mut Inst::AluRRR { alu_op, rd, rn, rm } => Inst::AluRRR {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 rm: map(u, rm),
        //zz             },
        //zz             &mut Inst::AluRRImm12 {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 ref imm12,
        //zz             } => Inst::AluRRImm12 {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 imm12: imm12.clone(),
        //zz             },
        //zz             &mut Inst::AluRRImmLogic {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 ref imml,
        //zz             } => Inst::AluRRImmLogic {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 imml: imml.clone(),
        //zz             },
        //zz             &mut Inst::AluRRImmShift {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 ref immshift,
        //zz             } => Inst::AluRRImmShift {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 immshift: immshift.clone(),
        //zz             },
        //zz             &mut Inst::AluRRRShift {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 rm,
        //zz                 ref shiftop,
        //zz             } => Inst::AluRRRShift {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 rm: map(u, rm),
        //zz                 shiftop: shiftop.clone(),
        //zz             },
        //zz             &mut Inst::AluRRRExtend {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 rm,
        //zz                 ref extendop,
        //zz             } => Inst::AluRRRExtend {
        //zz                 alu_op,
        //zz                 rd: map(d, rd),
        //zz                 rn: map(u, rn),
        //zz                 rm: map(u, rm),
        //zz                 extendop: extendop.clone(),
        //zz             },
        //zz             &mut Inst::ULoad8 { rd, ref mem } => Inst::ULoad8 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::SLoad8 { rd, ref mem } => Inst::SLoad8 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::ULoad16 { rd, ref mem } => Inst::ULoad16 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::SLoad16 { rd, ref mem } => Inst::SLoad16 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::ULoad32 { rd, ref mem } => Inst::ULoad32 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::SLoad32 { rd, ref mem } => Inst::SLoad32 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::ULoad64 { rd, ref mem } => Inst::ULoad64 {
        //zz                 rd: map(d, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::Store8 { rd, ref mem } => Inst::Store8 {
        //zz                 rd: map(u, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::Store16 { rd, ref mem } => Inst::Store16 {
        //zz                 rd: map(u, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::Store32 { rd, ref mem } => Inst::Store32 {
        //zz                 rd: map(u, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::Store64 { rd, ref mem } => Inst::Store64 {
        //zz                 rd: map(u, rd),
        //zz                 mem: map_mem(u, mem),
        //zz             },
        //zz             &mut Inst::MovZ { rd, ref imm } => Inst::MovZ {
        //zz                 rd: map(d, rd),
        //zz                 imm: imm.clone(),
        //zz             },
        //zz             &mut Inst::Jump { dest } => Inst::Jump { dest },
        //zz             &mut Inst::Call { dest } => Inst::Call { dest },
        //zz             &mut Inst::Ret {} => Inst::Ret {},
        //zz             &mut Inst::CallInd { rn } => Inst::CallInd { rn: map(u, rn) },
        //zz             &mut Inst::CondBr {
        //zz                 taken,
        //zz                 not_taken,
        //zz                 kind,
        //zz             } => Inst::CondBr {
        //zz                 taken,
        //zz                 not_taken,
        //zz                 kind: map_br(u, &kind),
        //zz             },
        //zz             &mut Inst::CondBrLowered {
        //zz                 target,
        //zz                 inverted,
        //zz                 kind,
        //zz             } => Inst::CondBrLowered {
        //zz                 target,
        //zz                 inverted,
        //zz                 kind: map_br(u, &kind),
        //zz             },
        //zz             &mut Inst::CondBrLoweredCompound {
        //zz                 taken,
        //zz                 not_taken,
        //zz                 kind,
        //zz             } => Inst::CondBrLoweredCompound {
        //zz                 taken,
        //zz                 not_taken,
        //zz                 kind: map_br(u, &kind),
        //zz             },
        //zz             &mut Inst::Nop => Inst::Nop,
        //zz             &mut Inst::Nop4 => Inst::Nop4,
        //zz             &mut Inst::LiveIns => Inst::LiveIns,
        //zz         };
        //zz         *self = newval;
    }

    fn is_move(&self) -> Option<(Reg, Reg)> {
        unimplemented!()
        //zz         match self {
        //zz             &Inst::AluRRR { alu_op, rd, rn, rm } if alu_op == ALUOp::Add64 && rn == zero_reg() => {
        //zz                 Some((rd, rm))
        //zz             }
        //zz             _ => None,
        //zz         }
    }

    fn is_term(&self) -> MachTerminator {
        unimplemented!()
        //zz         match self {
        //zz             &Inst::Ret {} => MachTerminator::Ret,
        //zz             &Inst::Jump { dest } => MachTerminator::Uncond(dest.as_block_index().unwrap()),
        //zz             &Inst::CondBr {
        //zz                 taken, not_taken, ..
        //zz             } => MachTerminator::Cond(
        //zz                 taken.as_block_index().unwrap(),
        //zz                 not_taken.as_block_index().unwrap(),
        //zz             ),
        //zz             _ => MachTerminator::None,
        //zz         }
    }

    fn get_spillslot_size(_rc: RegClass) -> u32 {
        unimplemented!()
        //zz         // We allocate in terms of 8-byte slots.
        //zz         match rc {
        //zz             RegClass::I64 => 1,
        //zz             RegClass::V128 => 2,
        //zz             _ => panic!("Unexpected register class!"),
        //zz         }
    }

    fn gen_spill(_to_slot: SpillSlot, _from_reg: RealReg) -> Inst {
        unimplemented!()
        //zz         let mem = MemArg::stackslot(to_slot.get());
        //zz         match from_reg.get_class() {
        //zz             RegClass::I64 => Inst::Store64 {
        //zz                 rd: from_reg.to_reg(),
        //zz                 mem,
        //zz             },
        //zz             RegClass::V128 => unimplemented!(),
        //zz             _ => panic!("Unexpected register class!"),
        //zz         }
    }

    fn gen_reload(_to_reg: RealReg, _from_slot: SpillSlot) -> Inst {
        unimplemented!()
        //zz         let mem = MemArg::stackslot(from_slot.get());
        //zz         match to_reg.get_class() {
        //zz             RegClass::I64 => Inst::ULoad64 {
        //zz                 rd: to_reg.to_reg(),
        //zz                 mem,
        //zz             },
        //zz             RegClass::V128 => unimplemented!(),
        //zz             _ => panic!("Unexpected register class!"),
        //zz         }
    }

    fn gen_move(_to_reg: Reg, _from_reg: Reg) -> Inst {
        unimplemented!()
        //zz         Inst::mov(to_reg, from_reg)
    }

    fn gen_nop(_preferred_size: usize) -> Inst {
        unimplemented!()
        //zz         // We can't give a NOP (or any insn) < 4 bytes.
        //zz         assert!(preferred_size >= 4);
        //zz         Inst::Nop4
    }

    fn maybe_direct_reload(&self, _reg: VirtualReg, _slot: SpillSlot) -> Option<Inst> {
        None
    }

    fn rc_for_type(_ty: Type) -> RegClass {
        unimplemented!()
        //zz         match ty {
        //zz             I8 | I16 | I32 | I64 | B1 | B8 | B16 | B32 | B64 => RegClass::I64,
        //zz             F32 | F64 => RegClass::V128,
        //zz             I128 | B128 => RegClass::V128,
        //zz             _ => panic!("Unexpected SSA-value type!"),
        //zz         }
    }

    fn gen_jump(_blockindex: BlockIndex) -> Inst {
        unimplemented!()
    }

    fn with_block_rewrites(&mut self, _block_target_map: &[BlockIndex]) {
        unimplemented!()
    }

    fn with_fallthrough_block(&mut self, _fallthrough: Option<BlockIndex>) {
        //zz         match self {
        //zz             &mut Inst::CondBr {
        //zz                 taken,
        //zz                 not_taken,
        //zz                 kind,
        //zz             } => {
        //zz                 if taken.as_block_index() == fallthrough {
        //zz                     *self = Inst::CondBrLowered {
        //zz                         target: not_taken,
        //zz                         inverted: true,
        //zz                         kind,
        //zz                     };
        //zz                 } else if not_taken.as_block_index() == fallthrough {
        //zz                     *self = Inst::CondBrLowered {
        //zz                         target: taken,
        //zz                         inverted: false,
        //zz                         kind,
        //zz                     };
        //zz                 } else {
        //zz                     // We need a compound sequence (condbr / uncond-br).
        //zz                     *self = Inst::CondBrLoweredCompound {
        //zz                         taken,
        //zz                         not_taken,
        //zz                         kind,
        //zz                     };
        //zz                 }
        //zz             }
        //zz             &mut Inst::Jump { dest } => {
        //zz                 if dest.as_block_index() == fallthrough {
        //zz                     *self = Inst::Nop;
        //zz                 }
        //zz             }
        //zz             _ => {}
        //zz         }
    }

    fn with_block_offsets(&mut self, _my_offset: CodeOffset, _targets: &[CodeOffset]) {
        unimplemented!()
        //zz         match self {
        //zz             &mut Inst::CondBrLowered { ref mut target, .. } => {
        //zz                 target.lower(targets, my_offset);
        //zz             }
        //zz             &mut Inst::CondBrLoweredCompound {
        //zz                 ref mut taken,
        //zz                 ref mut not_taken,
        //zz                 ..
        //zz             } => {
        //zz                 taken.lower(targets, my_offset);
        //zz                 not_taken.lower(targets, my_offset);
        //zz             }
        //zz             &mut Inst::Jump { ref mut dest } => {
        //zz                 dest.lower(targets, my_offset);
        //zz             }
        //zz             _ => {}
        //zz         }
    }

    fn reg_universe() -> RealRegUniverse {
        unimplemented!()
        //zz         get_reg_universe()
    }

    fn is_special_reg(_reg: RealReg) -> bool {
        false
    }
}

//zz impl BranchTarget {
//zz     fn lower(&mut self, targets: &[CodeOffset], my_offset: CodeOffset) {
//zz         match self {
//zz             &mut BranchTarget::Block(bix) => {
//zz                 let bix = bix as usize;
//zz                 assert!(bix < targets.len());
//zz                 let block_offset_in_func = targets[bix];
//zz                 let branch_offset = (block_offset_in_func as isize) - (my_offset as isize);
//zz                 *self = BranchTarget::ResolvedOffset(branch_offset);
//zz             }
//zz             &mut BranchTarget::ResolvedOffset(..) => {}
//zz         }
//zz     }
//zz
//zz     fn as_block_index(&self) -> Option<BlockIndex> {
//zz         match self {
//zz             &BranchTarget::Block(bix) => Some(bix),
//zz             _ => None,
//zz         }
//zz     }
//zz
//zz     fn as_offset(&self) -> Option<isize> {
//zz         match self {
//zz             &BranchTarget::ResolvedOffset(off) => Some(off),
//zz             _ => None,
//zz         }
//zz     }
//zz
//zz     fn as_off26(&self) -> Option<u32> {
//zz         self.as_offset().and_then(|i| {
//zz             if (i < (1 << 26)) && (i >= -(1 << 26)) {
//zz                 Some((i as u32) & ((1 << 26) - 1))
//zz             } else {
//zz                 None
//zz             }
//zz         })
//zz     }
//zz }
//zz
//zz fn machreg_to_gpr(m: Reg) -> u32 {
//zz     assert!(m.is_real());
//zz     m.to_real_reg().get_hw_encoding() as u32
//zz }
//zz
//zz fn enc_arith_rrr(bits_31_21: u16, bits_15_10: u8, rd: Reg, rn: Reg, rm: Reg) -> u32 {
//zz     ((bits_31_21 as u32) << 21)
//zz         | ((bits_15_10 as u32) << 10)
//zz         | machreg_to_gpr(rd)
//zz         | (machreg_to_gpr(rn) << 5)
//zz         | (machreg_to_gpr(rm) << 16)
//zz }
//zz
//zz fn enc_arith_rr_imm12(bits_31_24: u8, immshift: u8, imm12: u16, rn: Reg, rd: Reg) -> u32 {
//zz     ((bits_31_24 as u32) << 24)
//zz         | ((immshift as u32) << 22)
//zz         | ((imm12 as u32) << 10)
//zz         | (machreg_to_gpr(rn) << 5)
//zz         | machreg_to_gpr(rd)
//zz }
//zz
//zz fn enc_arith_rr_imml(bits_31_23: u16, imm_bits: u16, rn: Reg, rd: Reg) -> u32 {
//zz     ((bits_31_23 as u32) << 23)
//zz         | ((imm_bits as u32) << 10)
//zz         | (machreg_to_gpr(rn) << 5)
//zz         | machreg_to_gpr(rd)
//zz }
//zz
//zz fn enc_jump26(off_26_0: u32) -> u32 {
//zz     assert!(off_26_0 < (1 << 26));
//zz     (0b000101u32 << 26) | off_26_0
//zz }

impl<CS: CodeSink> MachInstEmit<CS> for Inst {
    fn emit(&self, _sink: &mut CS) {
        unimplemented!()
        //zz         match self {
        //zz             &Inst::AluRRR { alu_op, rd, rn, rm } => {
        //zz                 let top11 = match alu_op {
        //zz                     ALUOp::Add32 => 0b00001011_001,
        //zz                     ALUOp::Add64 => 0b10001011_001,
        //zz                     ALUOp::Sub32 => 0b01001011_001,
        //zz                     ALUOp::Sub64 => 0b11001011_001,
        //zz                     ALUOp::Orr32 => 0b00101010_000,
        //zz                     ALUOp::Orr64 => 0b10101010_000,
        //zz                     ALUOp::SubS32 => 0b01101011_001,
        //zz                     ALUOp::SubS64 => 0b11101011_001,
        //zz                     _ => unimplemented!(),
        //zz                 };
        //zz                 sink.put4(enc_arith_rrr(top11, 0b000_000, rd, rn, rm));
        //zz             }
        //zz             &Inst::AluRRImm12 {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 ref imm12,
        //zz             } => {
        //zz                 let top8 = match alu_op {
        //zz                     ALUOp::Add32 => 0b000_10001,
        //zz                     ALUOp::Add64 => 0b100_10001,
        //zz                     ALUOp::Sub32 => 0b010_10001,
        //zz                     ALUOp::Sub64 => 0b010_10001,
        //zz                     _ => unimplemented!(),
        //zz                 };
        //zz                 sink.put4(enc_arith_rr_imm12(
        //zz                     top8,
        //zz                     imm12.shift_bits(),
        //zz                     imm12.imm_bits(),
        //zz                     rn,
        //zz                     rd,
        //zz                 ));
        //zz             }
        //zz             &Inst::AluRRImmLogic {
        //zz                 alu_op,
        //zz                 rd,
        //zz                 rn,
        //zz                 ref imml,
        //zz             } => {
        //zz                 let top9 = match alu_op {
        //zz                     ALUOp::Orr32 => 0b001_100100,
        //zz                     ALUOp::Orr64 => 0b101_100100,
        //zz                     ALUOp::And32 => 0b000_100100,
        //zz                     ALUOp::And64 => 0b100_100100,
        //zz                     _ => unimplemented!(),
        //zz                 };
        //zz                 sink.put4(enc_arith_rr_imml(top9, imml.enc_bits(), rn, rd));
        //zz             }
        //zz             &Inst::AluRRImmShift { rd: _, rn: _, .. } => unimplemented!(),
        //zz             &Inst::AluRRRShift { rd: _, rn: _, rm: _, .. } => unimplemented!(),
        //zz             &Inst::AluRRRExtend { rd: _, rn: _, rm: _, .. } => unimplemented!(),
        //zz             &Inst::ULoad8 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::SLoad8 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::ULoad16 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::SLoad16 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::ULoad32 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::SLoad32 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::ULoad64 { rd: _, /*ref*/ mem: _, .. } => unimplemented!(),
        //zz             &Inst::Store8 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::Store16 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::Store32 { rd: _, /*ref*/ mem: _, .. }
        //zz             | &Inst::Store64 { rd: _, /*ref*/ mem: _, .. } => unimplemented!(),
        //zz             &Inst::MovZ { rd: _, .. } => unimplemented!(),
        //zz             &Inst::Jump { ref dest } => {
        //zz                 assert!(dest.as_off26().is_some());
        //zz                 sink.put4(enc_jump26(dest.as_off26().unwrap()));
        //zz             }
        //zz             &Inst::Ret {} => {
        //zz                 sink.put4(0xd65f03c0);
        //zz             }
        //zz             &Inst::Call { .. } => unimplemented!(),
        //zz             &Inst::CallInd { rn: _, .. } => unimplemented!(),
        //zz             &Inst::CondBr { .. } => panic!("Unlowered CondBr during binemit!"),
        //zz             &Inst::CondBrLowered { .. } => unimplemented!(),
        //zz             &Inst::CondBrLoweredCompound { .. } => unimplemented!(),
        //zz             &Inst::Nop => {}
        //zz             &Inst::Nop4 => {
        //zz                 sink.put4(0xd503201f);
        //zz             }
        //zz             &Inst::LiveIns => {}
        //zz         }
    }
}
