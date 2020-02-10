//! This module defines x86_64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{Addend, CodeOffset, CodeSink, Reloc};
//zz use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{ConstantOffset, ExternalName, Function, JumpTable, SourceLoc, TrapCode};
use crate::ir::{FuncRef, GlobalValue, Type, Value};
use crate::isa::TargetIsa;
use crate::machinst::*;

use regalloc::InstRegUses;
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
// Registers, the Universe thereof, and printing

// These are the hardware encodings for various integer registers.
const ENC_RSP: u8 = 4;
const ENC_RBP: u8 = 5;
const ENC_R12: u8 = 12;
const ENC_R13: u8 = 13;

// These are ordered by sequence number, as required in the Universe.  The
// strange ordering is intended to make callee-save registers available before
// caller-saved ones.  This is a net win provided that each function makes at
// least one onward call.  It'll be a net loss for leaf functions, and we
// should change the ordering in that case, so as to make caller-save regs
// available first.
//
// FIXME JRS 2020Feb07: maybe have two different universes, one for leaf
// functions and one for non-leaf functions?  Also, they will have to be ABI
// dependent.  Need to find a way to avoid constructing a universe for each
// function we compile.

fn info_R12() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_R12, /*index=*/ 0).to_real_reg(),
        "%r12".to_string(),
    )
}
fn info_R13() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_R13, /*index=*/ 1).to_real_reg(),
        "%r13".to_string(),
    )
}
fn info_R14() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 14, /*index=*/ 2).to_real_reg(),
        "%r14".to_string(),
    )
}
fn info_R15() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 15, /*index=*/ 3).to_real_reg(),
        "%r15".to_string(),
    )
}
fn info_RBX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 3, /*index=*/ 4).to_real_reg(),
        "%rbx".to_string(),
    )
}

fn info_RSI() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 6, /*index=*/ 5).to_real_reg(),
        "%rsi".to_string(),
    )
}
fn info_RDI() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 7, /*index=*/ 6).to_real_reg(),
        "%rdi".to_string(),
    )
}
fn info_RAX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 0, /*index=*/ 7).to_real_reg(),
        "%rax".to_string(),
    )
}
fn info_RCX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 1, /*index=*/ 8).to_real_reg(),
        "%rcx".to_string(),
    )
}
fn info_RDX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 2, /*index=*/ 9).to_real_reg(),
        "%rdx".to_string(),
    )
}

fn info_R8() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 8, /*index=*/ 10).to_real_reg(),
        "%r8".to_string(),
    )
}
fn info_R9() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 9, /*index=*/ 11).to_real_reg(),
        "%r9".to_string(),
    )
}
fn info_R10() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 10, /*index=*/ 12).to_real_reg(),
        "%r10".to_string(),
    )
}
fn info_R11() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, /*enc=*/ 11, /*index=*/ 13).to_real_reg(),
        "%r11".to_string(),
    )
}

fn info_XMM0() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 0, /*index=*/ 14).to_real_reg(),
        "%xmm0".to_string(),
    )
}
fn info_XMM1() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 1, /*index=*/ 15).to_real_reg(),
        "%xmm1".to_string(),
    )
}
fn info_XMM2() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 2, /*index=*/ 16).to_real_reg(),
        "%xmm2".to_string(),
    )
}
fn info_XMM3() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 3, /*index=*/ 17).to_real_reg(),
        "%xmm3".to_string(),
    )
}
fn info_XMM4() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 4, /*index=*/ 18).to_real_reg(),
        "%xmm4".to_string(),
    )
}
fn info_XMM5() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 5, /*index=*/ 19).to_real_reg(),
        "%xmm5".to_string(),
    )
}
fn info_XMM6() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 6, /*index=*/ 20).to_real_reg(),
        "%xmm6".to_string(),
    )
}
fn info_XMM7() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 7, /*index=*/ 21).to_real_reg(),
        "%xmm7".to_string(),
    )
}
fn info_XMM8() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 8, /*index=*/ 22).to_real_reg(),
        "%xmm8".to_string(),
    )
}
fn info_XMM9() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 9, /*index=*/ 23).to_real_reg(),
        "%xmm9".to_string(),
    )
}
fn info_XMM10() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 10, /*index=*/ 24).to_real_reg(),
        "%xmm10".to_string(),
    )
}
fn info_XMM11() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 11, /*index=*/ 25).to_real_reg(),
        "%xmm11".to_string(),
    )
}
fn info_XMM12() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 12, /*index=*/ 26).to_real_reg(),
        "%xmm12".to_string(),
    )
}
fn info_XMM13() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 13, /*index=*/ 27).to_real_reg(),
        "%xmm13".to_string(),
    )
}
fn info_XMM14() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 14, /*index=*/ 28).to_real_reg(),
        "%xmm14".to_string(),
    )
}
fn info_XMM15() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::V128, /*enc=*/ 15, /*index=*/ 29).to_real_reg(),
        "%xmm15".to_string(),
    )
}

fn info_RSP() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_RSP, /*index=*/ 30).to_real_reg(),
        "%rsp".to_string(),
    )
}
fn info_RBP() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_RBP, /*index=*/ 31).to_real_reg(),
        "%rbp".to_string(),
    )
}

// For external consumption.  It's probably important that LLVM optimises
// these into a constant.
pub fn reg_RCX() -> Reg {
    info_RCX().0.to_reg()
}
pub fn reg_RSP() -> Reg {
    info_RSP().0.to_reg()
}

/// Create the register universe for X64.
pub fn create_reg_universe() -> RealRegUniverse {
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

// If |ireg| denotes an I64-classed reg, make a best-effort attempt to show
// its name at some smaller size (4, 2 or 1 bytes).
fn show_ireg_sized(reg: Reg, mb_rru: Option<&RealRegUniverse>, size: u8) -> String {
    let mut s = reg.show_rru(mb_rru);
    if reg.get_class() != RegClass::I64 || size == 8 {
        // We can't do any better.
        return s;
    }

    if reg.is_real() {
        // Change (eg) "rax" into "eax", "ax" or "al" as appropriate.  This is
        // something one could describe diplomatically as "a kludge", but it's
        // only debug code.
        let remapper = match s.as_str() {
            "%rax" => Some(["%eax", "%ax", "%al"]),
            "%rbx" => Some(["%ebx", "%bx", "%bl"]),
            "%rcx" => Some(["%ecx", "%cx", "%cl"]),
            "%rdx" => Some(["%edx", "%dx", "%dl"]),
            "%rsi" => Some(["%esi", "%si", "%sil"]),
            "%rdi" => Some(["%edi", "%di", "%dil"]),
            "%rbp" => Some(["%ebp", "%bp", "%bpl"]),
            "%rsp" => Some(["%esp", "%sp", "%spl"]),
            "%r8" => Some(["%r8d", "%r8w", "%r8b"]),
            "%r9" => Some(["%r9d", "%r9w", "%r9b"]),
            "%r10" => Some(["%r10d", "%r10w", "%r10b"]),
            "%r11" => Some(["%r11d", "%r11w", "%r11b"]),
            "%r12" => Some(["%r12d", "%r12w", "%r12b"]),
            "%r13" => Some(["%r13d", "%r13w", "%r13b"]),
            "%r14" => Some(["%r14d", "%r14w", "%r14b"]),
            "%r15" => Some(["%r15d", "%r15w", "%r15b"]),
            _ => None,
        };
        if let Some(smaller_names) = remapper {
            match size {
                4 => s = smaller_names[0].to_string(),
                2 => s = smaller_names[1].to_string(),
                1 => s = smaller_names[2].to_string(),
                _ => panic!("show_ireg_sized: real"),
            }
        }
    } else {
        // Add a "l", "w" or "b" suffix to RegClass::I64 vregs used at
        // narrower widths
        let suffix = match size {
            4 => "l",
            2 => "w",
            1 => "b",
            _ => panic!("show_ireg_sized: virtual"),
        };
        s = s + &suffix.to_string();
    }
    s
}

//=============================================================================
// Instruction sub-components: definitions and printing

// A Memory Address.  These denote a 64-bit value only.
#[derive(Clone)]
pub enum Addr {
    // sign-extend-32-to-64(Immediate) + Register
    IR {
        simm32: u32,
        base: Reg,
    },
    // sign-extend-32-to-64(Immediate) + Register1 + (Register2 << Shift)
    IRRS {
        simm32: u32,
        base: Reg,
        index: Reg,
        shift: u8, /* 0 .. 3 only */
    },
}
pub fn Addr_IR(simm32: u32, base: Reg) -> Addr {
    Addr::IR { simm32, base }
}
pub fn Addr_IRRS(simm32: u32, base: Reg, index: Reg, shift: u8) -> Addr {
    debug_assert!(shift <= 3);
    Addr::IRRS {
        simm32,
        base,
        index,
        shift,
    }
}
impl ShowWithRRU for Addr {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            Addr::IR { simm32, base } => format!("{}({})", *simm32 as i32, base.show_rru(mb_rru)),
            Addr::IRRS {
                simm32,
                base,
                index,
                shift,
            } => format!(
                "{}({},{},{})",
                *simm32 as i32,
                base.show_rru(mb_rru),
                index.show_rru(mb_rru),
                1 << shift
            ),
        }
    }
}

// An operand which is either an integer Register, a value in Memory or an
// Immediate.  This can denote an 8, 16, 32 or 64 bit value.  For the
// Immediate form, in the 8- and 16-bit case, only the lower 8 or 16 bits of
// |simm32| is relevant.  In the 64-bit case, the value denoted by |simm32| is
// its sign-extension out to 64 bits.
#[derive(Clone)]
pub enum RMI {
    R { reg: Reg },
    M { addr: Addr },
    I { simm32: u32 },
}
pub fn RMI_R(reg: Reg) -> RMI {
    RMI::R { reg }
}
pub fn RMI_M(addr: Addr) -> RMI {
    RMI::M { addr }
}
pub fn RMI_I(simm32: u32) -> RMI {
    RMI::I { simm32 }
}
impl ShowWithRRU for RMI {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        self.show_rru_sized(mb_rru, 8)
    }
    fn show_rru_sized(&self, mb_rru: Option<&RealRegUniverse>, size: u8) -> String {
        match self {
            RMI::R { reg } => show_ireg_sized(*reg, mb_rru, size),
            RMI::M { addr } => addr.show_rru(mb_rru),
            RMI::I { simm32 } => format!("${}", *simm32 as i32),
        }
    }
}

// An operand which is either an integer Register or a value in Memory.  This
// can denote an 8, 16, 32 or 64 bit value.
#[derive(Clone)]
pub enum RM {
    R { reg: Reg },
    M { addr: Addr },
}
pub fn RM_R(reg: Reg) -> RM {
    RM::R { reg }
}
pub fn RM_M(addr: Addr) -> RM {
    RM::M { addr }
}
impl ShowWithRRU for RM {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        self.show_rru_sized(mb_rru, 8)
    }
    fn show_rru_sized(&self, mb_rru: Option<&RealRegUniverse>, size: u8) -> String {
        match self {
            RM::R { reg } => show_ireg_sized(*reg, mb_rru, size),
            RM::M { addr } => addr.show_rru(mb_rru),
        }
    }
}

#[derive(Clone)]
// Some basic ALU operations.  TODO: maybe add Adc, Sbb.
pub enum RMI_R_Op {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Mul, // the signless, non-extending (N x N -> N, for N in {32,64}) variant
}
impl RMI_R_Op {
    fn to_string(&self) -> String {
        match self {
            RMI_R_Op::Add => "add".to_string(),
            RMI_R_Op::Sub => "sub".to_string(),
            RMI_R_Op::And => "and".to_string(),
            RMI_R_Op::Or => "or".to_string(),
            RMI_R_Op::Xor => "xor".to_string(),
            RMI_R_Op::Mul => "imul".to_string(),
        }
    }
}
impl fmt::Debug for RMI_R_Op {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

#[derive(Clone, PartialEq)]
// These indicate ways of extending (widening) a value, using the Intel
// naming: B(yte) = u8, W(ord) = u16, L(ong)word = u32, Q(uad)word = u64
pub enum ExtMode {
    BL, // B -> L
    BQ, // B -> Q
    WL, // W -> L
    WQ, // W -> Q
    LQ, // L -> Q
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
    fn dst_size(&self) -> u8 {
        match self {
            ExtMode::BL => 4,
            ExtMode::BQ => 8,
            ExtMode::WL => 4,
            ExtMode::WQ => 8,
            ExtMode::LQ => 8,
        }
    }
}
impl fmt::Debug for ExtMode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

// These indicate the form of a scalar shift: left, signed right, unsigned
// right.
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

// These indicate condition code tests.  Not all are represented since not all
// are useful in compiler-generated code.
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

//=============================================================================
// Instructions (top level): definition

/// Instructions.  Destinations are on the RIGHT (a la AT&T syntax).
#[derive(Clone)]
pub enum Inst {
    /// (add sub and or xor mul adc? sbb?) (32 64) (reg addr imm) reg
    Alu_RMI_R {
        is64: bool,
        op: RMI_R_Op,
        src: RMI,
        dst: Reg,
    },

    /// (imm32 imm64) reg.
    /// Either: movl $imm32, %reg32 or movabsq $imm64, %reg32
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

    /// movz (bl bq wl wq lq) addr reg (good for all ZX loads except 64->64).
    /// Note that the lq variant doesn't really exist since the default
    /// zero-extend rule makes it unnecessary.  For that case we emit the
    /// equivalent "movl AM, reg32".
    MovZX_M_R {
        extMode: ExtMode,
        addr: Addr,
        dst: Reg,
    },

    /// A plain 64-bit integer load, since MovXZ_M_R can't represent that
    Mov64_M_R {
        addr: Addr,
        dst: Reg,
    },

    /// movs (bl bq wl wq lq) addr reg (good for all SX loads)
    MovSX_M_R {
        extMode: ExtMode,
        addr: Addr,
        dst: Reg,
    },

    /// mov (b w l q) reg addr (good for all integer stores)
    Mov_R_M {
        size: u8, // 1, 2, 4 or 8
        src: Reg,
        addr: Addr,
    },

    /// (shl shr sar) (l q) imm reg
    Shift_R {
        is64: bool,
        kind: ShiftKind,
        nBits: u8, // 1 .. #bits-in-type - 1, or 0 to mean "%cl"
        dst: Reg,
    },

    /// cmp (b w l q) (reg addr imm) reg
    Cmp_RMI_R {
        size: u8, // 1, 2, 4 or 8
        src: RMI,
        dst: Reg,
    },

    /// pushq (reg addr imm)
    Push64 {
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

    // ret
    Ret {},
}

// Handy constructors for Insts.

// Will the lower 32 bits of the given value sign-extend back to the same value?
fn fitsIn32bits(x: u64) -> bool {
    let xs = x as i64;
    xs == ((xs << 32) >> 32)
}

// Same question for 8 bits.
fn fitsIn8bits(x: u32) -> bool {
    let xs = x as i32;
    xs == ((xs << 24) >> 24)
}

pub fn i_Alu_RMI_R(is64: bool, op: RMI_R_Op, src: RMI, dst: Reg) -> Inst {
    Inst::Alu_RMI_R { is64, op, src, dst }
}

pub fn i_Imm_R(dstIs64: bool, simm64: u64, dst: Reg) -> Inst {
    if !dstIs64 {
        debug_assert!(simm64 <= 0xFFFF_FFFF);
    }
    Inst::Imm_R {
        dstIs64,
        simm64,
        dst,
    }
}

pub fn i_Mov_R_R(is64: bool, src: Reg, dst: Reg) -> Inst {
    Inst::Mov_R_R { is64, src, dst }
}

pub fn i_MovZX_M_R(extMode: ExtMode, addr: Addr, dst: Reg) -> Inst {
    Inst::MovZX_M_R { extMode, addr, dst }
}

pub fn i_Mov64_M_R(addr: Addr, dst: Reg) -> Inst {
    Inst::Mov64_M_R { addr, dst }
}

pub fn i_MovSX_M_R(extMode: ExtMode, addr: Addr, dst: Reg) -> Inst {
    Inst::MovSX_M_R { extMode, addr, dst }
}

pub fn i_Mov_R_M(
    size: u8, // 1, 2, 4 or 8
    src: Reg,
    addr: Addr,
) -> Inst {
    debug_assert!(size == 8 || size == 4 || size == 2 || size == 1);
    Inst::Mov_R_M { size, src, addr }
}

pub fn i_Shift_R(
    is64: bool,
    kind: ShiftKind,
    nBits: u8, // 1 .. #bits-in-type - 1, or 0 to mean "%cl"
    dst: Reg,
) -> Inst {
    debug_assert!(nBits < if is64 { 64 } else { 32 });
    Inst::Shift_R {
        is64,
        kind,
        nBits,
        dst,
    }
}

pub fn i_Cmp_RMI_R(
    size: u8, // 1, 2, 4 or 8
    src: RMI,
    dst: Reg,
) -> Inst {
    debug_assert!(size == 8 || size == 4 || size == 2 || size == 1);
    Inst::Cmp_RMI_R { size, src, dst }
}

pub fn i_Push64(src: RMI) -> Inst {
    Inst::Push64 { src }
}

pub fn i_JmpKnown(simm32: u32) -> Inst {
    Inst::JmpKnown { simm32 }
}

pub fn i_JmpUnknown(target: RM) -> Inst {
    Inst::JmpUnknown { target }
}

pub fn i_JmpCond(cc: CC, tsimm32: u32, fsimm32: u32) -> Inst {
    Inst::JmpCond {
        cc,
        tsimm32,
        fsimm32,
    }
}

pub fn i_CallKnown(target: FuncRef) -> Inst {
    Inst::CallKnown { target }
}

pub fn i_CallUnknown(target: RM) -> Inst {
    Inst::CallUnknown { target }
}

pub fn i_Ret() -> Inst {
    Inst::Ret {}
}

//=============================================================================
// Instructions: printing

impl ShowWithRRU for Inst {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
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
            (if is64 { "q" } else { "l" }).to_string()
        }
        fn sizeLQ(is64: bool) -> u8 {
            if is64 {
                8
            } else {
                4
            }
        }
        fn suffixBWLQ(size: u8) -> String {
            match size {
                1 => "b".to_string(),
                2 => "w".to_string(),
                4 => "l".to_string(),
                8 => "q".to_string(),
                _ => panic!("Inst(x64).show.suffixBWLQ"),
            }
        }

        match self {
            Inst::Alu_RMI_R { is64, op, src, dst } => format!(
                "{} {}, {}",
                ljustify2(op.to_string(), suffixLQ(*is64)),
                src.show_rru_sized(mb_rru, sizeLQ(*is64)),
                show_ireg_sized(*dst, mb_rru, sizeLQ(*is64)),
            ),
            Inst::Imm_R {
                dstIs64,
                simm64,
                dst,
            } => {
                let name = if *dstIs64 { "movabsq" } else { "movl" };
                format!(
                    "{} ${}, {}",
                    ljustify(name.to_string()),
                    simm64,
                    show_ireg_sized(*dst, mb_rru, sizeLQ(*dstIs64))
                )
            }
            Inst::Mov_R_R { is64, src, dst } => format!(
                "{} {}, {}",
                ljustify2("mov".to_string(), suffixLQ(*is64)),
                show_ireg_sized(*src, mb_rru, sizeLQ(*is64)),
                show_ireg_sized(*dst, mb_rru, sizeLQ(*is64))
            ),
            Inst::MovZX_M_R { extMode, addr, dst } => {
                if *extMode == ExtMode::LQ {
                    format!(
                        "{} {}, {}",
                        ljustify("movl".to_string()),
                        addr.show_rru(mb_rru),
                        show_ireg_sized(*dst, mb_rru, 4)
                    )
                } else {
                    format!(
                        "{} {}, {}",
                        ljustify2("movz".to_string(), extMode.to_string()),
                        addr.show_rru(mb_rru),
                        show_ireg_sized(*dst, mb_rru, extMode.dst_size())
                    )
                }
            }
            Inst::Mov64_M_R { addr, dst } => format!(
                "{} {}, {}",
                ljustify("movq".to_string()),
                addr.show_rru(mb_rru),
                dst.show_rru(mb_rru)
            ),
            Inst::MovSX_M_R { extMode, addr, dst } => format!(
                "{} {}, {}",
                ljustify2("movs".to_string(), extMode.to_string()),
                addr.show_rru(mb_rru),
                show_ireg_sized(*dst, mb_rru, extMode.dst_size())
            ),
            Inst::Mov_R_M { size, src, addr } => format!(
                "{} {}, {}",
                ljustify2("mov".to_string(), suffixBWLQ(*size)),
                show_ireg_sized(*src, mb_rru, *size),
                addr.show_rru(mb_rru)
            ),
            Inst::Shift_R {
                is64,
                kind,
                nBits,
                dst,
            } => {
                if *nBits == 0 {
                    format!(
                        "{} %cl, {}",
                        ljustify2(kind.to_string(), suffixLQ(*is64)),
                        show_ireg_sized(*dst, mb_rru, sizeLQ(*is64))
                    )
                } else {
                    format!(
                        "{} ${}, {}",
                        ljustify2(kind.to_string(), suffixLQ(*is64)),
                        nBits,
                        show_ireg_sized(*dst, mb_rru, sizeLQ(*is64))
                    )
                }
            }
            Inst::Cmp_RMI_R { size, src, dst } => format!(
                "{} {}, {}",
                ljustify2("cmp".to_string(), suffixBWLQ(*size)),
                src.show_rru_sized(mb_rru, *size),
                show_ireg_sized(*dst, mb_rru, *size)
            ),
            Inst::Push64 { src } => {
                format!("{} {}", ljustify("pushq".to_string()), src.show_rru(mb_rru))
            }
            Inst::JmpKnown { simm32 } => {
                format!("{} simm32={}", ljustify("jmp".to_string()), *simm32)
            }
            Inst::JmpUnknown { target } => format!(
                "{} *{}",
                ljustify("jmp".to_string()),
                target.show_rru(mb_rru)
            ),
            Inst::JmpCond {
                cc,
                tsimm32,
                fsimm32,
            } => format!(
                "{} tsimm32={} fsimm32={}",
                ljustify2("j".to_string(), cc.to_string()),
                *tsimm32,
                *fsimm32
            ),
            Inst::CallKnown { target } => format!("{} {:?}", ljustify("call".to_string()), target),
            Inst::CallUnknown { target } => format!(
                "{} *{}",
                ljustify("call".to_string()),
                target.show_rru(mb_rru)
            ),
            Inst::Ret {} => "ret".to_string(),
        }
    }
}

// Temp hook for legacy printing machinery
impl fmt::Debug for Inst {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // Print the insn without a Universe :-(
        write!(fmt, "{}", self.show_rru(None))
    }
}

//=============================================================================
// Instructions and subcomponents: get_regs

impl Addr {
    // Add the regs mentioned by |self| to |set|.  The role in which they
    // appear (def/mod/use) is meaningless here, hence the use of plain |set|.
    fn get_regs(&self, set: &mut Set<Reg>) {
        match self {
            Addr::IR { simm32: _, base } => {
                set.insert(*base);
            }
            Addr::IRRS {
                simm32: _,
                base,
                index,
                shift: _,
            } => {
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
            RMI::M { addr } => addr.get_regs(set),
            RMI::I { simm32: _ } => {}
        }
    }
}

impl RM {
    // Add the regs mentioned by |self| to |set|.  Same comment as above.
    fn get_regs(&self, set: &mut Set<Reg>) {
        match self {
            RM::R { reg } => set.insert(*reg),
            RM::M { addr } => addr.get_regs(set),
        }
    }
}

fn x64_get_regs(inst: &Inst) -> InstRegUses {
    // This is a bit subtle.  If some register is in the modified set, then it
    // may not be in either the use or def sets.  However, enforcing that
    // directly is somewhat difficult.  Hence we postprocess the sets at the
    // end of this function.
    let mut iru = InstRegUses::new();

    match inst {
        Inst::Alu_RMI_R {
            is64: _,
            op: _,
            src,
            dst,
        } => {
            src.get_regs(&mut iru.used);
            iru.modified.insert(*dst);
        }
        Inst::Imm_R {
            dstIs64: _,
            simm64: _,
            dst,
        } => {
            iru.defined.insert(*dst);
        }
        Inst::Mov_R_R { is64: _, src, dst } => {
            iru.used.insert(*src);
            iru.defined.insert(*dst);
        }
        Inst::MovZX_M_R {
            extMode: _,
            addr,
            dst,
        } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(*dst);
        }
        Inst::Mov64_M_R { addr, dst } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(*dst);
        }
        Inst::MovSX_M_R {
            extMode: _,
            addr,
            dst,
        } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(*dst);
        }
        Inst::Mov_R_M { size: _, src, addr } => {
            iru.used.insert(*src);
            addr.get_regs(&mut iru.used);
        }
        Inst::Shift_R {
            is64: _,
            kind: _,
            nBits,
            dst,
        } => {
            if *nBits == 0 {
                iru.used.insert(reg_RCX());
            }
            iru.modified.insert(*dst);
        }
        Inst::Cmp_RMI_R { size: _, src, dst } => {
            src.get_regs(&mut iru.used);
            iru.used.insert(*dst); // yes, really |iru.used|
        }
        Inst::Push64 { src } => {
            src.get_regs(&mut iru.used);
            iru.modified.insert(reg_RSP());
        }
        Inst::JmpKnown { simm32: _ } => {}
        Inst::JmpUnknown { target } => {
            target.get_regs(&mut iru.used);
        }
        Inst::JmpCond {
            cc: _,
            tsimm32: _,
            fsimm32: _,
        } => {}
        Inst::CallKnown { target: _ } => {
            // FIXME add arg regs (iru.used) and caller-saved regs (iru.defined)
            unimplemented!();
        }
        Inst::CallUnknown { target } => {
            target.get_regs(&mut iru.used);
        }
        Inst::Ret {} => {}
    }

    // Enforce invariants described above.
    iru.defined.remove(&iru.modified);
    iru.used.remove(&iru.modified);

    iru
}

//=============================================================================
// Instructions and subcomponents: map_regs

fn apply_map(reg: &mut Reg, map: &RegallocMap<VirtualReg, RealReg>) {
    // The allocator interface conveniently does this for us.
    reg.apply_defs_or_uses(map)
}

fn apply_maps(
    reg: &mut Reg,
    pre_map__aka__map_uses: &RegallocMap<VirtualReg, RealReg>,
    post_map__aka__map_defs: &RegallocMap<VirtualReg, RealReg>,
) {
    // The allocator interface conveniently does this for us.  It also,
    // conveniently, panics if the two maps disagree -- that would imply a
    // serious error in the allocator or in our |get_regs| function.
    reg.apply_mods(post_map__aka__map_defs, pre_map__aka__map_uses)
}

impl Addr {
    fn apply_map(&mut self, map: &RegallocMap<VirtualReg, RealReg>) {
        match self {
            Addr::IR {
                simm32: _,
                ref mut base,
            } => apply_map(base, map),
            Addr::IRRS {
                simm32: _,
                ref mut base,
                ref mut index,
                shift: _,
            } => {
                apply_map(base, map);
                apply_map(index, map);
            }
        }
    }
}

impl RMI {
    fn apply_map(&mut self, map: &RegallocMap<VirtualReg, RealReg>) {
        match self {
            RMI::R { ref mut reg } => apply_map(reg, map),
            RMI::M { ref mut addr } => addr.apply_map(map),
            RMI::I { simm32: _ } => {}
        }
    }
}

impl RM {
    fn apply_map(&mut self, map: &RegallocMap<VirtualReg, RealReg>) {
        match self {
            RM::R { ref mut reg } => apply_map(reg, map),
            RM::M { ref mut addr } => addr.apply_map(map),
        }
    }
}

fn x64_map_regs(
    inst: &mut Inst,
    pre_map: &RegallocMap<VirtualReg, RealReg>,
    post_map: &RegallocMap<VirtualReg, RealReg>,
) {
    // For registers that appear in a read role, apply |pre_map|.  For those
    // that appear in a write role, apply |post_map|.  For registers that
    // appear in a modify role, the two maps *must* return the same result, so
    // we can apply either.
    //
    // Note that this function must be closely coordinated with |fn regs|, in
    // the sense that each arm "agrees" with the one in |fn regs| about which
    // fields are read, modifed or written.
    match inst {
        Inst::Alu_RMI_R {
            is64: _,
            op: _,
            ref mut src,
            ref mut dst,
        } => {
            src.apply_map(pre_map);
            apply_maps(dst, pre_map, post_map);
        }
        Inst::Imm_R {
            dstIs64: _,
            simm64: _,
            ref mut dst,
        } => {
            apply_map(dst, post_map);
        }
        Inst::Mov_R_R {
            is64: _,
            ref mut src,
            ref mut dst,
        } => {
            apply_map(src, pre_map);
            apply_map(dst, post_map);
        }
        Inst::MovZX_M_R {
            extMode: _,
            ref mut addr,
            ref mut dst,
        } => {
            addr.apply_map(pre_map);
            apply_map(dst, post_map);
        }
        Inst::Mov64_M_R { addr, dst } => {
            addr.apply_map(pre_map);
            apply_map(dst, post_map);
        }
        Inst::MovSX_M_R {
            extMode: _,
            ref mut addr,
            ref mut dst,
        } => {
            addr.apply_map(pre_map);
            apply_map(dst, post_map);
        }
        Inst::Mov_R_M {
            size: _,
            ref mut src,
            ref mut addr,
        } => {
            apply_map(src, pre_map);
            addr.apply_map(pre_map);
        }
        Inst::Shift_R {
            is64: _,
            kind: _,
            nBits: _,
            ref mut dst,
        } => {
            apply_maps(dst, pre_map, post_map);
        }
        Inst::Cmp_RMI_R {
            size: _,
            ref mut src,
            ref mut dst,
        } => {
            src.apply_map(pre_map);
            apply_map(dst, pre_map); // yes, really |pre_map|
        }
        Inst::Push64 { ref mut src } => {
            src.apply_map(pre_map);
        }
        Inst::JmpKnown { simm32: _ } => {}
        Inst::JmpUnknown { target } => {
            target.apply_map(pre_map);
        }
        Inst::JmpCond {
            cc: _,
            tsimm32: _,
            fsimm32: _,
        } => {}
        Inst::CallKnown { target: _ } => {}
        Inst::CallUnknown { target } => {
            target.apply_map(pre_map);
        }
        Inst::Ret {} => {}
    }
}

//=============================================================================
// Instructions and subcomponents: emission

// For all of the routines that take both a memory-or-reg operand (sometimes
// called "E" in the Intel parlance) and a reg-only operand ("G" in Intelese),
// the order is always G first, then E.

#[inline(always)]
fn mkModRegRM(m0d: u8, encRegG: u8, rmE: u8) -> u8 {
    debug_assert!(m0d < 4);
    debug_assert!(encRegG < 8);
    debug_assert!(rmE < 8);
    ((m0d & 3) << 6) | ((encRegG & 7) << 3) | (rmE & 7)
}

#[inline(always)]
fn mkSIB(shift: u8, encIndex: u8, encBase: u8) -> u8 {
    debug_assert!(shift < 4);
    debug_assert!(encIndex < 8);
    debug_assert!(encBase < 8);
    ((shift & 3) << 6) | ((encIndex & 7) << 3) | (encBase & 7)
}

#[inline(always)]
// Get the encoding number from something which we sincerely hope is a real
// register of class I64.
fn iregEnc(reg: Reg) -> u8 {
    debug_assert!(reg.is_real());
    debug_assert!(reg.get_class() == RegClass::I64);
    reg.get_hw_encoding()
}

// Emit the REX prefix byte even if it appears to be redundant (== 0x40).
const F_RETAIN_REDUNDANT_REX: u32 = 1;

// Set the W bit in the REX prefix to zero.  By default it will be set to 1,
// indicating a 64-bit operation.
const F_CLEAR_W_BIT: u32 = 2;

// This is the core 'emit' function for instructions that reference memory.
//
// For an instruction that has as operands a register |encG| and a memory
// address |memE|, create and emit, first the REX prefix, then caller-supplied
// opcode byte(s) (|opcodes| and |numOpcodes|), then the MOD/RM byte, then
// optionally, a SIB byte, and finally optionally an immediate that will be
// derived from the |memE| operand.  For most instructions up to and including
// SSE4.2, that will be the whole instruction.
//
// The opcodes are written bigendianly for the convenience of callers.  For
// example, if the opcode bytes to be emitted are, in this order, F3 0F 27,
// then the caller should pass |opcodes| == 0xF3_0F_27 and |numOpcodes| == 3.
//
// The register operand is represented here not as a |Reg| but as its hardware
// encoding, |encG|.  |flags| can specify special handling for the REX prefix.
// By default, the REX prefix will indicate a 64-bit operation and will be
// deleted if it is redundant (0x40).  Note that for a 64-bit operation, the
// REX prefix will normally never be redundant, since REX.W must be 1 to
// indicate a 64-bit operation.
fn emit_REX_OPCODES_MODRM_SIB_IMM_for_EncG_MemE<CS: CodeSink>(
    sink: &mut CS,
    opcodes: u32,
    mut numOpcodes: usize,
    encG: u8,
    memE: &Addr,
    flags: u32,
) {
    // General comment for this function: the registers in |memE| must be
    // 64-bit integer registers, because they are part of an address
    // expression.  But |encG| can be derived from a register of any class.
    let clearWBit = (flags & F_CLEAR_W_BIT) != 0;
    let retainRedundant = (flags & F_RETAIN_REDUNDANT_REX) != 0;
    match memE {
        Addr::IR { simm32, base: regE } => {
            // First, cook up the REX byte.  This is easy.
            let encE = iregEnc(*regE);
            let w = if clearWBit { 0 } else { 1 };
            let r = (encG >> 3) & 1;
            let x = 0;
            let b = (encE >> 3) & 1;
            let rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
            if rex != 0x40 || retainRedundant {
                sink.put1(rex);
            }
            // Now the opcode(s).  These include any other prefixes the caller
            // hands to us.
            while numOpcodes > 0 {
                sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
                numOpcodes -= 1;
            }
            // Now the mod/rm and associated immediates.  This is
            // significantly complicated due to the multiple special cases.
            if *simm32 == 0
                && encE != ENC_RSP
                && encE != ENC_RBP
                && encE != ENC_R12
                && encE != ENC_R13
            {
                sink.put1(mkModRegRM(0, encG & 7, encE & 7));
            } else if fitsIn8bits(*simm32) && encE != ENC_RSP && encE != ENC_R12 {
                sink.put1(mkModRegRM(1, encG & 7, encE & 7));
                sink.put1((simm32 & 0xFF) as u8);
            } else if encE != ENC_RSP && encE != ENC_R12 {
                sink.put1(mkModRegRM(2, encG & 7, encE & 7));
                sink.put4(*simm32);
            } else if (encE == ENC_RSP || encE == ENC_R12) && fitsIn8bits(*simm32) {
                // REX.B distinguishes RSP from R12
                sink.put1(mkModRegRM(1, encG & 7, 4));
                sink.put1(0x24);
                sink.put1((simm32 & 0xFF) as u8);
            } else if encE == ENC_R12 {
                // || encE == ENC_RSP .. wait for test case for RSP case
                // REX.B distinguishes RSP from R12
                sink.put1(mkModRegRM(2, encG & 7, 4));
                sink.put1(0x24);
                sink.put4(*simm32);
            } else {
                panic!("emit_REX_OPCODES_MODRM_SIB_IMM_for_EncG_MemE: IR");
            }
        }
        // Bizarrely, the IRRS case is much simpler.
        Addr::IRRS {
            simm32,
            base: regBase,
            index: regIndex,
            shift,
        } => {
            let encBase = iregEnc(*regBase);
            let encIndex = iregEnc(*regIndex);
            // The rex byte
            let w = if clearWBit { 0 } else { 1 };
            let r = (encG >> 3) & 1;
            let x = (encIndex >> 3) & 1;
            let b = (encBase >> 3) & 1;
            let rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
            if rex != 0x40 || retainRedundant {
                sink.put1(rex);
            }
            // All other prefixes and opcodes
            while numOpcodes > 0 {
                sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
                numOpcodes -= 1;
            }
            // modrm, SIB, immediates
            if fitsIn8bits(*simm32) && encIndex != ENC_RSP {
                sink.put1(mkModRegRM(1, encG & 7, 4));
                sink.put1(mkSIB(*shift, encIndex & 7, encBase & 7));
                sink.put1(*simm32 as u8);
            } else if encIndex != ENC_RSP {
                sink.put1(mkModRegRM(2, encG & 7, 4));
                sink.put1(mkSIB(*shift, encIndex & 7, encBase & 7));
                sink.put4(*simm32);
            } else {
                panic!("emit_REX_OPCODES_MODRM_SIB_IMM_for_EncG_MemE: IRRS");
            }
        }
    }
}

// This is the core 'emit' function for instructions that reference memory.
//
// This is conceptually the same as
// emit_REX_OPCODES_MODRM_SIB_IMM_for_EncG_MemE, except it is for the case
// where the E operand is a register rather than memory.  Hence it is much
// simpler.
fn emit_REX_OPCODES_MODRM_for_EncG_EncE<CS: CodeSink>(
    sink: &mut CS,
    opcodes: u32,
    mut numOpcodes: usize,
    encG: u8,
    encE: u8,
    flags: u32,
) {
    // EncG and EncE can be derived from registers of any class, and they
    // don't even have to be from the same class.  For example, for an
    // integer-to-FP conversion insn, one might be RegClass::I64 and the other
    // RegClass::V128.
    let clearWBit = (flags & F_CLEAR_W_BIT) != 0;
    let retainRedundant = (flags & F_RETAIN_REDUNDANT_REX) != 0;
    // The rex byte
    let w = if clearWBit { 0 } else { 1 };
    let r = (encG >> 3) & 1;
    let x = 0;
    let b = (encE >> 3) & 1;
    let rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
    if rex != 0x40 || retainRedundant {
        sink.put1(rex);
    }
    // All other prefixes and opcodes
    while numOpcodes > 0 {
        sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
        numOpcodes -= 1;
    }
    // Now the mod/rm byte.  The instruction we're generating doesn't access
    // memory, so there is no SIB byte or immediate -- we're done.
    sink.put1(mkModRegRM(3, encG & 7, encE));
}

// These are merely wrappers for the above two functions that facilitate passing
// actual |Reg|s rather than their encodings.
fn emit_REX_OPCODES_MODRM_SIB_IMM_for_RegG_MemE<CS: CodeSink>(
    sink: &mut CS,
    opcodes: u32,
    numOpcodes: usize,
    memE: &Addr,
    regG: Reg,
    flags: u32,
) {
    // JRS FIXME 2020Feb07: this should really just be |regEnc| not |iregEnc|
    let encG = iregEnc(regG);
    emit_REX_OPCODES_MODRM_SIB_IMM_for_EncG_MemE(sink, opcodes, numOpcodes, encG, memE, flags);
}

fn emit_REX_OPCODES_MODRM_for_RegG_RegE<CS: CodeSink>(
    sink: &mut CS,
    opcodes: u32,
    numOpcodes: usize,
    regG: Reg,
    regE: Reg,
    flags: u32,
) {
    // JRS FIXME 2020Feb07: these should really just be |regEnc| not |iregEnc|
    let encG = iregEnc(regG);
    let encE = iregEnc(regE);
    emit_REX_OPCODES_MODRM_for_EncG_EncE(sink, opcodes, numOpcodes, encG, encE, flags);
}

fn x64_emit<CS: CodeSink>(inst: &Inst, sink: &mut CS) {
    match inst {
        Inst::Ret {} => sink.put1(0xC3),
        _ => panic!("x64_emit"),
    }
}

//=============================================================================
// Instructions: misc functions and external interface

impl MachInst for Inst {
    fn get_regs(&self) -> InstRegUses {
        x64_get_regs(&self)
    }

    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        x64_map_regs(self, pre_map, post_map);
    }

    fn is_move(&self) -> Option<(Reg, Reg)> {
        // Note (carefully!) that a 32-bit mov *isn't* a no-op since it zeroes
        // out the upper 32 bits of the destination.  For example, we could
        // conceivably use |movl %reg, %reg| to zero out the top 32 bits of
        // %reg.
        match self {
            Inst::Mov_R_R { is64, src, dst } if *is64 => Some((*dst, *src)),
            _ => None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        unimplemented!()
    }

    fn gen_move(_to_reg: Reg, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_nop(_preferred_size: usize) -> Inst {
        unimplemented!()
    }

    fn maybe_direct_reload(&self, _reg: VirtualReg, _slot: SpillSlot) -> Option<Inst> {
        None
    }

    fn rc_for_type(ty: Type) -> RegClass {
        match ty {
            I8 | I16 | I32 | I64 | B1 | B8 | B16 | B32 | B64 => RegClass::I64,
            F32 | F64 => RegClass::V128,
            I128 | B128 => RegClass::V128,
            _ => panic!("Unexpected SSA-value type!"),
        }
    }

    fn gen_jump(_blockindex: BlockIndex) -> Inst {
        unimplemented!()
    }

    fn with_block_rewrites(&mut self, _block_target_map: &[BlockIndex]) {
        unimplemented!()
    }

    fn with_fallthrough_block(&mut self, _fallthrough: Option<BlockIndex>) {
        unimplemented!()
    }

    fn with_block_offsets(&mut self, _my_offset: CodeOffset, _targets: &[CodeOffset]) {
        unimplemented!()
    }

    fn reg_universe() -> RealRegUniverse {
        create_reg_universe()
    }
}

impl<CS: CodeSink> MachInstEmit<CS> for Inst {
    fn emit(&self, sink: &mut CS) {
        x64_emit(self, sink);
    }
}

//=============================================================================
// Tests for the emitter

// to see stdout: cargo test -- --nocapture
//
// for this specific case:
//
// (cd cranelift-codegen && \
// RUST_BACKTRACE=1 \
//       cargo test isa::x64::inst::test_x64_insn_encoding_and_printing \
//                  -- --nocapture)

#[cfg(test)]
mod test_utils {
    use super::*;
    use super::{Addend, CodeOffset, CodeSink, Reloc};

    pub struct TestCodeSink {
        bytes: Vec<u8>,
    }

    impl TestCodeSink {
        /// Create a new TestCodeSink.
        pub fn new() -> TestCodeSink {
            TestCodeSink { bytes: vec![] }
        }

        /// Return the size of emitted code so far.
        pub fn size(&self) -> usize {
            self.bytes.len()
        }

        /// This is pretty lame, but whatever ..
        pub fn stringify(&self) -> String {
            let mut s = "".to_string();
            for b in &self.bytes {
                s = s + &format!("{:02X}", b).to_string();
            }
            s
        }
    }

    impl CodeSink for TestCodeSink {
        fn offset(&self) -> CodeOffset {
            unimplemented!()
        }

        fn put1(&mut self, x: u8) {
            self.bytes.push(x);
        }

        fn put2(&mut self, x: u16) {
            self.bytes.push((x >> 0) as u8);
            self.bytes.push((x >> 8) as u8);
        }

        fn put4(&mut self, mut x: u32) {
            for _ in 0..4 {
                self.bytes.push(x as u8);
                x >>= 8;
            }
        }

        fn put8(&mut self, mut x: u64) {
            for _ in 0..8 {
                self.bytes.push(x as u8);
                x >>= 8;
            }
        }

        fn reloc_block(&mut self, _rel: Reloc, _block_offset: CodeOffset) {}

        fn reloc_external(&mut self, _rel: Reloc, _name: &ExternalName, _addend: Addend) {}

        fn reloc_constant(&mut self, _rel: Reloc, _constant_offset: ConstantOffset) {}

        fn reloc_jt(&mut self, _rel: Reloc, _jt: JumpTable) {}

        fn trap(&mut self, _code: TrapCode, _srcloc: SourceLoc) {}

        fn begin_jumptables(&mut self) {}

        fn begin_rodata(&mut self) {}

        fn end_codegen(&mut self) {}

        fn add_stackmap(&mut self, _val_list: &[Value], _func: &Function, _isa: &dyn TargetIsa) {}
    }
}

#[test]
fn test_x64_insn_encoding_and_printing() {
    println!("QQQQ BEGIN test_x64_insn_encoding_and_printing");

    let rax = info_RAX().0.to_reg();
    let rbx = info_RBX().0.to_reg();
    let rcx = info_RCX().0.to_reg();
    let rdx = info_RDX().0.to_reg();
    let rsi = info_RSI().0.to_reg();
    let rdi = info_RDI().0.to_reg();
    let rsp = info_RSP().0.to_reg();
    let rbp = info_RBP().0.to_reg();
    let r8 = info_R8().0.to_reg();
    let r9 = info_R9().0.to_reg();
    let r10 = info_R10().0.to_reg();
    let r11 = info_R11().0.to_reg();
    let r12 = info_R12().0.to_reg();
    let r13 = info_R13().0.to_reg();
    let r14 = info_R14().0.to_reg();
    let r15 = info_R15().0.to_reg();

    let mut insns = Vec::<(Inst, &str, &str)>::new();

    // Cases aimed at checking Addr-esses: IR (Imm + Reg)
    //
    // offset zero
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rax), rdi),
        "488B38",
        "movq    0(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rbx), rdi),
        "488B3B",
        "movq    0(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rcx), rdi),
        "488B39",
        "movq    0(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rdx), rdi),
        "488B3A",
        "movq    0(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rbp), rdi),
        "488B7D00",
        "movq    0(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rsp), rdi),
        "488B3C24",
        "movq    0(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rsi), rdi),
        "488B3E",
        "movq    0(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, rdi), rdi),
        "488B3F",
        "movq    0(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r8), rdi),
        "498B38",
        "movq    0(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r9), rdi),
        "498B39",
        "movq    0(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r10), rdi),
        "498B3A",
        "movq    0(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r11), rdi),
        "498B3B",
        "movq    0(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r12), rdi),
        "498B3C24",
        "movq    0(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r13), rdi),
        "498B7D00",
        "movq    0(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r14), rdi),
        "498B3E",
        "movq    0(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0, r15), rdi),
        "498B3F",
        "movq    0(%r15), %rdi",
    ));
    // offset simm8, hi
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rax), rdi),
        "488B787F",
        "movq    127(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rbx), rdi),
        "488B7B7F",
        "movq    127(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rcx), rdi),
        "488B797F",
        "movq    127(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rdx), rdi),
        "488B7A7F",
        "movq    127(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rbp), rdi),
        "488B7D7F",
        "movq    127(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rsp), rdi),
        "488B7C247F",
        "movq    127(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rsi), rdi),
        "488B7E7F",
        "movq    127(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, rdi), rdi),
        "488B7F7F",
        "movq    127(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r8), rdi),
        "498B787F",
        "movq    127(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r9), rdi),
        "498B797F",
        "movq    127(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r10), rdi),
        "498B7A7F",
        "movq    127(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r11), rdi),
        "498B7B7F",
        "movq    127(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r12), rdi),
        "498B7C247F",
        "movq    127(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r13), rdi),
        "498B7D7F",
        "movq    127(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r14), rdi),
        "498B7E7F",
        "movq    127(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(127, r15), rdi),
        "498B7F7F",
        "movq    127(%r15), %rdi",
    ));
    // offset simm8, lo
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rax), rdi),
        "488B7880",
        "movq    -128(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rbx), rdi),
        "488B7B80",
        "movq    -128(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rcx), rdi),
        "488B7980",
        "movq    -128(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rdx), rdi),
        "488B7A80",
        "movq    -128(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rbp), rdi),
        "488B7D80",
        "movq    -128(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rsp), rdi),
        "488B7C2480",
        "movq    -128(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rsi), rdi),
        "488B7E80",
        "movq    -128(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, rdi), rdi),
        "488B7F80",
        "movq    -128(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r8), rdi),
        "498B7880",
        "movq    -128(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r9), rdi),
        "498B7980",
        "movq    -128(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r10), rdi),
        "498B7A80",
        "movq    -128(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r11), rdi),
        "498B7B80",
        "movq    -128(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r12), rdi),
        "498B7C2480",
        "movq    -128(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r13), rdi),
        "498B7D80",
        "movq    -128(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r14), rdi),
        "498B7E80",
        "movq    -128(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-128i32 as u32, r15), rdi),
        "498B7F80",
        "movq    -128(%r15), %rdi",
    ));
    // offset simm32, minimal hi
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rax), rdi),
        "488BB880000000",
        "movq    128(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rbx), rdi),
        "488BBB80000000",
        "movq    128(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rcx), rdi),
        "488BB980000000",
        "movq    128(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rdx), rdi),
        "488BBA80000000",
        "movq    128(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rbp), rdi),
        "488BBD80000000",
        "movq    128(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rsp), rdi),
        "488BBC2480000000",
        "movq    128(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rsi), rdi),
        "488BBE80000000",
        "movq    128(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, rdi), rdi),
        "488BBF80000000",
        "movq    128(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r8), rdi),
        "498BB880000000",
        "movq    128(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r9), rdi),
        "498BB980000000",
        "movq    128(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r10), rdi),
        "498BBA80000000",
        "movq    128(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r11), rdi),
        "498BBB80000000",
        "movq    128(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r12), rdi),
        "498BBC2480000000",
        "movq    128(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r13), rdi),
        "498BBD80000000",
        "movq    128(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r14), rdi),
        "498BBE80000000",
        "movq    128(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(128, r15), rdi),
        "498BBF80000000",
        "movq    128(%r15), %rdi",
    ));
    // offset simm32, minimal lo
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rax), rdi),
        "488BB87FFFFFFF",
        "movq    -129(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rbx), rdi),
        "488BBB7FFFFFFF",
        "movq    -129(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rcx), rdi),
        "488BB97FFFFFFF",
        "movq    -129(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rdx), rdi),
        "488BBA7FFFFFFF",
        "movq    -129(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rbp), rdi),
        "488BBD7FFFFFFF",
        "movq    -129(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rsp), rdi),
        "488BBC247FFFFFFF",
        "movq    -129(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rsi), rdi),
        "488BBE7FFFFFFF",
        "movq    -129(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, rdi), rdi),
        "488BBF7FFFFFFF",
        "movq    -129(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r8), rdi),
        "498BB87FFFFFFF",
        "movq    -129(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r9), rdi),
        "498BB97FFFFFFF",
        "movq    -129(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r10), rdi),
        "498BBA7FFFFFFF",
        "movq    -129(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r11), rdi),
        "498BBB7FFFFFFF",
        "movq    -129(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r12), rdi),
        "498BBC247FFFFFFF",
        "movq    -129(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r13), rdi),
        "498BBD7FFFFFFF",
        "movq    -129(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r14), rdi),
        "498BBE7FFFFFFF",
        "movq    -129(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-129i32 as u32, r15), rdi),
        "498BBF7FFFFFFF",
        "movq    -129(%r15), %rdi",
    ));
    // offset simm32, large hi
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rax), rdi),
        "488BB877207317",
        "movq    393420919(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rbx), rdi),
        "488BBB77207317",
        "movq    393420919(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rcx), rdi),
        "488BB977207317",
        "movq    393420919(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rdx), rdi),
        "488BBA77207317",
        "movq    393420919(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rbp), rdi),
        "488BBD77207317",
        "movq    393420919(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rsp), rdi),
        "488BBC2477207317",
        "movq    393420919(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rsi), rdi),
        "488BBE77207317",
        "movq    393420919(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, rdi), rdi),
        "488BBF77207317",
        "movq    393420919(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r8), rdi),
        "498BB877207317",
        "movq    393420919(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r9), rdi),
        "498BB977207317",
        "movq    393420919(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r10), rdi),
        "498BBA77207317",
        "movq    393420919(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r11), rdi),
        "498BBB77207317",
        "movq    393420919(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r12), rdi),
        "498BBC2477207317",
        "movq    393420919(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r13), rdi),
        "498BBD77207317",
        "movq    393420919(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r14), rdi),
        "498BBE77207317",
        "movq    393420919(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(0x17732077, r15), rdi),
        "498BBF77207317",
        "movq    393420919(%r15), %rdi",
    ));
    // offset simm32, large lo
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rax), rdi),
        "488BB8D9A6BECE",
        "movq    -826366247(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rbx), rdi),
        "488BBBD9A6BECE",
        "movq    -826366247(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rcx), rdi),
        "488BB9D9A6BECE",
        "movq    -826366247(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rdx), rdi),
        "488BBAD9A6BECE",
        "movq    -826366247(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rbp), rdi),
        "488BBDD9A6BECE",
        "movq    -826366247(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rsp), rdi),
        "488BBC24D9A6BECE",
        "movq    -826366247(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rsi), rdi),
        "488BBED9A6BECE",
        "movq    -826366247(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rdi), rdi),
        "488BBFD9A6BECE",
        "movq    -826366247(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r8), rdi),
        "498BB8D9A6BECE",
        "movq    -826366247(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r9), rdi),
        "498BB9D9A6BECE",
        "movq    -826366247(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r10), rdi),
        "498BBAD9A6BECE",
        "movq    -826366247(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r11), rdi),
        "498BBBD9A6BECE",
        "movq    -826366247(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r12), rdi),
        "498BBC24D9A6BECE",
        "movq    -826366247(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r13), rdi),
        "498BBDD9A6BECE",
        "movq    -826366247(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r14), rdi),
        "498BBED9A6BECE",
        "movq    -826366247(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r15), rdi),
        "498BBFD9A6BECE",
        "movq    -826366247(%r15), %rdi",
    ));

    // Cases aimed at checking Addr-esses: IRRS (Imm + Reg + (Reg << Shift))
    // Note these don't check the case where the index reg is RSP, since we
    // don't encode any of those.
    //
    // offset simm8
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, rax, rax, 0), r11),
        "4C8B5C007F",
        "movq    127(%rax,%rax,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, rdi, rax, 1), r11),
        "4C8B5C477F",
        "movq    127(%rdi,%rax,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, r8, rax, 2), r11),
        "4D8B5C807F",
        "movq    127(%r8,%rax,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, r15, rax, 3), r11),
        "4D8B5CC77F",
        "movq    127(%r15,%rax,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, rax, rdi, 3), r11),
        "4C8B5CF87F",
        "movq    127(%rax,%rdi,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, rdi, rdi, 2), r11),
        "4C8B5CBF7F",
        "movq    127(%rdi,%rdi,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, r8, rdi, 1), r11),
        "4D8B5C787F",
        "movq    127(%r8,%rdi,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(127, r15, rdi, 0), r11),
        "4D8B5C3F7F",
        "movq    127(%r15,%rdi,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rax, r8, 2), r11),
        "4E8B5C8080",
        "movq    -128(%rax,%r8,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rdi, r8, 3), r11),
        "4E8B5CC780",
        "movq    -128(%rdi,%r8,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r8, r8, 0), r11),
        "4F8B5C0080",
        "movq    -128(%r8,%r8,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r15, r8, 1), r11),
        "4F8B5C4780",
        "movq    -128(%r15,%r8,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rax, r15, 1), r11),
        "4E8B5C7880",
        "movq    -128(%rax,%r15,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rdi, r15, 0), r11),
        "4E8B5C3F80",
        "movq    -128(%rdi,%r15,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r8, r15, 3), r11),
        "4F8B5CF880",
        "movq    -128(%r8,%r15,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r15, r15, 2), r11),
        "4F8B5CBF80",
        "movq    -128(%r15,%r15,4), %r11",
    ));
    // offset simm32
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, rax, rax, 0), r11),
        "4C8B9C00BE25664F",
        "movq    1332094398(%rax,%rax,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, rdi, rax, 1), r11),
        "4C8B9C47BE25664F",
        "movq    1332094398(%rdi,%rax,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, r8, rax, 2), r11),
        "4D8B9C80BE25664F",
        "movq    1332094398(%r8,%rax,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, r15, rax, 3), r11),
        "4D8B9CC7BE25664F",
        "movq    1332094398(%r15,%rax,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, rax, rdi, 3), r11),
        "4C8B9CF8BE25664F",
        "movq    1332094398(%rax,%rdi,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, rdi, rdi, 2), r11),
        "4C8B9CBFBE25664F",
        "movq    1332094398(%rdi,%rdi,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, r8, rdi, 1), r11),
        "4D8B9C78BE25664F",
        "movq    1332094398(%r8,%rdi,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(0x4f6625be, r15, rdi, 0), r11),
        "4D8B9C3FBE25664F",
        "movq    1332094398(%r15,%rdi,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rax, r8, 2), r11),
        "4E8B9C8070E9B2D9",
        "movq    -642586256(%rax,%r8,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rdi, r8, 3), r11),
        "4E8B9CC770E9B2D9",
        "movq    -642586256(%rdi,%r8,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r8, r8, 0), r11),
        "4F8B9C0070E9B2D9",
        "movq    -642586256(%r8,%r8,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r15, r8, 1), r11),
        "4F8B9C4770E9B2D9",
        "movq    -642586256(%r15,%r8,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rax, r15, 1), r11),
        "4E8B9C7870E9B2D9",
        "movq    -642586256(%rax,%r15,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rdi, r15, 0), r11),
        "4E8B9C3F70E9B2D9",
        "movq    -642586256(%rdi,%r15,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r8, r15, 3), r11),
        "4F8B9CF870E9B2D9",
        "movq    -642586256(%r8,%r15,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r15, r15, 2), r11),
        "4F8B9CBF70E9B2D9",
        "movq    -642586256(%r15,%r15,4), %r11",
    ));

    // Check sub-word printing of integer registers
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), rdx),
        "01CA",
        "addl    %ecx, %edx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_R(rcx), rdx),
        "4801CA",
        "addq    %rcx, %rdx",
    ));
    //
    insns.push((
        i_Imm_R(false, 1234567, r15),
        "41BF87D61200",
        "movl    $1234567, %r15d",
    ));
    insns.push((
        i_Imm_R(true, 1234567898765, r15),
        "49BF8D26FB711F010000",
        "movabsq $1234567898765, %r15",
    ));
    //
    insns.push((
        i_MovZX_M_R(ExtMode::BL, Addr_IR(-0x80i32 as u32, rax), r8),
        "440FB64080",
        "movzbl  -128(%rax), %r8d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "4C0FB64080",
        "movzbq  -128(%rax), %r8",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, Addr_IR(-0x80i32 as u32, rax), r8),
        "440FB74080",
        "movzwl  -128(%rax), %r8d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "4C0FB74080",
        "movzwq  -128(%rax), %r8",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "448B4080",
        "movl    -128(%rax), %r8d",
    ));
    //
    insns.push((
        i_MovSX_M_R(ExtMode::BL, Addr_IR(-0x80i32 as u32, rax), r8),
        "440FBE4080",
        "movsbl  -128(%rax), %r8d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "4C0FBE4080",
        "movsbq  -128(%rax), %r8",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, Addr_IR(-0x80i32 as u32, rax), r8),
        "440FBF4080",
        "movswl  -128(%rax), %r8d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "4C0FBF4080",
        "movswq  -128(%rax), %r8",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, Addr_IR(-0x80i32 as u32, rax), r8),
        "4C634080",
        "movslq  -128(%rax), %r8",
    ));
    //
    insns.push((
        i_Mov_R_M(1, rsi, Addr_IRRS(0x7F, rax, r8, 3)),
        "428874C07F",
        "movb    %sil, 127(%rax,%r8,8)",
    ));
    insns.push((
        i_Mov_R_M(2, rsi, Addr_IRRS(0x7F, rax, r8, 3)),
        "66428974C07F",
        "movw    %si, 127(%rax,%r8,8)",
    ));
    insns.push((
        i_Mov_R_M(4, rsi, Addr_IRRS(0x7F, rax, r8, 3)),
        "428974C07F",
        "movl    %esi, 127(%rax,%r8,8)",
    ));
    insns.push((
        i_Mov_R_M(8, rsi, Addr_IRRS(0x7F, rax, r8, 3)),
        "4A8974C07F",
        "movq    %rsi, 127(%rax,%r8,8)",
    ));
    //
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 0, rdi),
        "D3E7",
        "shll    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 5, rdi),
        "C1E705",
        "shll    $5, %edi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 5, rdi),
        "48C1E705",
        "shlq    $5, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 5, rdi),
        "48C1EF05",
        "shrq    $5, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 5, rdi),
        "48C1FF05",
        "sarq    $5, %rdi",
    ));
    //
    insns.push((i_Cmp_RMI_R(1, RMI_R(rcx), rdx), "38CA", "cmpb    %cl, %dl"));
    insns.push((
        i_Cmp_RMI_R(2, RMI_R(rcx), rdx),
        "6639CA",
        "cmpw    %cx, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_R(rcx), rdx),
        "39CA",
        "cmpl    %ecx, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_R(rcx), rdx),
        "4839CA",
        "cmpq    %rcx, %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx),
        "423A54C07F",
        "cmpb    127(%rax,%r8,8), %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx),
        "66423B54C07F",
        "cmpw    127(%rax,%r8,8), %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx),
        "423B54C07F",
        "cmpl    127(%rax,%r8,8), %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx),
        "4A3B54C07F",
        "cmpq    127(%rax,%r8,8), %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_I(123), rdx),
        "80FA7B",
        "cmpb    $123, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_I(456), rdx),
        "6681FAC801",
        "cmpw    $456, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_I(789), rdx),
        "81FA15030000",
        "cmpl    $789, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_I(-123i32 as u32), rdx),
        "4883FA85",
        "cmpq    $-123, %rdx",
    ));
    //
    insns.push((i_Push64(RMI_R(rcx)), "51", "pushq   %rcx"));
    insns.push((
        i_Push64(RMI_M(Addr_IR(0x7FFF_FFFF, rax))),
        "FFB0FFFFFF7F",
        "pushq   2147483647(%rax)",
    ));
    insns.push((i_Push64(RMI_I(456)), "68C8010000", "pushq   $456"));

    // Minimal general tests for each insn.  Don't forget to include cases
    // that test for removal of redundant REX prefixes.
    //
    // Alu_RMI_R
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_R(r15), rdx),
        "4C01FA",
        "addq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), r8),
        "4101C8",
        "addl    %ecx, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), rsi),
        "01CE",
        "addl    %ecx, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), rdx),
        "48035763",
        "addq    99(%rdi), %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), r8),
        "44034763",
        "addl    99(%rdi), %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), rsi),
        "037763",
        "addl    99(%rdi), %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_I(76543210), rdx),
        "4881C2EAF48F04",
        "addq    $76543210, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_I(-76543210i32 as u32), r8),
        "4181C0160B70FB",
        "addl    $-76543210, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_I(76543210), rsi),
        "81C6EAF48F04",
        "addl    $76543210, %esi",
    ));
    // This is pretty feeble
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Sub, RMI_R(r15), rdx),
        "4C29FA",
        "subq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::And, RMI_R(r15), rdx),
        "4C21FA",
        "andq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Or, RMI_R(r15), rdx),
        "4C09FA",
        "orq     %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Xor, RMI_R(r15), rdx),
        "4C31FA",
        "xorq    %r15, %rdx",
    ));
    // Test all mul cases, though
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_R(r15), rdx),
        "490FAFD7",
        "imulq   %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_R(rcx), r8),
        "440FAFC1",
        "imull   %ecx, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_R(rcx), rsi),
        "0FAFF1",
        "imull   %ecx, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), rdx),
        "480FAF5763",
        "imulq   99(%rdi), %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), r8),
        "440FAF4763",
        "imull   99(%rdi), %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), rsi),
        "0FAF7763",
        "imull   99(%rdi), %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_I(76543210), rdx),
        "4869D2EAF48F04",
        "imulq   $76543210, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_I(-76543210i32 as u32), r8),
        "4569C0160B70FB",
        "imull   $-76543210, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_I(76543210), rsi),
        "69F6EAF48F04",
        "imull   $76543210, %esi",
    ));
    //
    // Imm_R
    insns.push((
        i_Imm_R(false, 1234567, r14),
        "41BE87D61200",
        "movl    $1234567, %r14d",
    ));
    insns.push((
        i_Imm_R(true, 1234567898765, r14),
        "49BE8D26FB711F010000",
        "movabsq $1234567898765, %r14",
    ));
    insns.push((
        i_Imm_R(false, 1234567, rcx),
        "B987D61200",
        "movl    $1234567, %ecx",
    ));
    insns.push((
        i_Imm_R(true, 1234567898765, rsi),
        "48BE8D26FB711F010000",
        "movabsq $1234567898765, %rsi",
    ));
    //
    // Mov_R_R
    insns.push((i_Mov_R_R(false, rbx, rsi), "89DE", "movl    %ebx, %esi"));
    insns.push((i_Mov_R_R(false, rbx, r9), "4189D9", "movl    %ebx, %r9d"));
    insns.push((i_Mov_R_R(false, r11, rsi), "4489DE", "movl    %r11d, %esi"));
    insns.push((i_Mov_R_R(false, r12, r9), "4589E1", "movl    %r12d, %r9d"));
    insns.push((i_Mov_R_R(true, rbx, rsi), "4889DE", "movq    %rbx, %rsi"));
    insns.push((i_Mov_R_R(true, rbx, r9), "4989D9", "movq    %rbx, %r9"));
    insns.push((i_Mov_R_R(true, r11, rsi), "4C89DE", "movq    %r11, %rsi"));
    insns.push((i_Mov_R_R(true, r12, r9), "4D89E1", "movq    %r12, %r9"));
    //
    // MovZX_M_R
    insns.push((
        i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, rcx), rsi),
        "0FB671F9",
        "movzbl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r8), rbx),
        "410FB658F9",
        "movzbl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r10), r9),
        "450FB64AF9",
        "movzbl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r11), rdx),
        "410FB653F9",
        "movzbl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "480FB671F9",
        "movzbq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r8), rbx),
        "490FB658F9",
        "movzbq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r10), r9),
        "4D0FB64AF9",
        "movzbq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r11), rdx),
        "490FB653F9",
        "movzbq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, rcx), rsi),
        "0FB771F9",
        "movzwl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r8), rbx),
        "410FB758F9",
        "movzwl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r10), r9),
        "450FB74AF9",
        "movzwl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r11), rdx),
        "410FB753F9",
        "movzwl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "480FB771F9",
        "movzwq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r8), rbx),
        "490FB758F9",
        "movzwq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r10), r9),
        "4D0FB74AF9",
        "movzwq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r11), rdx),
        "490FB753F9",
        "movzwq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "8B71F9",
        "movl    -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r8), rbx),
        "418B58F9",
        "movl    -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r10), r9),
        "458B4AF9",
        "movl    -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r11), rdx),
        "418B53F9",
        "movl    -7(%r11), %edx",
    ));
    //
    // Mov64_M_R
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, rax, rbx, 0), rcx),
        "488B8C18B3000000",
        "movq    179(%rax,%rbx,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, rax, rbx, 0), r8),
        "4C8B8418B3000000",
        "movq    179(%rax,%rbx,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, rax, r9, 0), rcx),
        "4A8B8C08B3000000",
        "movq    179(%rax,%r9,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, rax, r9, 0), r8),
        "4E8B8408B3000000",
        "movq    179(%rax,%r9,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, r10, rbx, 0), rcx),
        "498B8C1AB3000000",
        "movq    179(%r10,%rbx,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, r10, rbx, 0), r8),
        "4D8B841AB3000000",
        "movq    179(%r10,%rbx,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, r10, r9, 0), rcx),
        "4B8B8C0AB3000000",
        "movq    179(%r10,%r9,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(Addr_IRRS(179, r10, r9, 0), r8),
        "4F8B840AB3000000",
        "movq    179(%r10,%r9,1), %r8",
    ));
    //
    // MovSX_M_R
    insns.push((
        i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, rcx), rsi),
        "0FBE71F9",
        "movsbl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r8), rbx),
        "410FBE58F9",
        "movsbl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r10), r9),
        "450FBE4AF9",
        "movsbl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r11), rdx),
        "410FBE53F9",
        "movsbl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "480FBE71F9",
        "movsbq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r8), rbx),
        "490FBE58F9",
        "movsbq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r10), r9),
        "4D0FBE4AF9",
        "movsbq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r11), rdx),
        "490FBE53F9",
        "movsbq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, rcx), rsi),
        "0FBF71F9",
        "movswl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r8), rbx),
        "410FBF58F9",
        "movswl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r10), r9),
        "450FBF4AF9",
        "movswl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r11), rdx),
        "410FBF53F9",
        "movswl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "480FBF71F9",
        "movswq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r8), rbx),
        "490FBF58F9",
        "movswq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r10), r9),
        "4D0FBF4AF9",
        "movswq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r11), rdx),
        "490FBF53F9",
        "movswq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, rcx), rsi),
        "486371F9",
        "movslq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r8), rbx),
        "496358F9",
        "movslq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r10), r9),
        "4D634AF9",
        "movslq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r11), rdx),
        "496353F9",
        "movslq  -7(%r11), %rdx",
    ));
    //
    // Mov_R_M.  Byte stores are tricky.  Check everything carefully.
    insns.push((
        i_Mov_R_M(8, rax, Addr_IR(99, rdi)),
        "48894763",
        "movq    %rax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(8, rbx, Addr_IR(99, r8)),
        "49895863",
        "movq    %rbx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(8, rcx, Addr_IR(99, rsi)),
        "48894E63",
        "movq    %rcx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(8, rdx, Addr_IR(99, r9)),
        "49895163",
        "movq    %rdx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(8, rsi, Addr_IR(99, rax)),
        "48897063",
        "movq    %rsi, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(8, rdi, Addr_IR(99, r15)),
        "49897F63",
        "movq    %rdi, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(8, rsp, Addr_IR(99, rcx)),
        "48896163",
        "movq    %rsp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(8, rbp, Addr_IR(99, r14)),
        "49896E63",
        "movq    %rbp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(8, r8, Addr_IR(99, rdi)),
        "4C894763",
        "movq    %r8, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(8, r9, Addr_IR(99, r8)),
        "4D894863",
        "movq    %r9, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(8, r10, Addr_IR(99, rsi)),
        "4C895663",
        "movq    %r10, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(8, r11, Addr_IR(99, r9)),
        "4D895963",
        "movq    %r11, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(8, r12, Addr_IR(99, rax)),
        "4C896063",
        "movq    %r12, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(8, r13, Addr_IR(99, r15)),
        "4D896F63",
        "movq    %r13, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(8, r14, Addr_IR(99, rcx)),
        "4C897163",
        "movq    %r14, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(8, r15, Addr_IR(99, r14)),
        "4D897E63",
        "movq    %r15, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(4, rax, Addr_IR(99, rdi)),
        "894763",
        "movl    %eax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(4, rbx, Addr_IR(99, r8)),
        "41895863",
        "movl    %ebx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(4, rcx, Addr_IR(99, rsi)),
        "894E63",
        "movl    %ecx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(4, rdx, Addr_IR(99, r9)),
        "41895163",
        "movl    %edx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(4, rsi, Addr_IR(99, rax)),
        "897063",
        "movl    %esi, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(4, rdi, Addr_IR(99, r15)),
        "41897F63",
        "movl    %edi, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(4, rsp, Addr_IR(99, rcx)),
        "896163",
        "movl    %esp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(4, rbp, Addr_IR(99, r14)),
        "41896E63",
        "movl    %ebp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(4, r8, Addr_IR(99, rdi)),
        "44894763",
        "movl    %r8d, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(4, r9, Addr_IR(99, r8)),
        "45894863",
        "movl    %r9d, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(4, r10, Addr_IR(99, rsi)),
        "44895663",
        "movl    %r10d, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(4, r11, Addr_IR(99, r9)),
        "45895963",
        "movl    %r11d, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(4, r12, Addr_IR(99, rax)),
        "44896063",
        "movl    %r12d, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(4, r13, Addr_IR(99, r15)),
        "45896F63",
        "movl    %r13d, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(4, r14, Addr_IR(99, rcx)),
        "44897163",
        "movl    %r14d, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(4, r15, Addr_IR(99, r14)),
        "45897E63",
        "movl    %r15d, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(2, rax, Addr_IR(99, rdi)),
        "66894763",
        "movw    %ax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(2, rbx, Addr_IR(99, r8)),
        "6641895863",
        "movw    %bx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(2, rcx, Addr_IR(99, rsi)),
        "66894E63",
        "movw    %cx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(2, rdx, Addr_IR(99, r9)),
        "6641895163",
        "movw    %dx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(2, rsi, Addr_IR(99, rax)),
        "66897063",
        "movw    %si, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(2, rdi, Addr_IR(99, r15)),
        "6641897F63",
        "movw    %di, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(2, rsp, Addr_IR(99, rcx)),
        "66896163",
        "movw    %sp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(2, rbp, Addr_IR(99, r14)),
        "6641896E63",
        "movw    %bp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(2, r8, Addr_IR(99, rdi)),
        "6644894763",
        "movw    %r8w, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(2, r9, Addr_IR(99, r8)),
        "6645894863",
        "movw    %r9w, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(2, r10, Addr_IR(99, rsi)),
        "6644895663",
        "movw    %r10w, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(2, r11, Addr_IR(99, r9)),
        "6645895963",
        "movw    %r11w, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(2, r12, Addr_IR(99, rax)),
        "6644896063",
        "movw    %r12w, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(2, r13, Addr_IR(99, r15)),
        "6645896F63",
        "movw    %r13w, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(2, r14, Addr_IR(99, rcx)),
        "6644897163",
        "movw    %r14w, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(2, r15, Addr_IR(99, r14)),
        "6645897E63",
        "movw    %r15w, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(1, rax, Addr_IR(99, rdi)),
        "884763",
        "movb    %al, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(1, rbx, Addr_IR(99, r8)),
        "41885863",
        "movb    %bl, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(1, rcx, Addr_IR(99, rsi)),
        "884E63",
        "movb    %cl, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(1, rdx, Addr_IR(99, r9)),
        "41885163",
        "movb    %dl, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(1, rsi, Addr_IR(99, rax)),
        "40887063",
        "movb    %sil, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(1, rdi, Addr_IR(99, r15)),
        "41887F63",
        "movb    %dil, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(1, rsp, Addr_IR(99, rcx)),
        "40886163",
        "movb    %spl, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(1, rbp, Addr_IR(99, r14)),
        "41886E63",
        "movb    %bpl, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(1, r8, Addr_IR(99, rdi)),
        "44884763",
        "movb    %r8b, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(1, r9, Addr_IR(99, r8)),
        "45884863",
        "movb    %r9b, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(1, r10, Addr_IR(99, rsi)),
        "44885663",
        "movb    %r10b, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(1, r11, Addr_IR(99, r9)),
        "45885963",
        "movb    %r11b, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(1, r12, Addr_IR(99, rax)),
        "44886063",
        "movb    %r12b, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(1, r13, Addr_IR(99, r15)),
        "45886F63",
        "movb    %r13b, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(1, r14, Addr_IR(99, rcx)),
        "44887163",
        "movb    %r14b, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(1, r15, Addr_IR(99, r14)),
        "45887E63",
        "movb    %r15b, 99(%r14)",
    ));
    //
    // Shift_R
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 0, rdi),
        "D3E7",
        "shll    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 1, r8),
        "41D1E0",
        "shll    $1, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 31, r13),
        "41C1E51F",
        "shll    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 0, rdi),
        "48D3E7",
        "shlq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 1, r8),
        "49D1E0",
        "shlq    $1, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 63, r13),
        "49C1E53F",
        "shlq    $63, %r13",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 0, rdi),
        "D3EF",
        "shrl    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 1, r8),
        "41D1E8",
        "shrl    $1, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 31, r13),
        "41C1ED1F",
        "shrl    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 0, rdi),
        "48D3EF",
        "shrq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 1, r8),
        "49D1E8",
        "shrq    $1, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 63, r13),
        "49C1ED3F",
        "shrq    $63, %r13",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 0, rdi),
        "D3FF",
        "sarl    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 1, r8),
        "41D1F8",
        "sarl    $1, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 31, r13),
        "41C1FD1F",
        "sarl    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 0, rdi),
        "48D3FF",
        "sarq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 1, r8),
        "49D1F8",
        "sarq    $1, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 63, r13),
        "49C1FD3F",
        "sarq    $63, %r13",
    ));
    //
    // Cmp_RMI_R
    insns.push((
        i_Cmp_RMI_R(8, RMI_R(r15), rdx),
        "4C39FA",
        "cmpq    %r15, %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_R(rcx), r8),
        "4939C8",
        "cmpq    %rcx, %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_R(rcx), rsi),
        "4839CE",
        "cmpq    %rcx, %rsi",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), rdx),
        "483B5763",
        "cmpq    99(%rdi), %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), r8),
        "4C3B4763",
        "cmpq    99(%rdi), %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), rsi),
        "483B7763",
        "cmpq    99(%rdi), %rsi",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_I(76543210), rdx),
        "4881FAEAF48F04",
        "cmpq    $76543210, %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_I(-76543210i32 as u32), r8),
        "4981F8160B70FB",
        "cmpq    $-76543210, %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, RMI_I(76543210), rsi),
        "4881FEEAF48F04",
        "cmpq    $76543210, %rsi",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(4, RMI_R(r15), rdx),
        "4439FA",
        "cmpl    %r15d, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_R(rcx), r8),
        "4139C8",
        "cmpl    %ecx, %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_R(rcx), rsi),
        "39CE",
        "cmpl    %ecx, %esi",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), rdx),
        "3B5763",
        "cmpl    99(%rdi), %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), r8),
        "443B4763",
        "cmpl    99(%rdi), %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), rsi),
        "3B7763",
        "cmpl    99(%rdi), %esi",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_I(76543210), rdx),
        "81FAEAF48F04",
        "cmpl    $76543210, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_I(-76543210i32 as u32), r8),
        "4181F8160B70FB",
        "cmpl    $-76543210, %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, RMI_I(76543210), rsi),
        "81FEEAF48F04",
        "cmpl    $76543210, %esi",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(2, RMI_R(r15), rdx),
        "664439FA",
        "cmpw    %r15w, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_R(rcx), r8),
        "664139C8",
        "cmpw    %cx, %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_R(rcx), rsi),
        "6639CE",
        "cmpw    %cx, %si",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), rdx),
        "663B5763",
        "cmpw    99(%rdi), %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), r8),
        "66443B4763",
        "cmpw    99(%rdi), %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), rsi),
        "663B7763",
        "cmpw    99(%rdi), %si",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_I(23210), rdx),
        "6681FAAA5A",
        "cmpw    $23210, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_I(-7654i32 as u32), r8),
        "664181F81AE2",
        "cmpw    $-7654, %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, RMI_I(7654), rsi),
        "6681FEE61D",
        "cmpw    $7654, %si",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(1, RMI_R(r15), rdx),
        "4438FA",
        "cmpb    %r15b, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_R(rcx), r8),
        "4138C8",
        "cmpb    %cl, %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_R(rcx), rsi),
        "4038CE",
        "cmpb    %cl, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), rdx),
        "3A5763",
        "cmpb    99(%rdi), %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), r8),
        "443A4763",
        "cmpb    99(%rdi), %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), rsi),
        "403A7763",
        "cmpb    99(%rdi), %sil",
    ));
    insns.push((i_Cmp_RMI_R(1, RMI_I(70), rdx), "80FA46", "cmpb    $70, %dl"));
    insns.push((
        i_Cmp_RMI_R(1, RMI_I(-76i32 as u32), r8),
        "4180F8B4",
        "cmpb    $-76, %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, RMI_I(76), rsi),
        "4080FE4C",
        "cmpb    $76, %sil",
    ));
    //
    // Push64
    insns.push((i_Push64(RMI_R(rdi)), "57", "pushq   %rdi"));
    insns.push((i_Push64(RMI_R(r8)), "4150", "pushq   %r8"));
    insns.push((
        i_Push64(RMI_M(Addr_IRRS(321, rsi, rcx, 3))),
        "FFB4CE41010000",
        "pushq   321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_Push64(RMI_M(Addr_IRRS(321, r9, rbx, 2))),
        "41FFB49941010000",
        "pushq   321(%r9,%rbx,4)",
    ));
    insns.push((i_Push64(RMI_I(0)), "6A00", "pushq   $0"));
    insns.push((i_Push64(RMI_I(127)), "6A7F", "pushq   $127"));
    insns.push((i_Push64(RMI_I(128)), "6880000000", "pushq   $128"));
    insns.push((
        i_Push64(RMI_I(0x31415927)),
        "6827594131",
        "pushq   $826366247",
    ));
    insns.push((i_Push64(RMI_I(-128i32 as u32)), "6A80", "pushq   $-128"));
    insns.push((
        i_Push64(RMI_I(-129i32 as u32)),
        "687FFFFFFF",
        "pushq   $-129",
    ));
    insns.push((
        i_Push64(RMI_I(-0x75c4e8a1i32 as u32)),
        "685F173B8A",
        "pushq   $-1975838881",
    ));
    //
    // JmpKnown skipped for now
    //
    // JmpUnknown
    insns.push((i_JmpUnknown(RM_R(rbp)), "FFE5", "jmp     *%rbp"));
    insns.push((i_JmpUnknown(RM_R(r11)), "41FFE3", "jmp     *%r11"));
    insns.push((
        i_JmpUnknown(RM_M(Addr_IRRS(321, rsi, rcx, 3))),
        "FFA4CE41010000",
        "jmp     *321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_JmpUnknown(RM_M(Addr_IRRS(321, r10, rdx, 2))),
        "41FFA49241010000",
        "jmp     *321(%r10,%rdx,4)",
    ));
    //
    // JmpCond skipped for now
    //
    // CallKnown skipped for now
    //
    // CallUnknown
    insns.push((i_CallUnknown(RM_R(rbp)), "FFD5", "call    *%rbp"));
    insns.push((i_CallUnknown(RM_R(r11)), "41FFD3", "call    *%r11"));
    insns.push((
        i_CallUnknown(RM_M(Addr_IRRS(321, rsi, rcx, 3))),
        "FF94CE41010000",
        "call    *321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_CallUnknown(RM_M(Addr_IRRS(321, r10, rdx, 2))),
        "41FF949241010000",
        "call    *321(%r10,%rdx,4)",
    ));
    //
    // Ret
    insns.push((i_Ret(), "C3", "ret"));

    // Actually run the tests
    let rru = create_reg_universe();
    for (insn, expected_encoding, expected_printing) in insns {
        println!("     {}", insn.show_rru(Some(&rru)));
        // Check the printed text is as expected.
        let actual_printing = insn.show_rru(Some(&rru));
        assert_eq!(expected_printing, actual_printing);

        // Check the encoding is as expected.
        let mut sink = test_utils::TestCodeSink::new();
        insn.emit(&mut sink);
        let actual_encoding = &sink.stringify();
        assert_eq!(expected_encoding, actual_encoding);
    }
    println!("QQQQ END test_x64_insn_encoding_and_printing");
}
