//! This module defines x86_64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{CodeOffset, CodeSink};
//zz use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{FuncRef, GlobalValue, Type};
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
    Ret { }

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
    Inst::Ret { }
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
            Inst::MovZX_M_R { extMode, addr, dst } =>
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
                },
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
            Inst::Push64 { src } => format!(
                "{} {}",
                ljustify("pushq".to_string()),
                src.show_rru(mb_rru)
            ),
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
            Inst::Ret { } => "ret".to_string()
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
        },
        Inst::Ret { } => { }
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
        Inst::Push64 {
            ref mut src,
        } => {
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
        },
        Inst::Ret { } => { }
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

fn x64_emit<CS: CodeSink>(_inst: &Inst, _sink: &mut CS) {
    unimplemented!()
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

#[cfg(test)]
mod test_utils {
    use super::*;
}

#[test]
fn i_am_a_test() {
    println!("QQQQ BEGIN I am a test");

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

    let mut insts = Vec::<Inst>::new();

    // Cases aimed at checking Addr-esses: IR (Imm + Reg)
    //
    // offset zero
    insts.push(i_Mov64_M_R(Addr_IR(0, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0, r15), rdi));
    // offset simm8, hi
    insts.push(i_Mov64_M_R(Addr_IR(127, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(127, r15), rdi));
    // offset simm8, lo
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-128i32 as u32, r15), rdi));
    // offset simm32, minimal hi
    insts.push(i_Mov64_M_R(Addr_IR(128, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(128, r15), rdi));
    // offset simm32, minimal lo
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-129i32 as u32, r15), rdi));
    // offset simm32, large hi
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(0x17732077, r15), rdi));
    // offset simm32, large lo
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rax), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rbx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rcx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rdx), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rbp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rsp), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rsi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, rdi), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r8), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r9), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r10), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r11), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r12), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r13), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r14), rdi));
    insts.push(i_Mov64_M_R(Addr_IR(-0x31415927i32 as u32, r15), rdi));

    // Cases aimed at checking Addr-esses: IRRS (Imm + Reg + (Reg << Shift))
    // Note these don't check the case where the index reg is RSP, since we
    // don't encode any of those.
    //
    // offset simm8
    insts.push(i_Mov64_M_R(Addr_IRRS(127, rax, rax, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, rdi, rax, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, r8,  rax, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, r15, rax, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, rax, rdi, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, rdi, rdi, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, r8,  rdi, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(127, r15, rdi, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rax, r8, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rdi, r8, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r8,  r8, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r15, r8, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rax, r15, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, rdi, r15, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r8,  r15, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-128i32 as u32, r15, r15, 2), r11));
    // offset simm32
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, rax, rax, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, rdi, rax, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, r8,  rax, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, r15, rax, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, rax, rdi, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, rdi, rdi, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, r8,  rdi, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(0x4f6625be, r15, rdi, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rax, r8, 2), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rdi, r8, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r8,  r8, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r15, r8, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rax, r15, 1), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, rdi, r15, 0), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r8,  r15, 3), r11));
    insts.push(i_Mov64_M_R(Addr_IRRS(-0x264d1690i32 as u32, r15, r15, 2), r11));

    // Check sub-word printing of integer registers
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), rdx));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_R(rcx), rdx));
    //
    insts.push(i_Imm_R(false, 1234567, r15));
    insts.push(i_Imm_R(true, 1234567898765, r15));
    //
    insts.push(i_MovZX_M_R(ExtMode::BL, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovZX_M_R(ExtMode::BQ, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovZX_M_R(ExtMode::WL, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovZX_M_R(ExtMode::WQ, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovZX_M_R(ExtMode::LQ, Addr_IR(-0x80i32 as u32, rax), r8));
    //
    insts.push(i_MovSX_M_R(ExtMode::BL, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovSX_M_R(ExtMode::BQ, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovSX_M_R(ExtMode::WL, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovSX_M_R(ExtMode::WQ, Addr_IR(-0x80i32 as u32, rax), r8));
    insts.push(i_MovSX_M_R(ExtMode::LQ, Addr_IR(-0x80i32 as u32, rax), r8));
    //
    insts.push(i_Mov_R_M(1, rsi, Addr_IRRS(0x7F, rax, r8, 3)));
    insts.push(i_Mov_R_M(2, rsi, Addr_IRRS(0x7F, rax, r8, 3)));
    insts.push(i_Mov_R_M(4, rsi, Addr_IRRS(0x7F, rax, r8, 3)));
    insts.push(i_Mov_R_M(8, rsi, Addr_IRRS(0x7F, rax, r8, 3)));
    //
    insts.push(i_Shift_R(false, ShiftKind::Left, 0, rdi));
    insts.push(i_Shift_R(false, ShiftKind::Left, 5, rdi));
    insts.push(i_Shift_R(true, ShiftKind::Left, 5, rdi));
    insts.push(i_Shift_R(true, ShiftKind::RightZ, 5, rdi));
    insts.push(i_Shift_R(true, ShiftKind::RightS, 5, rdi));
    //
    insts.push(i_Cmp_RMI_R(1, RMI_R(rcx), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_R(rcx), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_R(rcx), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_R(rcx), rdx));
    insts.push(i_Cmp_RMI_R(1, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_M(Addr_IRRS(0x7F, rax, r8, 3)), rdx));
    insts.push(i_Cmp_RMI_R(1, RMI_I(123), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_I(456), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_I(789), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_I(-123i32 as u32), rdx));
    //
    insts.push(i_Push64(RMI_R(rcx)));
    insts.push(i_Push64(RMI_M(Addr_IR(0x7FFF_FFFF, rax))));
    insts.push(i_Push64(RMI_I(456)));

    // Minimal general tests for each insn.  Don't forget to include cases
    // that test for removal of redundant REX prefixes.
    //
    // Alu_RMI_R
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_R(r15), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_R(rcx), rsi));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Add, RMI_I(76543210), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_I(-76543210i32 as u32), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Add, RMI_I(76543210), rsi));
    // This is pretty feeble
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Sub, RMI_R(r15), rdx));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::And, RMI_R(r15), rdx));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Or, RMI_R(r15), rdx));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Xor, RMI_R(r15), rdx));
    // Test all mul cases, though
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_R(r15), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_R(rcx), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_R(rcx), rsi));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Alu_RMI_R(true, RMI_R_Op::Mul, RMI_I(76543210), rdx));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_I(-76543210i32 as u32), r8));
    insts.push(i_Alu_RMI_R(false, RMI_R_Op::Mul, RMI_I(76543210), rsi));
    //
    // Imm_R
    insts.push(i_Imm_R(false, 1234567, r14));
    insts.push(i_Imm_R(true, 1234567898765, r14));
    insts.push(i_Imm_R(false, 1234567, rcx));
    insts.push(i_Imm_R(true, 1234567898765, rsi));
    //
    // Mov_R_R
    insts.push(i_Mov_R_R(false, rbx, rsi));
    insts.push(i_Mov_R_R(false, rbx, r9));
    insts.push(i_Mov_R_R(false, r11, rsi));
    insts.push(i_Mov_R_R(false, r12, r9));
    insts.push(i_Mov_R_R(true, rbx, rsi));
    insts.push(i_Mov_R_R(true, rbx, r9));
    insts.push(i_Mov_R_R(true, r11, rsi));
    insts.push(i_Mov_R_R(true, r12, r9));
    //
    // MovZX_M_R
    insts.push(i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovZX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovZX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovZX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovZX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovZX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r11), rdx));
    //
    // Mov64_M_R
    insts.push(i_Mov64_M_R(Addr_IRRS(179, rax, rbx, 0), rcx));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, rax, rbx, 0), r8));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, rax, r9, 0), rcx));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, rax, r9, 0), r8));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, r10, rbx, 0), rcx));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, r10, rbx, 0), r8));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, r10, r9, 0), rcx));
    insts.push(i_Mov64_M_R(Addr_IRRS(179, r10, r9, 0), r8));
    //
    // MovSX_M_R
    insts.push(i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovSX_M_R(ExtMode::BL, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovSX_M_R(ExtMode::BQ, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovSX_M_R(ExtMode::WL, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovSX_M_R(ExtMode::WQ, Addr_IR(-7i32 as u32, r11), rdx));
    insts.push(i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, rcx), rsi));
    insts.push(i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r8), rbx));
    insts.push(i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r10), r9));
    insts.push(i_MovSX_M_R(ExtMode::LQ, Addr_IR(-7i32 as u32, r11), rdx));
    //
    // Mov_R_M.  Byte stores are tricky.  Check everything carefully.
    insts.push(i_Mov_R_M(8, rax, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(8, rbx, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(8, rcx, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(8, rdx, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(8, rsi, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(8, rdi, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(8, rsp, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(8, rbp, Addr_IR(99, r14)));
    insts.push(i_Mov_R_M(8, r8, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(8, r9, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(8, r10, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(8, r11, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(8, r12, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(8, r13, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(8, r14, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(8, r15, Addr_IR(99, r14)));
    //
    insts.push(i_Mov_R_M(4, rax, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(4, rbx, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(4, rcx, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(4, rdx, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(4, rsi, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(4, rdi, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(4, rsp, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(4, rbp, Addr_IR(99, r14)));
    insts.push(i_Mov_R_M(4, r8, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(4, r9, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(4, r10, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(4, r11, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(4, r12, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(4, r13, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(4, r14, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(4, r15, Addr_IR(99, r14)));
    //
    insts.push(i_Mov_R_M(2, rax, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(2, rbx, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(2, rcx, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(2, rdx, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(2, rsi, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(2, rdi, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(2, rsp, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(2, rbp, Addr_IR(99, r14)));
    insts.push(i_Mov_R_M(2, r8, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(2, r9, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(2, r10, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(2, r11, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(2, r12, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(2, r13, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(2, r14, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(2, r15, Addr_IR(99, r14)));
    //
    insts.push(i_Mov_R_M(1, rax, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(1, rbx, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(1, rcx, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(1, rdx, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(1, rsi, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(1, rdi, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(1, rsp, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(1, rbp, Addr_IR(99, r14)));
    insts.push(i_Mov_R_M(1, r8, Addr_IR(99, rdi)));
    insts.push(i_Mov_R_M(1, r9, Addr_IR(99, r8)));
    insts.push(i_Mov_R_M(1, r10, Addr_IR(99, rsi)));
    insts.push(i_Mov_R_M(1, r11, Addr_IR(99, r9)));
    insts.push(i_Mov_R_M(1, r12, Addr_IR(99, rax)));
    insts.push(i_Mov_R_M(1, r13, Addr_IR(99, r15)));
    insts.push(i_Mov_R_M(1, r14, Addr_IR(99, rcx)));
    insts.push(i_Mov_R_M(1, r15, Addr_IR(99, r14)));
    //
    // Shift_R
    insts.push(i_Shift_R(false, ShiftKind::Left, 0, rdi));
    insts.push(i_Shift_R(false, ShiftKind::Left, 1, r8));
    insts.push(i_Shift_R(false, ShiftKind::Left, 31, r13));
    insts.push(i_Shift_R(true, ShiftKind::Left, 0, rdi));
    insts.push(i_Shift_R(true, ShiftKind::Left, 1, r8));
    insts.push(i_Shift_R(true, ShiftKind::Left, 63, r13));
    insts.push(i_Shift_R(false, ShiftKind::RightZ, 0, rdi));
    insts.push(i_Shift_R(false, ShiftKind::RightZ, 1, r8));
    insts.push(i_Shift_R(false, ShiftKind::RightZ, 31, r13));
    insts.push(i_Shift_R(true, ShiftKind::RightZ, 0, rdi));
    insts.push(i_Shift_R(true, ShiftKind::RightZ, 1, r8));
    insts.push(i_Shift_R(true, ShiftKind::RightZ, 63, r13));
    insts.push(i_Shift_R(false, ShiftKind::RightS, 0, rdi));
    insts.push(i_Shift_R(false, ShiftKind::RightS, 1, r8));
    insts.push(i_Shift_R(false, ShiftKind::RightS, 31, r13));
    insts.push(i_Shift_R(true, ShiftKind::RightS, 0, rdi));
    insts.push(i_Shift_R(true, ShiftKind::RightS, 1, r8));
    insts.push(i_Shift_R(true, ShiftKind::RightS, 63, r13));
    //
    // Cmp_RMI_R
    insts.push(i_Cmp_RMI_R(8, RMI_R(r15), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_R(rcx), r8));
    insts.push(i_Cmp_RMI_R(8, RMI_R(rcx), rsi));
    insts.push(i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Cmp_RMI_R(8, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Cmp_RMI_R(8, RMI_I(76543210), rdx));
    insts.push(i_Cmp_RMI_R(8, RMI_I(-76543210i32 as u32), r8));
    insts.push(i_Cmp_RMI_R(8, RMI_I(76543210), rsi));
    //
    insts.push(i_Cmp_RMI_R(4, RMI_R(r15), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_R(rcx), r8));
    insts.push(i_Cmp_RMI_R(4, RMI_R(rcx), rsi));
    insts.push(i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Cmp_RMI_R(4, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Cmp_RMI_R(4, RMI_I(76543210), rdx));
    insts.push(i_Cmp_RMI_R(4, RMI_I(-76543210i32 as u32), r8));
    insts.push(i_Cmp_RMI_R(4, RMI_I(76543210), rsi));
    //
    insts.push(i_Cmp_RMI_R(2, RMI_R(r15), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_R(rcx), r8));
    insts.push(i_Cmp_RMI_R(2, RMI_R(rcx), rsi));
    insts.push(i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Cmp_RMI_R(2, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Cmp_RMI_R(2, RMI_I(23210), rdx));
    insts.push(i_Cmp_RMI_R(2, RMI_I(-7654i32 as u32), r8));
    insts.push(i_Cmp_RMI_R(2, RMI_I(7654), rsi));
    //
    insts.push(i_Cmp_RMI_R(1, RMI_R(r15), rdx));
    insts.push(i_Cmp_RMI_R(1, RMI_R(rcx), r8));
    insts.push(i_Cmp_RMI_R(1, RMI_R(rcx), rsi));
    insts.push(i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), rdx));
    insts.push(i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), r8));
    insts.push(i_Cmp_RMI_R(1, RMI_M(Addr_IR(99, rdi)), rsi));
    insts.push(i_Cmp_RMI_R(1, RMI_I(70), rdx));
    insts.push(i_Cmp_RMI_R(1, RMI_I(-76i32 as u32), r8));
    insts.push(i_Cmp_RMI_R(1, RMI_I(76), rsi));
    //
    // Push64
    insts.push(i_Push64(RMI_R(rdi)));
    insts.push(i_Push64(RMI_R(r8)));
    insts.push(i_Push64(RMI_M(Addr_IRRS(321, rsi, rcx, 3))));
    insts.push(i_Push64(RMI_M(Addr_IRRS(321, r9, rbx, 2))));
    insts.push(i_Push64(RMI_I(0)));
    insts.push(i_Push64(RMI_I(127)));
    insts.push(i_Push64(RMI_I(128)));
    insts.push(i_Push64(RMI_I(0x31415927)));
    insts.push(i_Push64(RMI_I(-128i32 as u32)));
    insts.push(i_Push64(RMI_I(-129i32 as u32)));
    insts.push(i_Push64(RMI_I(-0x75c4e8a1i32 as u32)));
    //
    // JmpKnown skipped for now
    //
    // JmpUnknown
    insts.push(i_JmpUnknown(RM_R(rbp)));
    insts.push(i_JmpUnknown(RM_R(r11)));
    insts.push(i_JmpUnknown(RM_M(Addr_IRRS(321, rsi, rcx, 3))));
    insts.push(i_JmpUnknown(RM_M(Addr_IRRS(321, r10, rdx, 2))));
    //
    // JmpCond skipped for now
    //
    // CallKnown skipped for now
    //
    // CallUnknown
    insts.push(i_CallUnknown(RM_R(rbp)));
    insts.push(i_CallUnknown(RM_R(r11)));
    insts.push(i_CallUnknown(RM_M(Addr_IRRS(321, rsi, rcx, 3))));
    insts.push(i_CallUnknown(RM_M(Addr_IRRS(321, r10, rdx, 2))));
    //
    // Ret
    insts.push(i_Ret());

    /*
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    insts.push(i_());
    */
    let rru = create_reg_universe();
    for i in insts {
        println!("     {}", i.show_rru(Some(&rru)));
    }
    println!("QQQQ END I am a test");
}
