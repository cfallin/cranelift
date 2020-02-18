//! This module defines x86_64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{Addend, CodeOffset, CodeSink, ConstantPoolSink, NullConstantPoolSink, Reloc};
//zz use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{ConstantOffset, ExternalName, Function, JumpTable, SourceLoc, TrapCode};
use crate::ir::{FuncRef, GlobalValue, Type, Value};
use crate::isa::TargetIsa;
use crate::machinst::*;

use regalloc::InstRegUses;
use regalloc::Map as RegallocMap;
use regalloc::Set;
use regalloc::{
    RealReg, RealRegUniverse, Reg, RegClass, RegClassInfo, SpillSlot, VirtualReg, Writable,
    NUM_REG_CLASSES,
};

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
pub const ENC_RBX: u8 = 3;
pub const ENC_RSP: u8 = 4;
pub const ENC_RBP: u8 = 5;
pub const ENC_R12: u8 = 12;
pub const ENC_R13: u8 = 13;
pub const ENC_R14: u8 = 14;
pub const ENC_R15: u8 = 15;

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
        Reg::new_real(RegClass::I64, ENC_R14, /*index=*/ 2).to_real_reg(),
        "%r14".to_string(),
    )
}
fn info_R15() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_R15, /*index=*/ 3).to_real_reg(),
        "%r15".to_string(),
    )
}
fn info_RBX() -> (RealReg, String) {
    (
        Reg::new_real(RegClass::I64, ENC_RBX, /*index=*/ 4).to_real_reg(),
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
// these into a 32-bit constant.  That will require sprinkling a bunch of
// inline-always pragmas around the place.
pub fn reg_RAX() -> Reg {
    info_RAX().0.to_reg()
}
pub fn reg_RCX() -> Reg {
    info_RCX().0.to_reg()
}
pub fn reg_RDX() -> Reg {
    info_RDX().0.to_reg()
}
pub fn reg_RDI() -> Reg {
    info_RDI().0.to_reg()
}
pub fn reg_RSI() -> Reg {
    info_RSI().0.to_reg()
}
pub fn reg_R8() -> Reg {
    info_R8().0.to_reg()
}
pub fn reg_R9() -> Reg {
    info_R9().0.to_reg()
}

pub fn reg_RSP() -> Reg {
    info_RSP().0.to_reg()
}
pub fn reg_RBP() -> Reg {
    info_RBP().0.to_reg()
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
    allocable_by_class[RegClass::I64.rc_to_usize()] = Some(RegClassInfo {
        first: base,
        last: regs.len() - 1,
        suggested_scratch: Some(info_R12().0.get_index()),
    });

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
    allocable_by_class[RegClass::V128.rc_to_usize()] = Some(RegClassInfo {
        first: base,
        last: regs.len() - 1,
        suggested_scratch: Some(info_XMM15().0.get_index()),
    });

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
// Instruction operand sub-components (aka "parts"): definitions and printing

// Don't build these directly.  Instead use the ip_* functions to create them.
// "ip_" stands for "instruction part".

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
pub fn ip_Addr_IR(simm32: u32, base: Reg) -> Addr {
    debug_assert!(base.get_class() == RegClass::I64);
    Addr::IR { simm32, base }
}
pub fn ip_Addr_IRRS(simm32: u32, base: Reg, index: Reg, shift: u8) -> Addr {
    debug_assert!(base.get_class() == RegClass::I64);
    debug_assert!(index.get_class() == RegClass::I64);
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
pub fn ip_RMI_R(reg: Reg) -> RMI {
    debug_assert!(reg.get_class() == RegClass::I64);
    RMI::R { reg }
}
pub fn ip_RMI_M(addr: Addr) -> RMI {
    RMI::M { addr }
}
pub fn ip_RMI_I(simm32: u32) -> RMI {
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
pub fn ip_RM_R(reg: Reg) -> RM {
    debug_assert!(reg.get_class() == RegClass::I64);
    RM::R { reg }
}
pub fn ip_RM_M(addr: Addr) -> RM {
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

#[derive(Clone, PartialEq)]
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
    RightZ,
    RightS,
}
impl ShiftKind {
    fn to_string(&self) -> String {
        match self {
            ShiftKind::Left => "shl".to_string(),
            ShiftKind::RightZ => "shr".to_string(),
            ShiftKind::RightS => "sar".to_string(),
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
#[derive(Copy, Clone)]
pub enum CC {
    O,   //  0   overflow
    NO,  //  1   no overflow
    B,   //  2   <u
    NB,  //  3   >=u
    Z,   //  4   zero
    NZ,  //  5   not zero
    BE,  //  6   <=u
    NBE, //  7   >u
    S,   //  8   negative
    NS,  //  9   not negative
    L,   //  12  <s
    NL,  //  13  >=s
    LE,  //  14  <=s
    NLE, //  15  >s
}
impl CC {
    fn to_string(&self) -> String {
        match self {
            CC::O => "o".to_string(),
            CC::NO => "no".to_string(),
            CC::B => "b".to_string(),
            CC::NB => "nb".to_string(),
            CC::Z => "z".to_string(),
            CC::NZ => "nz".to_string(),
            CC::BE => "be".to_string(),
            CC::NBE => "nbe".to_string(),
            CC::S => "s".to_string(),
            CC::NS => "ns".to_string(),
            CC::L => "l".to_string(),
            CC::NL => "nl".to_string(),
            CC::LE => "le".to_string(),
            CC::NLE => "nle".to_string(),
        }
    }
    fn invert(&self) -> CC {
        match self {
            CC::O => CC::NO,
            CC::NO => CC::O,
            CC::B => CC::NB,
            CC::NB => CC::B,
            CC::Z => CC::NZ,
            CC::NZ => CC::Z,
            CC::BE => CC::NBE,
            CC::NBE => CC::BE,
            CC::S => CC::NS,
            CC::NS => CC::S,
            CC::L => CC::NL,
            CC::NL => CC::L,
            CC::LE => CC::NLE,
            CC::NLE => CC::LE,
        }
    }
    fn get_enc(&self) -> u8 {
        match self {
            CC::O => 0,
            CC::NO => 1,
            CC::B => 2,
            CC::NB => 3,
            CC::Z => 4,
            CC::NZ => 5,
            CC::BE => 6,
            CC::NBE => 7,
            CC::S => 8,
            CC::NS => 9,
            CC::L => 12,
            CC::NL => 13,
            CC::LE => 14,
            CC::NLE => 15,
        }
    }
}
impl fmt::Debug for CC {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_string())
    }
}

//=============================================================================
// Instruction sub-components: branch targets

/// A branch target. Either unresolved (basic-block index) or resolved (offset
/// from end of current instruction).
#[derive(Clone, Copy)]
pub enum BranchTarget {
    /// An unresolved reference to a BlockIndex, as passed into
    /// `lower_branch_group()`.
    Block(BlockIndex),
    /// A resolved reference to another instruction, after
    /// `Inst::with_block_offsets()`.  This offset is in bytes.
    ResolvedOffset(BlockIndex, isize),
}
impl ShowWithRRU for BranchTarget {
    // The RRU is totally irrelevant here :-/
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            BranchTarget::Block(bix) => format!("(Block {})", bix),
            BranchTarget::ResolvedOffset(bix, offs) => format!("(Block {}, offset {})", bix, offs),
        }
    }
}

impl BranchTarget {
    /// Lower the branch target given offsets of each block.
    pub fn lower(&mut self, targets: &[CodeOffset], my_offset: CodeOffset) {
        match self {
            &mut BranchTarget::Block(bix) => {
                let bix = bix as usize;
                assert!(bix < targets.len());
                let block_offset_in_func = targets[bix];
                let branch_offset = (block_offset_in_func as isize) - (my_offset as isize);
                *self = BranchTarget::ResolvedOffset(bix as BlockIndex, branch_offset);
            }
            &mut BranchTarget::ResolvedOffset(..) => {}
        }
    }

    /// Get the block index.
    pub fn as_block_index(&self) -> Option<BlockIndex> {
        match self {
            &BranchTarget::Block(bix) => Some(bix),
            _ => None,
        }
    }

    /// Get the offset as a signed 32 bit byte offset.  This returns the
    /// offset in bytes between the first byte of the source and the first
    /// byte of the target.  It does not take into account the Intel-specific
    /// rule that a branch offset is encoded as relative to the start of the
    /// following instruction.  That is a problem for the emitter to deal
    /// with.
    pub fn as_offset_i32(&self) -> Option<i32> {
        match self {
            &BranchTarget::ResolvedOffset(_, off) => {
                // Leave a bit of slack so that the emitter is guaranteed to
                // be able to add the length of the jump instruction encoding
                // to this value and still have a value in signed-32 range.
                if off >= -0x7FFF_FF00isize && off <= 0x7FFF_FF00isize {
                    Some(off as i32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Map the block index given a transform map.
    pub fn map(&mut self, block_index_map: &[BlockIndex]) {
        match self {
            &mut BranchTarget::Block(ref mut bix) => {
                let n = block_index_map[*bix as usize];
                *bix = n;
            }
            _ => panic!("BranchTarget::map() called on already-lowered BranchTarget!"),
        }
    }
}

//=============================================================================
// Instructions (top level): definition

// Don't build these directly.  Instead use the i_* functions to create them.
// "i_" stands for "instruction".

/// Instructions.  Destinations are on the RIGHT (a la AT&T syntax).
#[derive(Clone)]
pub enum Inst {
    /// nops of various sizes, including zero
    Nop { len: u8 },

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
    Mov_R_R { is64: bool, src: Reg, dst: Reg },

    /// movz (bl bq wl wq lq) addr reg (good for all ZX loads except 64->64).
    /// Note that the lq variant doesn't really exist since the default
    /// zero-extend rule makes it unnecessary.  For that case we emit the
    /// equivalent "movl AM, reg32".
    MovZX_M_R {
        extMode: ExtMode,
        addr: Addr,
        dst: Reg,
    },

    /// A plain 64-bit integer load, since MovZX_M_R can't represent that
    Mov64_M_R { addr: Addr, dst: Reg },

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
    Push64 { src: RMI },

    /// popq reg
    Pop64 { dst: Reg },

    /// call simm32
    CallKnown {
        dest: ExternalName,
        uses: Set<Reg>,
        defs: Set<Writable<Reg>>,
    },

    /// callq (reg mem)
    CallUnknown {
        dest: RM,
        //uses: Set<Reg>,
        //defs: Set<Writable<Reg>>,
    },

    // ---- branches (exactly one must appear at end of BB) ----
    /// ret
    Ret {},

    /// jmp simm32
    JmpKnown { dest: BranchTarget },

    /// jcond cond target target
    // Symmetrical two-way conditional branch.
    // Should never reach the emitter.
    JmpCondSymm {
        cc: CC,
        taken: BranchTarget,
        not_taken: BranchTarget,
    },

    /// Lowered conditional branch: contains the original instruction, and a
    /// flag indicating whether to invert the taken-condition or not. Only one
    /// BranchTarget is retained, and the other is implicitly the next
    /// instruction, given the final basic-block layout.
    JmpCond {
        cc: CC,
        //inverted: bool, is this needed?
        target: BranchTarget,
    },

    /// As for `CondBrLowered`, but represents a condbr/uncond-br sequence (two
    /// actual machine instructions). Needed when the final block layout implies
    /// that neither arm of a conditional branch targets the fallthrough block.
    // Should never reach the emitter
    JmpCondCompound {
        cc: CC,
        taken: BranchTarget,
        not_taken: BranchTarget,
    },

    /// jmpq (reg mem)
    JmpUnknown { target: RM },
}

// Handy constructors for Insts.

// For various sizes, will some number of lowest bits sign extend to be the
// same as the whole value?
fn low8willSXto64(x: u32) -> bool {
    let xs = (x as i32) as i64;
    xs == ((xs << 56) >> 56)
}
pub fn low32willSXto64(x: u64) -> bool {
    let xs = x as i64;
    xs == ((xs << 32) >> 32)
}
//fn low16willSXto64(x: u32) -> bool {
//    let xs = (x as i32) as i64;
//    xs == ((xs << 48) >> 48)
//}
fn low8willSXto32(x: u32) -> bool {
    let xs = x as i32;
    xs == ((xs << 24) >> 24)
}

pub fn i_Nop(len: u8) -> Inst {
    debug_assert!(len <= 16);
    Inst::Nop { len }
}

pub fn i_Alu_RMI_R(is64: bool, op: RMI_R_Op, src: RMI, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::Alu_RMI_R { is64, op, src, dst }
}

pub fn i_Imm_R(dstIs64: bool, simm64: u64, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(dst.get_class() == RegClass::I64);
    if !dstIs64 {
        debug_assert!(low32willSXto64(simm64));
    }
    Inst::Imm_R {
        dstIs64,
        simm64,
        dst,
    }
}

pub fn i_Mov_R_R(is64: bool, src: Reg, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(src.get_class() == RegClass::I64);
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::Mov_R_R { is64, src, dst }
}

pub fn i_MovZX_M_R(extMode: ExtMode, addr: Addr, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::MovZX_M_R { extMode, addr, dst }
}

pub fn i_Mov64_M_R(addr: Addr, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::Mov64_M_R { addr, dst }
}

pub fn i_MovSX_M_R(extMode: ExtMode, addr: Addr, wdst: Writable<Reg>) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::MovSX_M_R { extMode, addr, dst }
}

pub fn i_Mov_R_M(
    size: u8, // 1, 2, 4 or 8
    src: Reg,
    addr: Addr,
) -> Inst {
    debug_assert!(size == 8 || size == 4 || size == 2 || size == 1);
    debug_assert!(src.get_class() == RegClass::I64);
    Inst::Mov_R_M { size, src, addr }
}

pub fn i_Shift_R(
    is64: bool,
    kind: ShiftKind,
    nBits: u8, // 1 .. #bits-in-type - 1, or 0 to mean "%cl"
    wdst: Writable<Reg>,
) -> Inst {
    let dst = wdst.to_reg();
    debug_assert!(nBits < if is64 { 64 } else { 32 });
    debug_assert!(dst.get_class() == RegClass::I64);
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
    debug_assert!(dst.get_class() == RegClass::I64);
    Inst::Cmp_RMI_R { size, src, dst }
}

pub fn i_Push64(src: RMI) -> Inst {
    Inst::Push64 { src }
}

pub fn i_Pop64(wdst: Writable<Reg>) -> Inst {
    Inst::Pop64 { dst: wdst.to_reg() }
}

//pub fn i_CallKnown(target: FuncRef) -> Inst {
//    Inst::CallKnown { target }
//}

pub fn i_CallUnknown(dest: RM) -> Inst {
    Inst::CallUnknown { dest }
}

pub fn i_Ret() -> Inst {
    Inst::Ret {}
}

pub fn i_JmpKnown(dest: BranchTarget) -> Inst {
    Inst::JmpKnown { dest }
}

pub fn i_JmpCondSymm(cc: CC, taken: BranchTarget, not_taken: BranchTarget) -> Inst {
    Inst::JmpCondSymm {
        cc,
        taken,
        not_taken,
    }
}

pub fn i_JmpCond(cc: CC, target: BranchTarget) -> Inst {
    Inst::JmpCond { cc, target }
}

pub fn i_JmpCondCompound(cc: CC, taken: BranchTarget, not_taken: BranchTarget) -> Inst {
    Inst::JmpCondCompound {
        cc,
        taken,
        not_taken,
    }
}

pub fn i_JmpUnknown(target: RM) -> Inst {
    Inst::JmpUnknown { target }
}

//=============================================================================
// Instructions: printing

fn x64_show_rru(inst: &Inst, mb_rru: Option<&RealRegUniverse>) -> String {
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

    match inst {
        Inst::Nop { len } => format!("{} len={}", ljustify("nop".to_string()), len),
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
            if *dstIs64 {
                format!(
                    "{} ${}, {}",
                    ljustify("movabsq".to_string()),
                    *simm64 as i64,
                    show_ireg_sized(*dst, mb_rru, 8)
                )
            } else {
                format!(
                    "{} ${}, {}",
                    ljustify("movl".to_string()),
                    (*simm64 as u32) as i32,
                    show_ireg_sized(*dst, mb_rru, 4)
                )
            }
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
        Inst::Pop64 { dst } => format!("{} {}", ljustify("popq".to_string()), dst.show_rru(mb_rru)),
        //
        //Inst::CallKnown { target } => format!("{} {:?}", ljustify("call".to_string()), target),
        Inst::CallKnown { .. } => "**CallKnown**".to_string(),
        //
        Inst::CallUnknown { dest } => format!(
            "{} *{}",
            ljustify("call".to_string()),
            dest.show_rru(mb_rru)
        ),
        Inst::Ret {} => "ret".to_string(),
        Inst::JmpKnown { dest } => {
            format!("{} {}", ljustify("jmp".to_string()), dest.show_rru(mb_rru))
        }
        Inst::JmpCondSymm {
            cc,
            taken,
            not_taken,
        } => format!(
            "{} taken={} not_taken={}",
            ljustify2("j".to_string(), cc.to_string()),
            taken.show_rru(mb_rru),
            not_taken.show_rru(mb_rru)
        ),
        //
        Inst::JmpCond { .. } => "**JmpCond**".to_string(),
        //
        Inst::JmpCondCompound { .. } => "**JmpCondCompound**".to_string(),
        Inst::JmpUnknown { target } => format!(
            "{} *{}",
            ljustify("jmp".to_string()),
            target.show_rru(mb_rru)
        ),
    }
}

impl ShowWithRRU for Inst {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        x64_show_rru(self, mb_rru)
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
        // ** Nop
        Inst::Alu_RMI_R {
            is64: _,
            op: _,
            src,
            dst,
        } => {
            src.get_regs(&mut iru.used);
            iru.modified.insert(Writable::from_reg(*dst));
        }
        Inst::Imm_R {
            dstIs64: _,
            simm64: _,
            dst,
        } => {
            iru.defined.insert(Writable::from_reg(*dst));
        }
        Inst::Mov_R_R { is64: _, src, dst } => {
            iru.used.insert(*src);
            iru.defined.insert(Writable::from_reg(*dst));
        }
        Inst::MovZX_M_R {
            extMode: _,
            addr,
            dst,
        } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(Writable::from_reg(*dst));
        }
        Inst::Mov64_M_R { addr, dst } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(Writable::from_reg(*dst));
        }
        Inst::MovSX_M_R {
            extMode: _,
            addr,
            dst,
        } => {
            addr.get_regs(&mut iru.used);
            iru.defined.insert(Writable::from_reg(*dst));
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
            iru.modified.insert(Writable::from_reg(*dst));
        }
        Inst::Cmp_RMI_R { size: _, src, dst } => {
            src.get_regs(&mut iru.used);
            iru.used.insert(*dst); // yes, really |iru.used|
        }
        Inst::Push64 { src } => {
            src.get_regs(&mut iru.used);
            iru.modified.insert(Writable::from_reg(reg_RSP()));
        }
        Inst::Pop64 { dst } => {
            iru.defined.insert(Writable::from_reg(*dst));
        }
        Inst::CallKnown {
            dest: _,
            uses: _,
            defs: _,
        } => {
            // FIXME add arg regs (iru.used) and caller-saved regs (iru.defined)
            unimplemented!();
        }
        Inst::CallUnknown { dest } => {
            dest.get_regs(&mut iru.used);
        }
        Inst::Ret {} => {}
        Inst::JmpKnown { dest: _ } => {}
        Inst::JmpCondSymm {
            cc: _,
            taken: _,
            not_taken: _,
        } => {}
        //
        // ** JmpCond
        //
        // ** JmpCondCompound
        //
        //Inst::JmpUnknown { target } => {
        //    target.get_regs(&mut iru.used);
        //}
        other => panic!("x64_get_regs: {}", other.show_rru(None)),
    }

    // Enforce invariants described above.
    iru.defined.remove(&iru.modified);
    iru.used.remove(&Set::from_vec(
        iru.modified.iter().map(|r| r.to_reg()).collect(),
    ));

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
        // ** Nop
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
        Inst::Pop64 { ref mut dst } => {
            apply_map(dst, post_map);
        }
        Inst::CallKnown {
            dest: _,
            uses: _,
            defs: _,
        } => {}
        Inst::CallUnknown { dest } => {
            dest.apply_map(pre_map);
        }
        Inst::Ret {} => {}
        Inst::JmpKnown { dest: _ } => {}
        Inst::JmpCondSymm {
            cc: _,
            taken: _,
            not_taken: _,
        } => {}
        //
        // ** JmpCond
        //
        // ** JmpCondCompound
        //
        //Inst::JmpUnknown { target } => {
        //    target.apply_map(pre_map);
        //}
        other => panic!("x64_map_regs: {}", other.show_rru(None)),
    }
}

//=============================================================================
// Instructions and subcomponents: emission

// For all of the routines that take both a memory-or-reg operand (sometimes
// called "E" in the Intel documentation) and a reg-only operand ("G" in
// Intelese), the order is always G first, then E.
//
// "enc" in the following means "hardware register encoding number".

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

// F_*: these flags describe special handling of the insn to be generated.  Be
// careful with these.  It is easy to create nonsensical combinations.
const F_NONE: u32 = 0;

// Emit the REX prefix byte even if it appears to be redundant (== 0x40).
const F_RETAIN_REDUNDANT_REX: u32 = 1;

// Set the W bit in the REX prefix to zero.  By default it will be set to 1,
// indicating a 64-bit operation.
const F_CLEAR_REX_W: u32 = 2;

// Add an 0x66 (operand-size override) prefix.  This is necessary to indicate
// a 16-bit operation.  Normally this will be used together with F_CLEAR_REX_W.
const F_PREFIX_66: u32 = 4;

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
fn emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE<CS: CodeSink>(
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
    let prefix66 = (flags & F_PREFIX_66) != 0;
    let clearRexW = (flags & F_CLEAR_REX_W) != 0;
    let retainRedundant = (flags & F_RETAIN_REDUNDANT_REX) != 0;
    // The operand-size override, if requested.  This indicates a 16-bit
    // operation.
    if prefix66 {
        sink.put1(0x66);
    }
    match memE {
        Addr::IR { simm32, base: regE } => {
            // First, cook up the REX byte.  This is easy.
            let encE = iregEnc(*regE);
            let w = if clearRexW { 0 } else { 1 };
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
                numOpcodes -= 1;
                sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
            }
            // Now the mod/rm and associated immediates.  This is
            // significantly complicated due to the multiple special cases.
            if *simm32 == 0
                && encE != ENC_RSP
                && encE != ENC_RBP
                && encE != ENC_R12
                && encE != ENC_R13
            {
                // FIXME JRS 2020Feb11: those four tests can surely be
                // replaced by a single mask-and-compare check.  We should do
                // that because this routine is likely to be hot.
                sink.put1(mkModRegRM(0, encG & 7, encE & 7));
            } else if *simm32 == 0 && (encE == ENC_RSP || encE == ENC_R12) {
                sink.put1(mkModRegRM(0, encG & 7, 4));
                sink.put1(0x24);
            } else if low8willSXto32(*simm32) && encE != ENC_RSP && encE != ENC_R12 {
                sink.put1(mkModRegRM(1, encG & 7, encE & 7));
                sink.put1((simm32 & 0xFF) as u8);
            } else if encE != ENC_RSP && encE != ENC_R12 {
                sink.put1(mkModRegRM(2, encG & 7, encE & 7));
                sink.put4(*simm32);
            } else if (encE == ENC_RSP || encE == ENC_R12) && low8willSXto32(*simm32) {
                // REX.B distinguishes RSP from R12
                sink.put1(mkModRegRM(1, encG & 7, 4));
                sink.put1(0x24);
                sink.put1((simm32 & 0xFF) as u8);
            } else if encE == ENC_R12 || encE == ENC_RSP {
                //.. wait for test case for RSP case
                // REX.B distinguishes RSP from R12
                sink.put1(mkModRegRM(2, encG & 7, 4));
                sink.put1(0x24);
                sink.put4(*simm32);
            } else {
                panic!("emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE: IR");
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
            let w = if clearRexW { 0 } else { 1 };
            let r = (encG >> 3) & 1;
            let x = (encIndex >> 3) & 1;
            let b = (encBase >> 3) & 1;
            let rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
            if rex != 0x40 || retainRedundant {
                sink.put1(rex);
            }
            // All other prefixes and opcodes
            while numOpcodes > 0 {
                numOpcodes -= 1;
                sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
            }
            // modrm, SIB, immediates
            if low8willSXto32(*simm32) && encIndex != ENC_RSP {
                sink.put1(mkModRegRM(1, encG & 7, 4));
                sink.put1(mkSIB(*shift, encIndex & 7, encBase & 7));
                sink.put1(*simm32 as u8);
            } else if encIndex != ENC_RSP {
                sink.put1(mkModRegRM(2, encG & 7, 4));
                sink.put1(mkSIB(*shift, encIndex & 7, encBase & 7));
                sink.put4(*simm32);
            } else {
                panic!("emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE: IRRS");
            }
        }
    }
}

// This is the core 'emit' function for instructions that do not reference
// memory.
//
// This is conceptually the same as
// emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE, except it is for the case
// where the E operand is a register rather than memory.  Hence it is much
// simpler.
fn emit_REX_OPCODES_MODRM_encG_encE<CS: CodeSink>(
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
    let prefix66 = (flags & F_PREFIX_66) != 0;
    let clearRexW = (flags & F_CLEAR_REX_W) != 0;
    let retainRedundant = (flags & F_RETAIN_REDUNDANT_REX) != 0;
    // The operand-size override
    if prefix66 {
        sink.put1(0x66);
    }
    // The rex byte
    let w = if clearRexW { 0 } else { 1 };
    let r = (encG >> 3) & 1;
    let x = 0;
    let b = (encE >> 3) & 1;
    let rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | b;
    if rex != 0x40 || retainRedundant {
        sink.put1(rex);
    }
    // All other prefixes and opcodes
    while numOpcodes > 0 {
        numOpcodes -= 1;
        sink.put1(((opcodes >> (numOpcodes << 3)) & 0xFF) as u8);
    }
    // Now the mod/rm byte.  The instruction we're generating doesn't access
    // memory, so there is no SIB byte or immediate -- we're done.
    sink.put1(mkModRegRM(3, encG & 7, encE & 7));
}

// These are merely wrappers for the above two functions that facilitate passing
// actual |Reg|s rather than their encodings.
fn emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE<CS: CodeSink>(
    sink: &mut CS,
    opcodes: u32,
    numOpcodes: usize,
    regG: Reg,
    memE: &Addr,
    flags: u32,
) {
    // JRS FIXME 2020Feb07: this should really just be |regEnc| not |iregEnc|
    let encG = iregEnc(regG);
    emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE(sink, opcodes, numOpcodes, encG, memE, flags);
}

fn emit_REX_OPCODES_MODRM_regG_regE<CS: CodeSink>(
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
    emit_REX_OPCODES_MODRM_encG_encE(sink, opcodes, numOpcodes, encG, encE, flags);
}

// Write a suitable number of bits from an imm64 to the sink.
fn emit_simm<CS: CodeSink>(sink: &mut CS, size: u8, simm32: u32) {
    match size {
        8 | 4 => sink.put4(simm32),
        2 => sink.put2(simm32 as u16),
        1 => sink.put1(simm32 as u8),
        _ => panic!("x64::Inst::emit_simm: unreachable"),
    }
}

// The top-level emit function.
//
// Important!  Do not add improved (shortened) encoding cases to existing
// instructions without also adding tests for those improved encodings.  That
// is a dangerous game that leads to hard-to-track-down errors in the emitted
// code.
//
// For all instructions, make sure to have test coverage for all of the
// following situations.  Do this by creating the cross product resulting from
// applying the following rules to each operand:
//
// (1) for any insn that mentions a register: one test using a register from
//     the group [rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi] and a second one
//     using a register from the group [r8, r9, r10, r11, r12, r13, r14, r15].
//     This helps detect incorrect REX prefix construction.
//
// (2) for any insn that mentions a byte register: one test for each of the
//     four encoding groups [al, cl, dl, bl], [spl, bpl, sil, dil],
//     [r8b .. r11b] and [r12b .. r15b].  This checks that
//     apparently-redundant REX prefixes are retained when required.
//
// (3) for any insn that contains an immediate field, check the following
//     cases: field is zero, field is in simm8 range (-128 .. 127), field is
//     in simm32 range (-0x8000_0000 .. 0x7FFF_FFFF).  This is because some
//     instructions that require a 32-bit immediate have a short-form encoding
//     when the imm is in simm8 range.
//
// Rules (1), (2) and (3) don't apply for registers within address expressions
// (|Addr|s).  Those are already pretty well tested, and the registers in them
// don't have any effect on the containing instruction (apart from possibly
// require REX prefix bits).
//
// When choosing registers for a test, avoid using registers with the same
// offset within a given group.  For example, don't use rax and r8, since they
// both have the lowest 3 bits as 000, and so the test won't detect errors
// where those 3-bit register sub-fields are confused by the emitter.  Instead
// use (eg) rax (lo3 = 000) and r9 (lo3 = 001).  Similarly, don't use (eg) cl
// and bpl since they have the same offset in their group; use instead (eg) cl
// and sil.
//
// For all instructions, also add a test that uses only low-half registers
// (rax .. rdi, xmm0 .. xmm7) etc, so as to check that any redundant REX
// prefixes are correctly omitted.  This low-half restriction must apply to
// _all_ registers in the insn, even those in address expressions.
//
// Following these rules creates large numbers of test cases, but it's the
// only way to make the emitter reliable.
//
// Known possible improvements:
//
// * there's a shorter encoding for shl/shr/sar by a 1-bit immediate.  (Do we
//   care?)

fn x64_emit<CS: CodeSink>(inst: &Inst, sink: &mut CS) {
    match inst {
        // ** Nop
        Inst::Alu_RMI_R {
            is64,
            op,
            src: srcE,
            dst: regG,
        } => {
            let flags = if *is64 { F_NONE } else { F_CLEAR_REX_W };
            if *op == RMI_R_Op::Mul {
                // We kinda freeloaded Mul into RMI_R_Op, but it doesn't fit
                // the usual pattern, so we have to special-case it.
                match srcE {
                    RMI::R { reg: regE } => {
                        emit_REX_OPCODES_MODRM_regG_regE(sink, 0x0FAF, 2, *regG, *regE, flags);
                    }
                    RMI::M { addr } => {
                        emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                            sink, 0x0FAF, 2, *regG, addr, flags,
                        );
                    }
                    RMI::I { simm32 } => {
                        let useImm8 = low8willSXto32(*simm32);
                        let opcode = if useImm8 { 0x6B } else { 0x69 };
                        // Yes, really, regG twice.
                        emit_REX_OPCODES_MODRM_regG_regE(sink, opcode, 1, *regG, *regG, flags);
                        emit_simm(sink, if useImm8 { 1 } else { 4 }, *simm32);
                    }
                }
            } else {
                let (opcode_R, opcode_M, subopcode_I) = match op {
                    RMI_R_Op::Add => (0x01, 0x03, 0),
                    RMI_R_Op::Sub => (0x29, 0x2B, 5),
                    RMI_R_Op::And => (0x21, 0x23, 4),
                    RMI_R_Op::Or => (0x09, 0x0B, 1),
                    RMI_R_Op::Xor => (0x31, 0x33, 6),
                    RMI_R_Op::Mul => panic!("unreachable"),
                };
                match srcE {
                    RMI::R { reg: regE } => {
                        // Note.  The arguments .. regE .. regG .. sequence
                        // here is the opposite of what is expected.  I'm not
                        // sure why this is.  But I am fairly sure that the
                        // arg order could be switched back to the expected
                        // .. regG .. regE .. if opcode_rr is also switched
                        // over to the "other" basic integer opcode (viz, the
                        // R/RM vs RM/R duality).  However, that would mean
                        // that the test results won't be in accordance with
                        // the GNU as reference output.  In other words, the
                        // inversion exists as a result of using GNU as as a
                        // gold standard.
                        emit_REX_OPCODES_MODRM_regG_regE(sink, opcode_R, 1, *regE, *regG, flags);
                        // NB: if this is ever extended to handle byte size
                        // ops, be sure to retain redundant REX prefixes.
                    }
                    RMI::M { addr } => {
                        // Whereas here we revert to the "normal" G-E ordering.
                        emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                            sink, opcode_M, 1, *regG, addr, flags,
                        );
                    }
                    RMI::I { simm32 } => {
                        let useImm8 = low8willSXto32(*simm32);
                        let opcode = if useImm8 { 0x83 } else { 0x81 };
                        // And also here we use the "normal" G-E ordering.
                        let encG = iregEnc(*regG);
                        emit_REX_OPCODES_MODRM_encG_encE(sink, opcode, 1, subopcode_I, encG, flags);
                        emit_simm(sink, if useImm8 { 1 } else { 4 }, *simm32);
                    }
                }
            }
        }
        Inst::Imm_R {
            dstIs64,
            simm64,
            dst,
        } => {
            let encDst = iregEnc(*dst);
            if *dstIs64 {
                // FIXME JRS 2020Feb10: also use the 32-bit case here when
                // possible
                sink.put1(0x48 | ((encDst >> 3) & 1));
                sink.put1(0xB8 | (encDst & 7));
                sink.put8(*simm64);
            } else {
                if ((encDst >> 3) & 1) == 1 {
                    sink.put1(0x41);
                }
                sink.put1(0xB8 | (encDst & 7));
                sink.put4(*simm64 as u32);
            }
        }
        Inst::Mov_R_R { is64, src, dst } => {
            let flags = if *is64 { F_NONE } else { F_CLEAR_REX_W };
            emit_REX_OPCODES_MODRM_regG_regE(sink, 0x89, 1, *src, *dst, flags);
        }
        Inst::MovZX_M_R { extMode, addr, dst } => {
            match extMode {
                ExtMode::BL => {
                    // MOVZBL is (REX.W==0) 0F B6 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x0FB6,
                        2,
                        *dst,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
                ExtMode::BQ => {
                    // MOVZBQ is (REX.W==1) 0F B6 /r
                    // I'm not sure why the Intel manual offers different
                    // encodings for MOVZBQ than for MOVZBL.  AIUI they should
                    // achieve the same, since MOVZBL is just going to zero out
                    // the upper half of the destination anyway.
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x0FB6, 2, *dst, addr, F_NONE)
                }
                ExtMode::WL => {
                    // MOVZWL is (REX.W==0) 0F B7 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x0FB7,
                        2,
                        *dst,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
                ExtMode::WQ => {
                    // MOVZWQ is (REX.W==1) 0F B7 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x0FB7, 2, *dst, addr, F_NONE)
                }
                ExtMode::LQ => {
                    // This is just a standard 32 bit load, and we rely on the
                    // default zero-extension rule to perform the extension.
                    // MOV r/m32, r32 is (REX.W==0) 8B /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x8B,
                        1,
                        *dst,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
            }
        }
        Inst::Mov64_M_R { addr, dst } => {
            emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x8B, 1, *dst, addr, F_NONE)
        }
        Inst::MovSX_M_R { extMode, addr, dst } => {
            match extMode {
                ExtMode::BL => {
                    // MOVSBL is (REX.W==0) 0F BE /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x0FBE,
                        2,
                        *dst,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
                ExtMode::BQ => {
                    // MOVSBQ is (REX.W==1) 0F BE /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x0FBE, 2, *dst, addr, F_NONE)
                }
                ExtMode::WL => {
                    // MOVSWL is (REX.W==0) 0F BF /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x0FBF,
                        2,
                        *dst,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
                ExtMode::WQ => {
                    // MOVSWQ is (REX.W==1) 0F BF /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x0FBF, 2, *dst, addr, F_NONE)
                }
                ExtMode::LQ => {
                    // MOVSLQ is (REX.W==1) 63 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x63, 1, *dst, addr, F_NONE)
                }
            }
        }
        Inst::Mov_R_M { size, src, addr } => {
            match size {
                1 => {
                    // This is one of the few places where the presence of a
                    // redundant REX prefix changes the meaning of the
                    // instruction.
                    let encSrc = iregEnc(*src);
                    let retainRedundantRex = if encSrc >= 4 && encSrc <= 7 {
                        F_RETAIN_REDUNDANT_REX
                    } else {
                        0
                    };
                    // MOV r8, r/m8 is (REX.W==0) 88 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x88,
                        1,
                        *src,
                        addr,
                        F_CLEAR_REX_W | retainRedundantRex,
                    )
                }
                2 => {
                    // MOV r16, r/m16 is 66 (REX.W==0) 89 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x89,
                        1,
                        *src,
                        addr,
                        F_CLEAR_REX_W | F_PREFIX_66,
                    )
                }
                4 => {
                    // MOV r32, r/m32 is (REX.W==0) 89 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(
                        sink,
                        0x89,
                        1,
                        *src,
                        addr,
                        F_CLEAR_REX_W,
                    )
                }
                8 => {
                    // MOV r64, r/m64 is (REX.W==1) 89 /r
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, 0x89, 1, *src, addr, F_NONE)
                }
                _ => panic!("x64::Inst::Mov_R_M::emit: unreachable"),
            }
        }
        Inst::Shift_R {
            is64,
            kind,
            nBits,
            dst,
        } => {
            let encDst = iregEnc(*dst);
            let subopcode = match kind {
                ShiftKind::Left => 4,
                ShiftKind::RightZ => 5,
                ShiftKind::RightS => 7,
            };
            if *nBits == 0 {
                // SHL/SHR/SAR %cl, reg32 is (REX.W==0) D3 /subopcode
                // SHL/SHR/SAR %cl, reg64 is (REX.W==1) D3 /subopcode
                emit_REX_OPCODES_MODRM_encG_encE(
                    sink,
                    0xD3,
                    1,
                    subopcode,
                    encDst,
                    if *is64 { F_NONE } else { F_CLEAR_REX_W },
                );
            } else {
                // SHL/SHR/SAR $ib, reg32 is (REX.W==0) C1 /subopcode ib
                // SHL/SHR/SAR $ib, reg64 is (REX.W==1) C1 /subopcode ib
                // When the shift amount is 1, there's an even shorter
                // encoding, but we don't bother with that nicety here.
                emit_REX_OPCODES_MODRM_encG_encE(
                    sink,
                    0xC1,
                    1,
                    subopcode,
                    encDst,
                    if *is64 { F_NONE } else { F_CLEAR_REX_W },
                );
                sink.put1(*nBits);
            }
        }
        Inst::Cmp_RMI_R {
            size,
            src: srcE,
            dst: regG,
        } => {
            let mut retainRedundantRex = 0;
            if *size == 1 {
                // Here, a redundant REX prefix changes the meaning of the
                // instruction.
                let encG = iregEnc(*regG);
                if encG >= 4 && encG <= 7 {
                    retainRedundantRex = F_RETAIN_REDUNDANT_REX;
                }
            }
            let mut flags = match size {
                8 => F_NONE,
                4 => F_CLEAR_REX_W,
                2 => F_CLEAR_REX_W | F_PREFIX_66,
                1 => F_CLEAR_REX_W | retainRedundantRex,
                _ => panic!("x64::Inst::Cmp_RMI_R::emit: unreachable"),
            };
            match srcE {
                RMI::R { reg: regE } => {
                    let opcode = if *size == 1 { 0x38 } else { 0x39 };
                    if *size == 1 {
                        // We also need to check whether the E register forces
                        // the use of a redundant REX.
                        let encE = iregEnc(*regE);
                        if encE >= 4 && encE <= 7 {
                            flags |= F_RETAIN_REDUNDANT_REX;
                        }
                    }
                    // Same comment re swapped args as for Alu_RMI_R.
                    emit_REX_OPCODES_MODRM_regG_regE(sink, opcode, 1, *regE, *regG, flags);
                }
                RMI::M { addr } => {
                    let opcode = if *size == 1 { 0x3A } else { 0x3B };
                    // Whereas here we revert to the "normal" G-E ordering.
                    emit_REX_OPCODES_MODRM_SIB_IMM_regG_memE(sink, opcode, 1, *regG, addr, flags);
                }
                RMI::I { simm32 } => {
                    // FIXME JRS 2020Feb11: there are shorter encodings for
                    // cmp $imm, rax/eax/ax/al.
                    let useImm8 = low8willSXto32(*simm32);
                    let opcode = if *size == 1 {
                        0x80
                    } else if useImm8 {
                        0x83
                    } else {
                        0x81
                    };
                    // And also here we use the "normal" G-E ordering.
                    let encG = iregEnc(*regG);
                    emit_REX_OPCODES_MODRM_encG_encE(
                        sink, opcode, 1, 7, /*subopcode*/
                        encG, flags,
                    );
                    emit_simm(sink, if useImm8 { 1 } else { *size }, *simm32);
                }
            }
        }
        Inst::Push64 { src } => {
            match src {
                RMI::R { reg } => {
                    let encReg = iregEnc(*reg);
                    let rex = 0x40 | ((encReg >> 3) & 1);
                    if rex != 0x40 {
                        sink.put1(rex);
                    }
                    sink.put1(0x50 | (encReg & 7));
                }
                RMI::M { addr } => {
                    emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE(
                        sink,
                        0xFF,
                        1,
                        6, /*subopcode*/
                        addr,
                        F_CLEAR_REX_W,
                    );
                }
                RMI::I { simm32 } => {
                    if low8willSXto64(*simm32) {
                        sink.put1(0x6A);
                        sink.put1(*simm32 as u8);
                    } else {
                        sink.put1(0x68);
                        sink.put4(*simm32);
                    }
                }
            }
        }
        Inst::Pop64 { dst } => {
            let encDst = iregEnc(*dst);
            if encDst >= 8 {
                // 0x41 == REX.{W=0, B=1}.  It seems that REX.W is irrelevant
                // here.
                sink.put1(0x41);
            }
            sink.put1(0x58 + (encDst & 7));
        }
        //
        // ** Inst::CallKnown
        //
        Inst::CallUnknown { dest } => {
            match dest {
                RM::R { reg } => {
                    let regEnc = iregEnc(*reg);
                    emit_REX_OPCODES_MODRM_encG_encE(
                        sink,
                        0xFF,
                        1,
                        2, /*subopcode*/
                        regEnc,
                        F_CLEAR_REX_W,
                    );
                }
                RM::M { addr } => {
                    emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE(
                        sink,
                        0xFF,
                        1,
                        2, /*subopcode*/
                        addr,
                        F_CLEAR_REX_W,
                    );
                }
            }
        }
        Inst::Ret {} => sink.put1(0xC3),
        //
        // ** Inst::JmpKnown
        //
        // ** Inst::JmpCondSymm   XXXX should never happen
        //
        Inst::JmpCond {
            cc,
            target: BranchTarget::ResolvedOffset(bix, offset),
        } if *offset >= -0x7FFF_FF00 && *offset <= 0x7FFF_FF00 => {
            // This insn is 6 bytes long.  Currently |offset| is relative to
            // the start of this insn, but the Intel encoding requires it to
            // be relative to the start of the next instruction.  Hence the
            // adjustment.
            let mut offs_i32 = *offset as i32;
            offs_i32 -= 6;
            let offs_u32 = offs_i32 as u32;
            sink.put1(0x0F);
            sink.put1(0x80 + cc.get_enc());
            sink.put4(offs_u32);
        }
        //
        // ** Inst::JmpCondCompound   XXXX should never happen
        //
        Inst::JmpUnknown { target } => {
            match target {
                RM::R { reg } => {
                    let regEnc = iregEnc(*reg);
                    emit_REX_OPCODES_MODRM_encG_encE(
                        sink,
                        0xFF,
                        1,
                        4, /*subopcode*/
                        regEnc,
                        F_CLEAR_REX_W,
                    );
                }
                RM::M { addr } => {
                    emit_REX_OPCODES_MODRM_SIB_IMM_encG_memE(
                        sink,
                        0xFF,
                        1,
                        4, /*subopcode*/
                        addr,
                        F_CLEAR_REX_W,
                    );
                }
            }
        }
        _ => panic!("x64_emit: unhandled: {}", inst.show_rru(None)),
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

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        // Note (carefully!) that a 32-bit mov *isn't* a no-op since it zeroes
        // out the upper 32 bits of the destination.  For example, we could
        // conceivably use |movl %reg, %reg| to zero out the top 32 bits of
        // %reg.
        match self {
            Inst::Mov_R_R { is64, src, dst } if *is64 => Some((Writable::from_reg(*dst), *src)),
            _ => None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            // Interesting cases.
            &Inst::Ret {} => MachTerminator::Ret,
            &Inst::JmpKnown { dest } => MachTerminator::Uncond(dest.as_block_index().unwrap()),
            &Inst::JmpCondSymm {
                cc: _,
                taken,
                not_taken,
            } => MachTerminator::Cond(
                taken.as_block_index().unwrap(),
                not_taken.as_block_index().unwrap(),
            ),
            &Inst::JmpCond { .. } | &Inst::JmpCondCompound { .. } => {
                panic!("is_term() called after lowering branches");
            }
            // All other cases are boring.
            _ => MachTerminator::None,
        }
    }

    fn gen_move(dst_reg: Writable<Reg>, src_reg: Reg) -> Inst {
        let rcD = dst_reg.to_reg().get_class();
        let rcS = src_reg.get_class();
        // If this isn't true, we have gone way off the rails
        assert!(rcD == rcS);
        match rcD {
            RegClass::I64 => i_Mov_R_R(true, src_reg, dst_reg),
            _ => panic!("gen_move(x64): unhandled regclass"),
        }
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

    fn gen_jump(blockindex: BlockIndex) -> Inst {
        i_JmpKnown(BranchTarget::Block(blockindex))
    }

    fn with_block_rewrites(&mut self, block_target_map: &[BlockIndex]) {
        // This is identical (modulo renaming) to the arm64 version.
        match self {
            &mut Inst::JmpKnown { ref mut dest } => {
                dest.map(block_target_map);
            }
            &mut Inst::JmpCondSymm {
                cc: _,
                ref mut taken,
                ref mut not_taken,
            } => {
                taken.map(block_target_map);
                not_taken.map(block_target_map);
            }
            &mut Inst::JmpCond { .. } | &mut Inst::JmpCondCompound { .. } => {
                panic!("with_block_rewrites called after branch lowering!");
            }
            _ => {}
        }
    }

    fn with_fallthrough_block(&mut self, fallthrough: Option<BlockIndex>) {
        // This is identical (modulo renaming) to the arm64 version.
        match self {
            &mut Inst::JmpCondSymm {
                cc,
                taken,
                not_taken,
            } => {
                if taken.as_block_index() == fallthrough {
                    *self = i_JmpCond(cc, not_taken);
                } else if not_taken.as_block_index() == fallthrough {
                    *self = i_JmpCond(cc, taken);
                } else {
                    // We need a compound sequence (condbr / uncond-br).
                    *self = i_JmpCondCompound(cc, taken, not_taken);
                }
            }
            &mut Inst::JmpKnown { dest } => {
                if dest.as_block_index() == fallthrough {
                    *self = i_Nop(0);
                }
            }
            _ => {}
        }
    }

    fn with_block_offsets(&mut self, my_offset: CodeOffset, targets: &[CodeOffset]) {
        // This is identical (modulo renaming) to the arm64 version.
        match self {
            &mut Inst::JmpCond {
                cc: _,
                ref mut target,
            } => {
                target.lower(targets, my_offset);
            }
            &mut Inst::JmpCondCompound {
                cc: _,
                ref mut taken,
                ref mut not_taken,
                ..
            } => {
                taken.lower(targets, my_offset);
                not_taken.lower(targets, my_offset);
            }
            &mut Inst::JmpKnown { ref mut dest } => {
                dest.lower(targets, my_offset);
            }
            _ => {}
        }
    }

    fn reg_universe() -> RealRegUniverse {
        create_reg_universe()
    }
}

impl<CS: CodeSink, CPS: ConstantPoolSink> MachInstEmit<CS, CPS> for Inst {
    fn emit(&self, sink: &mut CS, _consts: &mut CPS) {
        x64_emit(self, sink);
    }
}

//=============================================================================
// Tests for the emitter

// See comments at the top of |fn x64_emit| for advice on how to create
// reliable test cases.

// to see stdout: cargo test -- --nocapture
//
// for this specific case:
//
// (cd cranelift-codegen && \
// RUST_BACKTRACE=1 \
//       cargo test isa::x64::inst::test_x64_insn_encoding_and_printing \
//                  -- --nocapture)

#[cfg(test)]
use crate::isa::test_utils;

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

    // And Writable<> versions of the same:
    let w_rax = Writable::<Reg>::from_reg(info_RAX().0.to_reg());
    let w_rbx = Writable::<Reg>::from_reg(info_RBX().0.to_reg());
    let w_rcx = Writable::<Reg>::from_reg(info_RCX().0.to_reg());
    let w_rdx = Writable::<Reg>::from_reg(info_RDX().0.to_reg());
    let w_rsi = Writable::<Reg>::from_reg(info_RSI().0.to_reg());
    let w_rdi = Writable::<Reg>::from_reg(info_RDI().0.to_reg());
    let _w_rsp = Writable::<Reg>::from_reg(info_RSP().0.to_reg());
    let _w_rbp = Writable::<Reg>::from_reg(info_RBP().0.to_reg());
    let w_r8 = Writable::<Reg>::from_reg(info_R8().0.to_reg());
    let w_r9 = Writable::<Reg>::from_reg(info_R9().0.to_reg());
    let _w_r10 = Writable::<Reg>::from_reg(info_R10().0.to_reg());
    let w_r11 = Writable::<Reg>::from_reg(info_R11().0.to_reg());
    let w_r12 = Writable::<Reg>::from_reg(info_R12().0.to_reg());
    let w_r13 = Writable::<Reg>::from_reg(info_R13().0.to_reg());
    let w_r14 = Writable::<Reg>::from_reg(info_R14().0.to_reg());
    let w_r15 = Writable::<Reg>::from_reg(info_R15().0.to_reg());

    let mut insns = Vec::<(Inst, &str, &str)>::new();

    // ========================================================
    // Cases aimed at checking Addr-esses: IR (Imm + Reg)
    //
    // These are just a bunch of loads with all supported (by the emitter)
    // permutations of address formats.
    //
    // Addr_IR, offset zero
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rax), w_rdi),
        "488B38",
        "movq    0(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rbx), w_rdi),
        "488B3B",
        "movq    0(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rcx), w_rdi),
        "488B39",
        "movq    0(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rdx), w_rdi),
        "488B3A",
        "movq    0(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rbp), w_rdi),
        "488B7D00",
        "movq    0(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rsp), w_rdi),
        "488B3C24",
        "movq    0(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rsi), w_rdi),
        "488B3E",
        "movq    0(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, rdi), w_rdi),
        "488B3F",
        "movq    0(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r8), w_rdi),
        "498B38",
        "movq    0(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r9), w_rdi),
        "498B39",
        "movq    0(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r10), w_rdi),
        "498B3A",
        "movq    0(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r11), w_rdi),
        "498B3B",
        "movq    0(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r12), w_rdi),
        "498B3C24",
        "movq    0(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r13), w_rdi),
        "498B7D00",
        "movq    0(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r14), w_rdi),
        "498B3E",
        "movq    0(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0, r15), w_rdi),
        "498B3F",
        "movq    0(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset max simm8
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rax), w_rdi),
        "488B787F",
        "movq    127(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rbx), w_rdi),
        "488B7B7F",
        "movq    127(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rcx), w_rdi),
        "488B797F",
        "movq    127(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rdx), w_rdi),
        "488B7A7F",
        "movq    127(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rbp), w_rdi),
        "488B7D7F",
        "movq    127(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rsp), w_rdi),
        "488B7C247F",
        "movq    127(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rsi), w_rdi),
        "488B7E7F",
        "movq    127(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, rdi), w_rdi),
        "488B7F7F",
        "movq    127(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r8), w_rdi),
        "498B787F",
        "movq    127(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r9), w_rdi),
        "498B797F",
        "movq    127(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r10), w_rdi),
        "498B7A7F",
        "movq    127(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r11), w_rdi),
        "498B7B7F",
        "movq    127(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r12), w_rdi),
        "498B7C247F",
        "movq    127(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r13), w_rdi),
        "498B7D7F",
        "movq    127(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r14), w_rdi),
        "498B7E7F",
        "movq    127(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(127, r15), w_rdi),
        "498B7F7F",
        "movq    127(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset min simm8
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rax), w_rdi),
        "488B7880",
        "movq    -128(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rbx), w_rdi),
        "488B7B80",
        "movq    -128(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rcx), w_rdi),
        "488B7980",
        "movq    -128(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rdx), w_rdi),
        "488B7A80",
        "movq    -128(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rbp), w_rdi),
        "488B7D80",
        "movq    -128(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rsp), w_rdi),
        "488B7C2480",
        "movq    -128(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rsi), w_rdi),
        "488B7E80",
        "movq    -128(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, rdi), w_rdi),
        "488B7F80",
        "movq    -128(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r8), w_rdi),
        "498B7880",
        "movq    -128(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r9), w_rdi),
        "498B7980",
        "movq    -128(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r10), w_rdi),
        "498B7A80",
        "movq    -128(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r11), w_rdi),
        "498B7B80",
        "movq    -128(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r12), w_rdi),
        "498B7C2480",
        "movq    -128(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r13), w_rdi),
        "498B7D80",
        "movq    -128(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r14), w_rdi),
        "498B7E80",
        "movq    -128(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-128i32 as u32, r15), w_rdi),
        "498B7F80",
        "movq    -128(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset smallest positive simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rax), w_rdi),
        "488BB880000000",
        "movq    128(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rbx), w_rdi),
        "488BBB80000000",
        "movq    128(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rcx), w_rdi),
        "488BB980000000",
        "movq    128(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rdx), w_rdi),
        "488BBA80000000",
        "movq    128(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rbp), w_rdi),
        "488BBD80000000",
        "movq    128(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rsp), w_rdi),
        "488BBC2480000000",
        "movq    128(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rsi), w_rdi),
        "488BBE80000000",
        "movq    128(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, rdi), w_rdi),
        "488BBF80000000",
        "movq    128(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r8), w_rdi),
        "498BB880000000",
        "movq    128(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r9), w_rdi),
        "498BB980000000",
        "movq    128(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r10), w_rdi),
        "498BBA80000000",
        "movq    128(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r11), w_rdi),
        "498BBB80000000",
        "movq    128(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r12), w_rdi),
        "498BBC2480000000",
        "movq    128(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r13), w_rdi),
        "498BBD80000000",
        "movq    128(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r14), w_rdi),
        "498BBE80000000",
        "movq    128(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(128, r15), w_rdi),
        "498BBF80000000",
        "movq    128(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset smallest negative simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rax), w_rdi),
        "488BB87FFFFFFF",
        "movq    -129(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rbx), w_rdi),
        "488BBB7FFFFFFF",
        "movq    -129(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rcx), w_rdi),
        "488BB97FFFFFFF",
        "movq    -129(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rdx), w_rdi),
        "488BBA7FFFFFFF",
        "movq    -129(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rbp), w_rdi),
        "488BBD7FFFFFFF",
        "movq    -129(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rsp), w_rdi),
        "488BBC247FFFFFFF",
        "movq    -129(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rsi), w_rdi),
        "488BBE7FFFFFFF",
        "movq    -129(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, rdi), w_rdi),
        "488BBF7FFFFFFF",
        "movq    -129(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r8), w_rdi),
        "498BB87FFFFFFF",
        "movq    -129(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r9), w_rdi),
        "498BB97FFFFFFF",
        "movq    -129(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r10), w_rdi),
        "498BBA7FFFFFFF",
        "movq    -129(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r11), w_rdi),
        "498BBB7FFFFFFF",
        "movq    -129(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r12), w_rdi),
        "498BBC247FFFFFFF",
        "movq    -129(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r13), w_rdi),
        "498BBD7FFFFFFF",
        "movq    -129(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r14), w_rdi),
        "498BBE7FFFFFFF",
        "movq    -129(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-129i32 as u32, r15), w_rdi),
        "498BBF7FFFFFFF",
        "movq    -129(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset large positive simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rax), w_rdi),
        "488BB877207317",
        "movq    393420919(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rbx), w_rdi),
        "488BBB77207317",
        "movq    393420919(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rcx), w_rdi),
        "488BB977207317",
        "movq    393420919(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rdx), w_rdi),
        "488BBA77207317",
        "movq    393420919(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rbp), w_rdi),
        "488BBD77207317",
        "movq    393420919(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rsp), w_rdi),
        "488BBC2477207317",
        "movq    393420919(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rsi), w_rdi),
        "488BBE77207317",
        "movq    393420919(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, rdi), w_rdi),
        "488BBF77207317",
        "movq    393420919(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r8), w_rdi),
        "498BB877207317",
        "movq    393420919(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r9), w_rdi),
        "498BB977207317",
        "movq    393420919(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r10), w_rdi),
        "498BBA77207317",
        "movq    393420919(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r11), w_rdi),
        "498BBB77207317",
        "movq    393420919(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r12), w_rdi),
        "498BBC2477207317",
        "movq    393420919(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r13), w_rdi),
        "498BBD77207317",
        "movq    393420919(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r14), w_rdi),
        "498BBE77207317",
        "movq    393420919(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(0x17732077, r15), w_rdi),
        "498BBF77207317",
        "movq    393420919(%r15), %rdi",
    ));

    // ========================================================
    // Addr_IR, offset large negative simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rax), w_rdi),
        "488BB8D9A6BECE",
        "movq    -826366247(%rax), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rbx), w_rdi),
        "488BBBD9A6BECE",
        "movq    -826366247(%rbx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rcx), w_rdi),
        "488BB9D9A6BECE",
        "movq    -826366247(%rcx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rdx), w_rdi),
        "488BBAD9A6BECE",
        "movq    -826366247(%rdx), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rbp), w_rdi),
        "488BBDD9A6BECE",
        "movq    -826366247(%rbp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rsp), w_rdi),
        "488BBC24D9A6BECE",
        "movq    -826366247(%rsp), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rsi), w_rdi),
        "488BBED9A6BECE",
        "movq    -826366247(%rsi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, rdi), w_rdi),
        "488BBFD9A6BECE",
        "movq    -826366247(%rdi), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r8), w_rdi),
        "498BB8D9A6BECE",
        "movq    -826366247(%r8), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r9), w_rdi),
        "498BB9D9A6BECE",
        "movq    -826366247(%r9), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r10), w_rdi),
        "498BBAD9A6BECE",
        "movq    -826366247(%r10), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r11), w_rdi),
        "498BBBD9A6BECE",
        "movq    -826366247(%r11), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r12), w_rdi),
        "498BBC24D9A6BECE",
        "movq    -826366247(%r12), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r13), w_rdi),
        "498BBDD9A6BECE",
        "movq    -826366247(%r13), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r14), w_rdi),
        "498BBED9A6BECE",
        "movq    -826366247(%r14), %rdi",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IR(-0x31415927i32 as u32, r15), w_rdi),
        "498BBFD9A6BECE",
        "movq    -826366247(%r15), %rdi",
    ));

    // ========================================================
    // Cases aimed at checking Addr-esses: IRRS (Imm + Reg + (Reg << Shift))
    // Note these don't check the case where the index reg is RSP, since we
    // don't encode any of those.
    //
    // Addr_IRRS, offset max simm8
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, rax, rax, 0), w_r11),
        "4C8B5C007F",
        "movq    127(%rax,%rax,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, rdi, rax, 1), w_r11),
        "4C8B5C477F",
        "movq    127(%rdi,%rax,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, r8, rax, 2), w_r11),
        "4D8B5C807F",
        "movq    127(%r8,%rax,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, r15, rax, 3), w_r11),
        "4D8B5CC77F",
        "movq    127(%r15,%rax,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, rax, rdi, 3), w_r11),
        "4C8B5CF87F",
        "movq    127(%rax,%rdi,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, rdi, rdi, 2), w_r11),
        "4C8B5CBF7F",
        "movq    127(%rdi,%rdi,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, r8, rdi, 1), w_r11),
        "4D8B5C787F",
        "movq    127(%r8,%rdi,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(127, r15, rdi, 0), w_r11),
        "4D8B5C3F7F",
        "movq    127(%r15,%rdi,1), %r11",
    ));

    // ========================================================
    // Addr_IRRS, offset min simm8
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, rax, r8, 2), w_r11),
        "4E8B5C8080",
        "movq    -128(%rax,%r8,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, rdi, r8, 3), w_r11),
        "4E8B5CC780",
        "movq    -128(%rdi,%r8,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, r8, r8, 0), w_r11),
        "4F8B5C0080",
        "movq    -128(%r8,%r8,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, r15, r8, 1), w_r11),
        "4F8B5C4780",
        "movq    -128(%r15,%r8,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, rax, r15, 1), w_r11),
        "4E8B5C7880",
        "movq    -128(%rax,%r15,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, rdi, r15, 0), w_r11),
        "4E8B5C3F80",
        "movq    -128(%rdi,%r15,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, r8, r15, 3), w_r11),
        "4F8B5CF880",
        "movq    -128(%r8,%r15,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-128i32 as u32, r15, r15, 2), w_r11),
        "4F8B5CBF80",
        "movq    -128(%r15,%r15,4), %r11",
    ));

    // ========================================================
    // Addr_IRRS, offset large positive simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, rax, rax, 0), w_r11),
        "4C8B9C00BE25664F",
        "movq    1332094398(%rax,%rax,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, rdi, rax, 1), w_r11),
        "4C8B9C47BE25664F",
        "movq    1332094398(%rdi,%rax,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, r8, rax, 2), w_r11),
        "4D8B9C80BE25664F",
        "movq    1332094398(%r8,%rax,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, r15, rax, 3), w_r11),
        "4D8B9CC7BE25664F",
        "movq    1332094398(%r15,%rax,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, rax, rdi, 3), w_r11),
        "4C8B9CF8BE25664F",
        "movq    1332094398(%rax,%rdi,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, rdi, rdi, 2), w_r11),
        "4C8B9CBFBE25664F",
        "movq    1332094398(%rdi,%rdi,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, r8, rdi, 1), w_r11),
        "4D8B9C78BE25664F",
        "movq    1332094398(%r8,%rdi,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(0x4f6625be, r15, rdi, 0), w_r11),
        "4D8B9C3FBE25664F",
        "movq    1332094398(%r15,%rdi,1), %r11",
    ));

    // ========================================================
    // Addr_IRRS, offset large negative simm32
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, rax, r8, 2), w_r11),
        "4E8B9C8070E9B2D9",
        "movq    -642586256(%rax,%r8,4), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, rdi, r8, 3), w_r11),
        "4E8B9CC770E9B2D9",
        "movq    -642586256(%rdi,%r8,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, r8, r8, 0), w_r11),
        "4F8B9C0070E9B2D9",
        "movq    -642586256(%r8,%r8,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, r15, r8, 1), w_r11),
        "4F8B9C4770E9B2D9",
        "movq    -642586256(%r15,%r8,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, rax, r15, 1), w_r11),
        "4E8B9C7870E9B2D9",
        "movq    -642586256(%rax,%r15,2), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, rdi, r15, 0), w_r11),
        "4E8B9C3F70E9B2D9",
        "movq    -642586256(%rdi,%r15,1), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, r8, r15, 3), w_r11),
        "4F8B9CF870E9B2D9",
        "movq    -642586256(%r8,%r15,8), %r11",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(-0x264d1690i32 as u32, r15, r15, 2), w_r11),
        "4F8B9CBF70E9B2D9",
        "movq    -642586256(%r15,%r15,4), %r11",
    ));

    // End of test cases for Addr
    // ========================================================

    // ========================================================
    // General tests for each insn.  Don't forget to follow the
    // guidelines commented just prior to |fn x64_emit|.
    //
    // Alu_RMI_R
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, ip_RMI_R(r15), w_rdx),
        "4C01FA",
        "addq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_R(rcx), w_r8),
        "4101C8",
        "addl    %ecx, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_R(rcx), w_rsi),
        "01CE",
        "addl    %ecx, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, ip_RMI_M(ip_Addr_IR(99, rdi)), w_rdx),
        "48035763",
        "addq    99(%rdi), %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_M(ip_Addr_IR(99, rdi)), w_r8),
        "44034763",
        "addl    99(%rdi), %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_M(ip_Addr_IR(99, rdi)), w_rsi),
        "037763",
        "addl    99(%rdi), %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, ip_RMI_I(-127i32 as u32), w_rdx),
        "4883C281",
        "addq    $-127, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, ip_RMI_I(-129i32 as u32), w_rdx),
        "4881C27FFFFFFF",
        "addq    $-129, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Add, ip_RMI_I(76543210), w_rdx),
        "4881C2EAF48F04",
        "addq    $76543210, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(-127i32 as u32), w_r8),
        "4183C081",
        "addl    $-127, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(-129i32 as u32), w_r8),
        "4181C07FFFFFFF",
        "addl    $-129, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(-76543210i32 as u32), w_r8),
        "4181C0160B70FB",
        "addl    $-76543210, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(-127i32 as u32), w_rsi),
        "83C681",
        "addl    $-127, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(-129i32 as u32), w_rsi),
        "81C67FFFFFFF",
        "addl    $-129, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Add, ip_RMI_I(76543210), w_rsi),
        "81C6EAF48F04",
        "addl    $76543210, %esi",
    ));
    // This is pretty feeble
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Sub, ip_RMI_R(r15), w_rdx),
        "4C29FA",
        "subq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::And, ip_RMI_R(r15), w_rdx),
        "4C21FA",
        "andq    %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Or, ip_RMI_R(r15), w_rdx),
        "4C09FA",
        "orq     %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Xor, ip_RMI_R(r15), w_rdx),
        "4C31FA",
        "xorq    %r15, %rdx",
    ));
    // Test all mul cases, though
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, ip_RMI_R(r15), w_rdx),
        "490FAFD7",
        "imulq   %r15, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_R(rcx), w_r8),
        "440FAFC1",
        "imull   %ecx, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_R(rcx), w_rsi),
        "0FAFF1",
        "imull   %ecx, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, ip_RMI_M(ip_Addr_IR(99, rdi)), w_rdx),
        "480FAF5763",
        "imulq   99(%rdi), %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_M(ip_Addr_IR(99, rdi)), w_r8),
        "440FAF4763",
        "imull   99(%rdi), %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_M(ip_Addr_IR(99, rdi)), w_rsi),
        "0FAF7763",
        "imull   99(%rdi), %esi",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, ip_RMI_I(-127i32 as u32), w_rdx),
        "486BD281",
        "imulq   $-127, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, ip_RMI_I(-129i32 as u32), w_rdx),
        "4869D27FFFFFFF",
        "imulq   $-129, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(true, RMI_R_Op::Mul, ip_RMI_I(76543210), w_rdx),
        "4869D2EAF48F04",
        "imulq   $76543210, %rdx",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(-127i32 as u32), w_r8),
        "456BC081",
        "imull   $-127, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(-129i32 as u32), w_r8),
        "4569C07FFFFFFF",
        "imull   $-129, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(-76543210i32 as u32), w_r8),
        "4569C0160B70FB",
        "imull   $-76543210, %r8d",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(-127i32 as u32), w_rsi),
        "6BF681",
        "imull   $-127, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(-129i32 as u32), w_rsi),
        "69F67FFFFFFF",
        "imull   $-129, %esi",
    ));
    insns.push((
        i_Alu_RMI_R(false, RMI_R_Op::Mul, ip_RMI_I(76543210), w_rsi),
        "69F6EAF48F04",
        "imull   $76543210, %esi",
    ));

    // ========================================================
    // Imm_R
    //
    insns.push((
        i_Imm_R(false, 1234567, w_r14),
        "41BE87D61200",
        "movl    $1234567, %r14d",
    ));
    insns.push((
        i_Imm_R(false, -126i64 as u64, w_r14),
        "41BE82FFFFFF",
        "movl    $-126, %r14d",
    ));
    insns.push((
        i_Imm_R(true, 1234567898765, w_r14),
        "49BE8D26FB711F010000",
        "movabsq $1234567898765, %r14",
    ));
    insns.push((
        i_Imm_R(true, -126i64 as u64, w_r14),
        "49BE82FFFFFFFFFFFFFF",
        "movabsq $-126, %r14",
    ));
    insns.push((
        i_Imm_R(false, 1234567, w_rcx),
        "B987D61200",
        "movl    $1234567, %ecx",
    ));
    insns.push((
        i_Imm_R(false, -126i64 as u64, w_rcx),
        "B982FFFFFF",
        "movl    $-126, %ecx",
    ));
    insns.push((
        i_Imm_R(true, 1234567898765, w_rsi),
        "48BE8D26FB711F010000",
        "movabsq $1234567898765, %rsi",
    ));
    insns.push((
        i_Imm_R(true, -126i64 as u64, w_rbx),
        "48BB82FFFFFFFFFFFFFF",
        "movabsq $-126, %rbx",
    ));

    // ========================================================
    // Mov_R_R
    insns.push((i_Mov_R_R(false, rbx, w_rsi), "89DE", "movl    %ebx, %esi"));
    insns.push((i_Mov_R_R(false, rbx, w_r9), "4189D9", "movl    %ebx, %r9d"));
    insns.push((
        i_Mov_R_R(false, r11, w_rsi),
        "4489DE",
        "movl    %r11d, %esi",
    ));
    insns.push((i_Mov_R_R(false, r12, w_r9), "4589E1", "movl    %r12d, %r9d"));
    insns.push((i_Mov_R_R(true, rbx, w_rsi), "4889DE", "movq    %rbx, %rsi"));
    insns.push((i_Mov_R_R(true, rbx, w_r9), "4989D9", "movq    %rbx, %r9"));
    insns.push((i_Mov_R_R(true, r11, w_rsi), "4C89DE", "movq    %r11, %rsi"));
    insns.push((i_Mov_R_R(true, r12, w_r9), "4D89E1", "movq    %r12, %r9"));

    // ========================================================
    // MovZX_M_R
    insns.push((
        i_MovZX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "0FB671F9",
        "movzbl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "410FB658F9",
        "movzbl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "450FB64AF9",
        "movzbl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "410FB653F9",
        "movzbl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "480FB671F9",
        "movzbq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "490FB658F9",
        "movzbq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "4D0FB64AF9",
        "movzbq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "490FB653F9",
        "movzbq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "0FB771F9",
        "movzwl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "410FB758F9",
        "movzwl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "450FB74AF9",
        "movzwl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "410FB753F9",
        "movzwl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "480FB771F9",
        "movzwq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "490FB758F9",
        "movzwq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "4D0FB74AF9",
        "movzwq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "490FB753F9",
        "movzwq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "8B71F9",
        "movl    -7(%rcx), %esi",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "418B58F9",
        "movl    -7(%r8), %ebx",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "458B4AF9",
        "movl    -7(%r10), %r9d",
    ));
    insns.push((
        i_MovZX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "418B53F9",
        "movl    -7(%r11), %edx",
    ));

    // ========================================================
    // Mov64_M_R
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, rax, rbx, 0), w_rcx),
        "488B8C18B3000000",
        "movq    179(%rax,%rbx,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, rax, rbx, 0), w_r8),
        "4C8B8418B3000000",
        "movq    179(%rax,%rbx,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, rax, r9, 0), w_rcx),
        "4A8B8C08B3000000",
        "movq    179(%rax,%r9,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, rax, r9, 0), w_r8),
        "4E8B8408B3000000",
        "movq    179(%rax,%r9,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, r10, rbx, 0), w_rcx),
        "498B8C1AB3000000",
        "movq    179(%r10,%rbx,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, r10, rbx, 0), w_r8),
        "4D8B841AB3000000",
        "movq    179(%r10,%rbx,1), %r8",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, r10, r9, 0), w_rcx),
        "4B8B8C0AB3000000",
        "movq    179(%r10,%r9,1), %rcx",
    ));
    insns.push((
        i_Mov64_M_R(ip_Addr_IRRS(179, r10, r9, 0), w_r8),
        "4F8B840AB3000000",
        "movq    179(%r10,%r9,1), %r8",
    ));

    // ========================================================
    // MovSX_M_R
    insns.push((
        i_MovSX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "0FBE71F9",
        "movsbl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "410FBE58F9",
        "movsbl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "450FBE4AF9",
        "movsbl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BL, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "410FBE53F9",
        "movsbl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "480FBE71F9",
        "movsbq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "490FBE58F9",
        "movsbq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "4D0FBE4AF9",
        "movsbq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::BQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "490FBE53F9",
        "movsbq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "0FBF71F9",
        "movswl  -7(%rcx), %esi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "410FBF58F9",
        "movswl  -7(%r8), %ebx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "450FBF4AF9",
        "movswl  -7(%r10), %r9d",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WL, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "410FBF53F9",
        "movswl  -7(%r11), %edx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "480FBF71F9",
        "movswq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "490FBF58F9",
        "movswq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "4D0FBF4AF9",
        "movswq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::WQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "490FBF53F9",
        "movswq  -7(%r11), %rdx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, rcx), w_rsi),
        "486371F9",
        "movslq  -7(%rcx), %rsi",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r8), w_rbx),
        "496358F9",
        "movslq  -7(%r8), %rbx",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r10), w_r9),
        "4D634AF9",
        "movslq  -7(%r10), %r9",
    ));
    insns.push((
        i_MovSX_M_R(ExtMode::LQ, ip_Addr_IR(-7i32 as u32, r11), w_rdx),
        "496353F9",
        "movslq  -7(%r11), %rdx",
    ));

    // ========================================================
    // Mov_R_M.  Byte stores are tricky.  Check everything carefully.
    insns.push((
        i_Mov_R_M(8, rax, ip_Addr_IR(99, rdi)),
        "48894763",
        "movq    %rax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(8, rbx, ip_Addr_IR(99, r8)),
        "49895863",
        "movq    %rbx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(8, rcx, ip_Addr_IR(99, rsi)),
        "48894E63",
        "movq    %rcx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(8, rdx, ip_Addr_IR(99, r9)),
        "49895163",
        "movq    %rdx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(8, rsi, ip_Addr_IR(99, rax)),
        "48897063",
        "movq    %rsi, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(8, rdi, ip_Addr_IR(99, r15)),
        "49897F63",
        "movq    %rdi, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(8, rsp, ip_Addr_IR(99, rcx)),
        "48896163",
        "movq    %rsp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(8, rbp, ip_Addr_IR(99, r14)),
        "49896E63",
        "movq    %rbp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(8, r8, ip_Addr_IR(99, rdi)),
        "4C894763",
        "movq    %r8, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(8, r9, ip_Addr_IR(99, r8)),
        "4D894863",
        "movq    %r9, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(8, r10, ip_Addr_IR(99, rsi)),
        "4C895663",
        "movq    %r10, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(8, r11, ip_Addr_IR(99, r9)),
        "4D895963",
        "movq    %r11, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(8, r12, ip_Addr_IR(99, rax)),
        "4C896063",
        "movq    %r12, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(8, r13, ip_Addr_IR(99, r15)),
        "4D896F63",
        "movq    %r13, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(8, r14, ip_Addr_IR(99, rcx)),
        "4C897163",
        "movq    %r14, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(8, r15, ip_Addr_IR(99, r14)),
        "4D897E63",
        "movq    %r15, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(4, rax, ip_Addr_IR(99, rdi)),
        "894763",
        "movl    %eax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(4, rbx, ip_Addr_IR(99, r8)),
        "41895863",
        "movl    %ebx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(4, rcx, ip_Addr_IR(99, rsi)),
        "894E63",
        "movl    %ecx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(4, rdx, ip_Addr_IR(99, r9)),
        "41895163",
        "movl    %edx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(4, rsi, ip_Addr_IR(99, rax)),
        "897063",
        "movl    %esi, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(4, rdi, ip_Addr_IR(99, r15)),
        "41897F63",
        "movl    %edi, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(4, rsp, ip_Addr_IR(99, rcx)),
        "896163",
        "movl    %esp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(4, rbp, ip_Addr_IR(99, r14)),
        "41896E63",
        "movl    %ebp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(4, r8, ip_Addr_IR(99, rdi)),
        "44894763",
        "movl    %r8d, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(4, r9, ip_Addr_IR(99, r8)),
        "45894863",
        "movl    %r9d, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(4, r10, ip_Addr_IR(99, rsi)),
        "44895663",
        "movl    %r10d, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(4, r11, ip_Addr_IR(99, r9)),
        "45895963",
        "movl    %r11d, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(4, r12, ip_Addr_IR(99, rax)),
        "44896063",
        "movl    %r12d, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(4, r13, ip_Addr_IR(99, r15)),
        "45896F63",
        "movl    %r13d, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(4, r14, ip_Addr_IR(99, rcx)),
        "44897163",
        "movl    %r14d, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(4, r15, ip_Addr_IR(99, r14)),
        "45897E63",
        "movl    %r15d, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(2, rax, ip_Addr_IR(99, rdi)),
        "66894763",
        "movw    %ax, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(2, rbx, ip_Addr_IR(99, r8)),
        "6641895863",
        "movw    %bx, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(2, rcx, ip_Addr_IR(99, rsi)),
        "66894E63",
        "movw    %cx, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(2, rdx, ip_Addr_IR(99, r9)),
        "6641895163",
        "movw    %dx, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(2, rsi, ip_Addr_IR(99, rax)),
        "66897063",
        "movw    %si, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(2, rdi, ip_Addr_IR(99, r15)),
        "6641897F63",
        "movw    %di, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(2, rsp, ip_Addr_IR(99, rcx)),
        "66896163",
        "movw    %sp, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(2, rbp, ip_Addr_IR(99, r14)),
        "6641896E63",
        "movw    %bp, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(2, r8, ip_Addr_IR(99, rdi)),
        "6644894763",
        "movw    %r8w, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(2, r9, ip_Addr_IR(99, r8)),
        "6645894863",
        "movw    %r9w, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(2, r10, ip_Addr_IR(99, rsi)),
        "6644895663",
        "movw    %r10w, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(2, r11, ip_Addr_IR(99, r9)),
        "6645895963",
        "movw    %r11w, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(2, r12, ip_Addr_IR(99, rax)),
        "6644896063",
        "movw    %r12w, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(2, r13, ip_Addr_IR(99, r15)),
        "6645896F63",
        "movw    %r13w, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(2, r14, ip_Addr_IR(99, rcx)),
        "6644897163",
        "movw    %r14w, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(2, r15, ip_Addr_IR(99, r14)),
        "6645897E63",
        "movw    %r15w, 99(%r14)",
    ));
    //
    insns.push((
        i_Mov_R_M(1, rax, ip_Addr_IR(99, rdi)),
        "884763",
        "movb    %al, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(1, rbx, ip_Addr_IR(99, r8)),
        "41885863",
        "movb    %bl, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(1, rcx, ip_Addr_IR(99, rsi)),
        "884E63",
        "movb    %cl, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(1, rdx, ip_Addr_IR(99, r9)),
        "41885163",
        "movb    %dl, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(1, rsi, ip_Addr_IR(99, rax)),
        "40887063",
        "movb    %sil, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(1, rdi, ip_Addr_IR(99, r15)),
        "41887F63",
        "movb    %dil, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(1, rsp, ip_Addr_IR(99, rcx)),
        "40886163",
        "movb    %spl, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(1, rbp, ip_Addr_IR(99, r14)),
        "41886E63",
        "movb    %bpl, 99(%r14)",
    ));
    insns.push((
        i_Mov_R_M(1, r8, ip_Addr_IR(99, rdi)),
        "44884763",
        "movb    %r8b, 99(%rdi)",
    ));
    insns.push((
        i_Mov_R_M(1, r9, ip_Addr_IR(99, r8)),
        "45884863",
        "movb    %r9b, 99(%r8)",
    ));
    insns.push((
        i_Mov_R_M(1, r10, ip_Addr_IR(99, rsi)),
        "44885663",
        "movb    %r10b, 99(%rsi)",
    ));
    insns.push((
        i_Mov_R_M(1, r11, ip_Addr_IR(99, r9)),
        "45885963",
        "movb    %r11b, 99(%r9)",
    ));
    insns.push((
        i_Mov_R_M(1, r12, ip_Addr_IR(99, rax)),
        "44886063",
        "movb    %r12b, 99(%rax)",
    ));
    insns.push((
        i_Mov_R_M(1, r13, ip_Addr_IR(99, r15)),
        "45886F63",
        "movb    %r13b, 99(%r15)",
    ));
    insns.push((
        i_Mov_R_M(1, r14, ip_Addr_IR(99, rcx)),
        "44887163",
        "movb    %r14b, 99(%rcx)",
    ));
    insns.push((
        i_Mov_R_M(1, r15, ip_Addr_IR(99, r14)),
        "45887E63",
        "movb    %r15b, 99(%r14)",
    ));

    // ========================================================
    // Shift_R
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 0, w_rdi),
        "D3E7",
        "shll    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 0, w_r12),
        "41D3E4",
        "shll    %cl, %r12d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 2, w_r8),
        "41C1E002",
        "shll    $2, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::Left, 31, w_r13),
        "41C1E51F",
        "shll    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 0, w_r13),
        "49D3E5",
        "shlq    %cl, %r13",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 0, w_rdi),
        "48D3E7",
        "shlq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 2, w_r8),
        "49C1E002",
        "shlq    $2, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 3, w_rbx),
        "48C1E303",
        "shlq    $3, %rbx",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::Left, 63, w_r13),
        "49C1E53F",
        "shlq    $63, %r13",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 0, w_rdi),
        "D3EF",
        "shrl    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 2, w_r8),
        "41C1E802",
        "shrl    $2, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightZ, 31, w_r13),
        "41C1ED1F",
        "shrl    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 0, w_rdi),
        "48D3EF",
        "shrq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 2, w_r8),
        "49C1E802",
        "shrq    $2, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightZ, 63, w_r13),
        "49C1ED3F",
        "shrq    $63, %r13",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 0, w_rdi),
        "D3FF",
        "sarl    %cl, %edi",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 2, w_r8),
        "41C1F802",
        "sarl    $2, %r8d",
    ));
    insns.push((
        i_Shift_R(false, ShiftKind::RightS, 31, w_r13),
        "41C1FD1F",
        "sarl    $31, %r13d",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 0, w_rdi),
        "48D3FF",
        "sarq    %cl, %rdi",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 2, w_r8),
        "49C1F802",
        "sarq    $2, %r8",
    ));
    insns.push((
        i_Shift_R(true, ShiftKind::RightS, 63, w_r13),
        "49C1FD3F",
        "sarq    $63, %r13",
    ));

    // ========================================================
    // Cmp_RMI_R
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_R(r15), rdx),
        "4C39FA",
        "cmpq    %r15, %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_R(rcx), r8),
        "4939C8",
        "cmpq    %rcx, %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_R(rcx), rsi),
        "4839CE",
        "cmpq    %rcx, %rsi",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_M(ip_Addr_IR(99, rdi)), rdx),
        "483B5763",
        "cmpq    99(%rdi), %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_M(ip_Addr_IR(99, rdi)), r8),
        "4C3B4763",
        "cmpq    99(%rdi), %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_M(ip_Addr_IR(99, rdi)), rsi),
        "483B7763",
        "cmpq    99(%rdi), %rsi",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_I(76543210), rdx),
        "4881FAEAF48F04",
        "cmpq    $76543210, %rdx",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_I(-76543210i32 as u32), r8),
        "4981F8160B70FB",
        "cmpq    $-76543210, %r8",
    ));
    insns.push((
        i_Cmp_RMI_R(8, ip_RMI_I(76543210), rsi),
        "4881FEEAF48F04",
        "cmpq    $76543210, %rsi",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_R(r15), rdx),
        "4439FA",
        "cmpl    %r15d, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_R(rcx), r8),
        "4139C8",
        "cmpl    %ecx, %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_R(rcx), rsi),
        "39CE",
        "cmpl    %ecx, %esi",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_M(ip_Addr_IR(99, rdi)), rdx),
        "3B5763",
        "cmpl    99(%rdi), %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_M(ip_Addr_IR(99, rdi)), r8),
        "443B4763",
        "cmpl    99(%rdi), %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_M(ip_Addr_IR(99, rdi)), rsi),
        "3B7763",
        "cmpl    99(%rdi), %esi",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_I(76543210), rdx),
        "81FAEAF48F04",
        "cmpl    $76543210, %edx",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_I(-76543210i32 as u32), r8),
        "4181F8160B70FB",
        "cmpl    $-76543210, %r8d",
    ));
    insns.push((
        i_Cmp_RMI_R(4, ip_RMI_I(76543210), rsi),
        "81FEEAF48F04",
        "cmpl    $76543210, %esi",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_R(r15), rdx),
        "664439FA",
        "cmpw    %r15w, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_R(rcx), r8),
        "664139C8",
        "cmpw    %cx, %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_R(rcx), rsi),
        "6639CE",
        "cmpw    %cx, %si",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_M(ip_Addr_IR(99, rdi)), rdx),
        "663B5763",
        "cmpw    99(%rdi), %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_M(ip_Addr_IR(99, rdi)), r8),
        "66443B4763",
        "cmpw    99(%rdi), %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_M(ip_Addr_IR(99, rdi)), rsi),
        "663B7763",
        "cmpw    99(%rdi), %si",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_I(23210), rdx),
        "6681FAAA5A",
        "cmpw    $23210, %dx",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_I(-7654i32 as u32), r8),
        "664181F81AE2",
        "cmpw    $-7654, %r8w",
    ));
    insns.push((
        i_Cmp_RMI_R(2, ip_RMI_I(7654), rsi),
        "6681FEE61D",
        "cmpw    $7654, %si",
    ));
    //
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r15), rdx),
        "4438FA",
        "cmpb    %r15b, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), r8),
        "4138C8",
        "cmpb    %cl, %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), rsi),
        "4038CE",
        "cmpb    %cl, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_M(ip_Addr_IR(99, rdi)), rdx),
        "3A5763",
        "cmpb    99(%rdi), %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_M(ip_Addr_IR(99, rdi)), r8),
        "443A4763",
        "cmpb    99(%rdi), %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_M(ip_Addr_IR(99, rdi)), rsi),
        "403A7763",
        "cmpb    99(%rdi), %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_I(70), rdx),
        "80FA46",
        "cmpb    $70, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_I(-76i32 as u32), r8),
        "4180F8B4",
        "cmpb    $-76, %r8b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_I(76), rsi),
        "4080FE4C",
        "cmpb    $76, %sil",
    ));
    // Extra byte-cases (paranoia!) for Cmp_RMI_R for first operand = R
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rax), rbx),
        "38C3",
        "cmpb    %al, %bl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rbx), rax),
        "38D8",
        "cmpb    %bl, %al",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), rdx),
        "38CA",
        "cmpb    %cl, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), rsi),
        "4038CE",
        "cmpb    %cl, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), r10),
        "4138CA",
        "cmpb    %cl, %r10b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rcx), r14),
        "4138CE",
        "cmpb    %cl, %r14b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rbp), rdx),
        "4038EA",
        "cmpb    %bpl, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rbp), rsi),
        "4038EE",
        "cmpb    %bpl, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rbp), r10),
        "4138EA",
        "cmpb    %bpl, %r10b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(rbp), r14),
        "4138EE",
        "cmpb    %bpl, %r14b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r9), rdx),
        "4438CA",
        "cmpb    %r9b, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r9), rsi),
        "4438CE",
        "cmpb    %r9b, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r9), r10),
        "4538CA",
        "cmpb    %r9b, %r10b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r9), r14),
        "4538CE",
        "cmpb    %r9b, %r14b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r13), rdx),
        "4438EA",
        "cmpb    %r13b, %dl",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r13), rsi),
        "4438EE",
        "cmpb    %r13b, %sil",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r13), r10),
        "4538EA",
        "cmpb    %r13b, %r10b",
    ));
    insns.push((
        i_Cmp_RMI_R(1, ip_RMI_R(r13), r14),
        "4538EE",
        "cmpb    %r13b, %r14b",
    ));

    // ========================================================
    // Push64
    insns.push((i_Push64(ip_RMI_R(rdi)), "57", "pushq   %rdi"));
    insns.push((i_Push64(ip_RMI_R(r8)), "4150", "pushq   %r8"));
    insns.push((
        i_Push64(ip_RMI_M(ip_Addr_IRRS(321, rsi, rcx, 3))),
        "FFB4CE41010000",
        "pushq   321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_Push64(ip_RMI_M(ip_Addr_IRRS(321, r9, rbx, 2))),
        "41FFB49941010000",
        "pushq   321(%r9,%rbx,4)",
    ));
    insns.push((i_Push64(ip_RMI_I(0)), "6A00", "pushq   $0"));
    insns.push((i_Push64(ip_RMI_I(127)), "6A7F", "pushq   $127"));
    insns.push((i_Push64(ip_RMI_I(128)), "6880000000", "pushq   $128"));
    insns.push((
        i_Push64(ip_RMI_I(0x31415927)),
        "6827594131",
        "pushq   $826366247",
    ));
    insns.push((i_Push64(ip_RMI_I(-128i32 as u32)), "6A80", "pushq   $-128"));
    insns.push((
        i_Push64(ip_RMI_I(-129i32 as u32)),
        "687FFFFFFF",
        "pushq   $-129",
    ));
    insns.push((
        i_Push64(ip_RMI_I(-0x75c4e8a1i32 as u32)),
        "685F173B8A",
        "pushq   $-1975838881",
    ));

    // ========================================================
    // Pop64
    insns.push((i_Pop64(w_rax), "58", "popq    %rax"));
    insns.push((i_Pop64(w_rdi), "5F", "popq    %rdi"));
    insns.push((i_Pop64(w_r8), "4158", "popq    %r8"));
    insns.push((i_Pop64(w_r15), "415F", "popq    %r15"));

    // ========================================================
    // CallKnown skipped for now

    // ========================================================
    // CallUnknown
    insns.push((i_CallUnknown(ip_RM_R(rbp)), "FFD5", "call    *%rbp"));
    insns.push((i_CallUnknown(ip_RM_R(r11)), "41FFD3", "call    *%r11"));
    insns.push((
        i_CallUnknown(ip_RM_M(ip_Addr_IRRS(321, rsi, rcx, 3))),
        "FF94CE41010000",
        "call    *321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_CallUnknown(ip_RM_M(ip_Addr_IRRS(321, r10, rdx, 2))),
        "41FF949241010000",
        "call    *321(%r10,%rdx,4)",
    ));

    // ========================================================
    // Ret
    insns.push((i_Ret(), "C3", "ret"));

    // ========================================================
    // JmpKnown skipped for now

    // ========================================================
    // JmpCondSymm isn't a real instruction

    // ========================================================
    // JmpCond skipped for now

    // ========================================================
    // JmpCondCompound isn't a real instruction

    // ========================================================
    // JmpUnknown
    insns.push((i_JmpUnknown(ip_RM_R(rbp)), "FFE5", "jmp     *%rbp"));
    insns.push((i_JmpUnknown(ip_RM_R(r11)), "41FFE3", "jmp     *%r11"));
    insns.push((
        i_JmpUnknown(ip_RM_M(ip_Addr_IRRS(321, rsi, rcx, 3))),
        "FFA4CE41010000",
        "jmp     *321(%rsi,%rcx,8)",
    ));
    insns.push((
        i_JmpUnknown(ip_RM_M(ip_Addr_IRRS(321, r10, rdx, 2))),
        "41FFA49241010000",
        "jmp     *321(%r10,%rdx,4)",
    ));

    // ========================================================
    // Actually run the tests!
    let rru = create_reg_universe();
    let mut nullcps = NullConstantPoolSink {};
    for (insn, expected_encoding, expected_printing) in insns {
        println!("     {}", insn.show_rru(Some(&rru)));
        // Check the printed text is as expected.
        let actual_printing = insn.show_rru(Some(&rru));
        assert_eq!(expected_printing, actual_printing);

        // Check the encoding is as expected.
        let mut sink = test_utils::TestCodeSink::new();
        insn.emit(&mut sink, &mut nullcps);
        let actual_encoding = &sink.stringify();
        assert_eq!(expected_encoding, actual_encoding);
    }
    println!("QQQQ END test_x64_insn_encoding_and_printing");
}
