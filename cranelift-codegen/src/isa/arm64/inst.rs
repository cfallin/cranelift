//! This module defines arm64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{CodeOffset, CodeSink};
use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{Ebb, FuncRef, GlobalValue, Type};
use crate::machinst::*;

use regalloc::Map as RegallocMap;
use regalloc::{InstRegUses, Set};
use regalloc::{RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, NUM_REG_CLASSES};

use smallvec::SmallVec;
use std::mem;
use std::string::{String, ToString};
use std::sync::Once;

// ------------- registers ----------------

/// Get a reference to an X-register (integer register).
pub fn xreg(num: u8) -> Reg {
    assert!(num < 32);
    Reg::new_real(
        RegClass::I64,
        /* enc = */ num,
        /* index = */ 32u8 + num,
    )
}

/// Get a reference to the zero-register.
pub fn zero_reg() -> Reg {
    // This should be the same as what xreg(31) returns.
    Reg::new_real(
        RegClass::I64,
        /* enc = */ 31,
        /* index = */ 32u8 + 31,
    )
}

/// Get a reference to the stack-pointer register.
pub fn stack_reg() -> Reg {
    // XSP (stack) and XZR (zero) are logically different registers which have
    // the same hardware encoding, and whose meaning, in real arm64
    // instructions, is context-dependent.  For convenience of
    // universe-construction and for correct printing, we make them be two
    // different real registers.
    Reg::new_real(
        RegClass::I64,
        /* enc = */ 31,
        /* index = */ 32u8 + 31 + 1,
    )
}

/// Get a reference to the link register (x30).
pub fn link_reg() -> Reg {
    xreg(30)
}

/// Get a reference to the frame pointer (x29).
pub fn fp_reg() -> Reg {
    xreg(29)
}

/// Get a reference to the "spill temp" register. This register is used to
/// compute the address of a spill slot when a direct offset addressing mode from
/// FP is not sufficient (+/- 2^11 words). We exclude this register from regalloc
/// and reserve it for this purpose for simplicity; otherwise we need a
/// multi-stage analysis where we first determine how many spill slots we have,
/// then perhaps remove the reg from the pool and recompute regalloc.
pub fn spilltmp_reg() -> Reg {
    xreg(15)
}

/// Create the register universe for ARM64.
fn create_reg_universe() -> RealRegUniverse {
    let mut regs = vec![];
    let mut allocable_by_class = [None; NUM_REG_CLASSES];

    // Numbering Scheme: we put V-regs first, then X-regs. The X-regs
    // exclude several registers: x18 (globally reserved for platform-specific
    // purposes), x29 (frame pointer), x30 (link register), x31 (stack pointer
    // or zero register, depending on context).

    let v_reg_base = 0u8; // in contiguous real-register index space
    let v_reg_count = 32;
    for i in 0u8..v_reg_count {
        let reg = Reg::new_real(
            RegClass::V128,
            /* enc = */ i,
            /* index = */ v_reg_base + i,
        )
        .to_real_reg();
        let name = format!("v{}", i);
        regs.push((reg, name));
    }
    let v_reg_last = v_reg_base + v_reg_count - 1;

    let x_reg_base = 32u8; // in contiguous real-register index space
    let mut x_reg_count = 0;
    for i in 0u8..32u8 {
        // See above for excluded registers.
        if i == 15 || i == 18 || i == 29 || i == 30 || i == 31 {
            continue;
        }
        let reg = Reg::new_real(
            RegClass::I64,
            /* enc = */ i,
            /* index = */ x_reg_base + x_reg_count,
        )
        .to_real_reg();
        let name = format!("x{}", i);
        regs.push((reg, name));
        x_reg_count += 1;
    }
    let x_reg_last = x_reg_base + x_reg_count - 1;

    allocable_by_class[RegClass::I64.rc_to_usize()] =
        Some((x_reg_base as usize, x_reg_last as usize));
    allocable_by_class[RegClass::V128.rc_to_usize()] =
        Some((v_reg_base as usize, v_reg_last as usize));

    // Other regs, not available to the allocator.
    let allocable = regs.len();
    regs.push((xreg(15).to_real_reg(), "x15".to_string()));
    regs.push((xreg(18).to_real_reg(), "x18".to_string()));
    regs.push((fp_reg().to_real_reg(), "fp".to_string()));
    regs.push((link_reg().to_real_reg(), "lr".to_string()));
    regs.push((zero_reg().to_real_reg(), "xzr".to_string()));
    regs.push((stack_reg().to_real_reg(), "xsp".to_string()));
    // FIXME JRS 2020Feb06: unfortunately this pushes the number of real regs
    // to 65, which is potentially inconvenient from a compiler performance
    // standpoint.  We could possibly drop back to 64 by "losing" a vector
    // register in future.

    RealRegUniverse {
        regs,
        allocable,
        allocable_by_class,
    }
}

/// An ALU operation. This can be paired with several instruction formats below (see `Inst`) in any
/// combination.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ALUOp {
    Add32,
    Add64,
    Sub32,
    Sub64,
    Orr32,
    Orr64,
    And32,
    And64,
    SubS32,
    SubS64,
}

/// Instruction formats.
#[derive(Clone, Debug)]
pub enum Inst {
    /// A no-op of zero size.
    Nop,

    /// A no-op that is one instruction large.
    Nop4,

    /// An ALU operation with two register sources and a register destination.
    AluRRR {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// An ALU operation with a register source and an immediate-12 source, and a register
    /// destination.
    AluRRImm12 {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        imm12: Imm12,
    },
    /// An ALU operation with a register source and an immediate-logic source, and a register destination.
    AluRRImmLogic {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        imml: ImmLogic,
    },
    /// An ALU operation with a register source and an immediate-shiftamt source, and a register destination.
    AluRRImmShift {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        immshift: ImmShift,
    },
    /// An ALU operation with two register sources, one of which can be shifted, and a register
    /// destination.
    AluRRRShift {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shiftop: ShiftOpAndAmt,
    },
    /// An ALU operation with two register sources, one of which can be {zero,sign}-extended and
    /// shifted, and a register destination.
    AluRRRExtend {
        alu_op: ALUOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        extendop: ExtendOp,
    },
    /// An unsigned (zero-extending) 8-bit load.
    ULoad8 { rd: Reg, mem: MemArg },
    /// A signed (sign-extending) 8-bit load.
    SLoad8 { rd: Reg, mem: MemArg },
    /// An unsigned (zero-extending) 16-bit load.
    ULoad16 { rd: Reg, mem: MemArg },
    /// A signed (sign-extending) 16-bit load.
    SLoad16 { rd: Reg, mem: MemArg },
    /// An unsigned (zero-extending) 32-bit load.
    ULoad32 { rd: Reg, mem: MemArg },
    /// A signed (sign-extending) 32-bit load.
    SLoad32 { rd: Reg, mem: MemArg },
    /// A 64-bit load.
    ULoad64 { rd: Reg, mem: MemArg },

    /// An 8-bit store.
    Store8 { rd: Reg, mem: MemArg },
    /// A 16-bit store.
    Store16 { rd: Reg, mem: MemArg },
    /// A 32-bit store.
    Store32 { rd: Reg, mem: MemArg },
    /// A 64-bit store.
    Store64 { rd: Reg, mem: MemArg },

    /// A store of a pair of registers.
    StoreP64 { rt: Reg, rt2: Reg, mem: PairMemArg },
    /// A load of a pair of registers.
    LoadP64 { rt: Reg, rt2: Reg, mem: PairMemArg },

    /// A MOV instruction. These are encoded as ORR's (AluRRR form) but we
    /// keep them separate at the `Inst` level for better pretty-printing
    /// and faster `is_move()` logic.
    Mov { rd: Reg, rm: Reg },

    /// A MOVZ with a 16-bit immediate.
    MovZ { rd: Reg, imm: MovZConst },

    /// A machine call instruction.
    Call { dest: FuncRef },
    /// A machine indirect-call instruction.
    CallInd { rn: Reg },

    // ---- branches (exactly one must appear at end of BB) ----
    /// A machine return instruction.
    Ret {},
    /// An unconditional branch.
    Jump { dest: BranchTarget },

    /// A conditional branch.
    CondBr {
        taken: BranchTarget,
        not_taken: BranchTarget,
        kind: CondBrKind,
    },

    /// Lowered conditional branch: contains the original instruction, and a
    /// flag indicating whether to invert the taken-condition or not. Only one
    /// BranchTarget is retained, and the other is implicitly the next
    /// instruction, given the final basic-block layout.
    CondBrLowered {
        target: BranchTarget,
        inverted: bool,
        kind: CondBrKind,
    },

    /// As for `CondBrLowered`, but represents a condbr/uncond-br sequence (two
    /// actual machine instructions). Needed when the final block layout implies
    /// that both arms of a conditional branch are not the fallthrough block.
    CondBrLoweredCompound {
        taken: BranchTarget,
        not_taken: BranchTarget,
        kind: CondBrKind,
    },
}

/// The kind of conditional branch: the common-case-optimized "reg-is-zero" /
/// "reg-is-nonzero" variants, or the generic one that tests the machine
/// condition codes.
#[derive(Clone, Copy, Debug)]
pub enum CondBrKind {
    /// Condition: given register is zero.
    Zero(Reg),
    /// Condition: given register is nonzero.
    NotZero(Reg),
    /// Condition: the given condition-code test is true.
    Cond(Cond),
}

impl CondBrKind {
    /// Return the inverted branch condition.
    pub fn invert(self) -> CondBrKind {
        match self {
            CondBrKind::Zero(reg) => CondBrKind::NotZero(reg),
            CondBrKind::NotZero(reg) => CondBrKind::Zero(reg),
            CondBrKind::Cond(c) => CondBrKind::Cond(c.invert()),
        }
    }
}

/// A shifted immediate value in 'imm12' format: supports 12 bits, shifted left by 0 or 12 places.
#[derive(Clone, Debug)]
pub struct Imm12 {
    /// The immediate bits.
    pub bits: usize,
    /// Whether the immediate bits are shifted left by 12 or not.
    pub shift12: bool,
}

impl Imm12 {
    /// Compute a Imm12 from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<Imm12> {
        if val == 0 {
            Some(Imm12 {
                bits: 0,
                shift12: false,
            })
        } else if val < 0xfff {
            Some(Imm12 {
                bits: val as usize,
                shift12: false,
            })
        } else if val < 0xfff_000 && (val & 0xfff == 0) {
            Some(Imm12 {
                bits: (val as usize) >> 12,
                shift12: true,
            })
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

    /// Bits for 12-bit "imm" field in e.g. AddI.
    pub fn imm_bits(&self) -> u16 {
        self.bits as u16
    }
}

/// An immediate for logical instructions.
#[derive(Clone, Debug)]
pub struct ImmLogic {
    /// `N` flag.
    pub N: bool,
    /// `S` field: element size and element bits.
    pub R: u8,
    /// `R` field: rotate amount.
    pub S: u8,
}

impl ImmLogic {
    /// Compute an ImmLogic from raw bits, if possible.
    pub fn maybe_from_u64(_val: u64) -> Option<ImmLogic> {
        // TODO: implement.
        None
    }

    /// Returns bits ready for encoding: (N:1, R:6, S:6)
    pub fn enc_bits(&self) -> u16 {
        ((self.N as u16) << 12) | ((self.R as u16) << 6) | (self.S as u16)
    }
}

/// An immediate for shift instructions.
#[derive(Clone, Debug)]
pub struct ImmShift {
    /// 6-bit shift amount.
    pub imm: u8,
}

impl ImmShift {
    /// Create an ImmShift from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<ImmShift> {
        if val > 0 && val < 64 {
            Some(ImmShift { imm: val as u8 })
        } else {
            None
        }
    }
}

/// A shift operator for a register or immediate.
#[derive(Clone, Copy, Debug)]
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

/// A shift operator with an amount, guaranteed to be within range.
#[derive(Clone, Debug)]
pub struct ShiftOpAndAmt {
    op: ShiftOp,
    shift: ShiftOpShiftImm,
}

/// A shift operator amount.
#[derive(Clone, Copy, Debug)]
pub struct ShiftOpShiftImm(u8);

impl ShiftOpShiftImm {
    /// Maximum shift for shifted-register operands.
    pub const MAX_SHIFT: u64 = 7;

    /// Create a new shiftop shift amount, if possible.
    pub fn maybe_from_shift(shift: u64) -> Option<ShiftOpShiftImm> {
        if shift <= Self::MAX_SHIFT {
            Some(ShiftOpShiftImm(shift as u8))
        } else {
            None
        }
    }
}

impl ShiftOpAndAmt {
    pub fn new(op: ShiftOp, shift: ShiftOpShiftImm) -> ShiftOpAndAmt {
        ShiftOpAndAmt { op, shift }
    }

    /// Get the shift op.
    pub fn op(&self) -> ShiftOp {
        self.op.clone()
    }

    /// Get the shift amount.
    pub fn amt(&self) -> ShiftOpShiftImm {
        self.shift
    }
}

/// An extend operator for a register.
#[derive(Clone, Copy, Debug)]
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
    Base(Reg),
    BaseSImm9(Reg, SImm9),
    BaseUImm12Scaled(Reg, UImm12Scaled),
    BasePlusReg(Reg, Reg),
    BasePlusRegScaled(Reg, Reg, Type),
    Label(MemLabel),
    PreIndexed(Reg, SImm9),
    PostIndexed(Reg, SImm9),
}

/// A memory argument to a load/store-pair.
#[derive(Clone, Debug)]
pub enum PairMemArg {
    Base(Reg),
    BaseSImm7(Reg, SImm7),
    PreIndexed(Reg, SImm7),
    PostIndexed(Reg, SImm7),
}

/// a 9-bit signed offset.
#[derive(Clone, Copy, Debug)]
pub struct SImm9 {
    bits: i16,
}

impl SImm9 {
    /// Create a signed 9-bit offset from a full-range value, if possible.
    pub fn maybe_from_i64(value: i64) -> Option<SImm9> {
        if value >= -256 && value <= 255 {
            Some(SImm9 { bits: value as i16 })
        } else {
            None
        }
    }
}

/// A 7-bit signed offset.
#[derive(Clone, Copy, Debug)]
pub struct SImm7 {
    bits: i16,
}

impl SImm7 {
    /// Create a signed 7-bit offset from a full-range value, if possible.
    pub fn maybe_from_i64(value: i64) -> Option<SImm7> {
        if value >= -128 && value <= 127 {
            Some(SImm7 { bits: value as i16 })
        } else {
            None
        }
    }
}

/// an unsigned, scaled 12-bit offset.
#[derive(Clone, Copy, Debug)]
pub struct UImm12Scaled {
    bits: u16,
    scale_ty: Type, // multiplied by the size of this type
}

impl UImm12Scaled {
    /// Create a UImm12Scaled from a raw offset and the known scale type, if possible.
    pub fn maybe_from_i64(value: i64, scale_ty: Type) -> Option<UImm12Scaled> {
        let scale = scale_ty.bytes() as i64;
        assert!((scale & (scale - 1)) == 0); // must be a power of 2.
        let limit = 4095 * scale;
        if value >= 0 && value <= limit && (value & (scale - 1)) == 0 {
            Some(UImm12Scaled {
                bits: (value / scale) as u16,
                scale_ty,
            })
        } else {
            None
        }
    }
}

/// A 16-bit immediate for a MOVZ instruction, with a {0,16,32,48}-bit shift.
#[derive(Clone, Copy, Debug)]
pub struct MovZConst {
    bits: u16,
    shift: u8, // shifted 16*shift bits to the left.
}

impl MovZConst {
    /// Construct a MovZConst from an arbitrary 64-bit constant if possible.
    pub fn maybe_from_u64(value: u64) -> Option<MovZConst> {
        let mask0 = 0x0000_0000_0000_ffffu64;
        let mask1 = 0x0000_0000_ffff_0000u64;
        let mask2 = 0x0000_ffff_0000_0000u64;
        let mask3 = 0xffff_0000_0000_0000u64;

        if value == (value & mask0) {
            return Some(MovZConst {
                bits: (value & mask0) as u16,
                shift: 0,
            });
        }
        if value == (value & mask1) {
            return Some(MovZConst {
                bits: ((value >> 16) & mask0) as u16,
                shift: 1,
            });
        }
        if value == (value & mask2) {
            return Some(MovZConst {
                bits: ((value >> 32) & mask0) as u16,
                shift: 2,
            });
        }
        if value == (value & mask3) {
            return Some(MovZConst {
                bits: ((value >> 48) & mask0) as u16,
                shift: 3,
            });
        }
        None
    }
}

impl MemArg {
    /// Memory reference using an address in a register.
    pub fn reg(reg: Reg) -> MemArg {
        MemArg::Base(reg)
    }

    /// Memory reference using an address in a register and an offset, if possible.
    pub fn reg_maybe_offset(reg: Reg, offset: i64, value_type: Type) -> Option<MemArg> {
        if offset == 0 {
            Some(MemArg::Base(reg))
        } else if let Some(simm9) = SImm9::maybe_from_i64(offset) {
            Some(MemArg::BaseSImm9(reg, simm9))
        } else if let Some(uimm12s) = UImm12Scaled::maybe_from_i64(offset, value_type) {
            Some(MemArg::BaseUImm12Scaled(reg, uimm12s))
        } else {
            None
        }
    }

    /// Memory reference using the sum of two registers as an address.
    pub fn reg_reg(reg1: Reg, reg2: Reg) -> MemArg {
        MemArg::BasePlusReg(reg1, reg2)
    }

    /// Memory reference to a label: a global function or value, or data in the constant pool.
    pub fn label(label: MemLabel) -> MemArg {
        MemArg::Label(label)
    }
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

/// Condition for conditional branches.
#[derive(Clone, Copy, Debug)]
pub enum Cond {
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

impl Cond {
    /// Return the inverted condition.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Nv,
            Cond::Nv => Cond::Al,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Cond::Eq => 0,
            Cond::Ne => 1,
            Cond::Hs => 2,
            Cond::Lo => 3,
            Cond::Mi => 4,
            Cond::Pl => 5,
            Cond::Vs => 6,
            Cond::Vc => 7,
            Cond::Hi => 8,
            Cond::Ls => 9,
            Cond::Ge => 10,
            Cond::Lt => 11,
            Cond::Gt => 12,
            Cond::Le => 13,
            Cond::Al => 14,
            Cond::Nv => 15,
        }
    }
}

/// A branch target. Either unresolved (basic-block index) or resolved (offset
/// from end of current instruction).
#[derive(Clone, Copy, Debug)]
pub enum BranchTarget {
    /// An unresolved reference to a BlockIndex, as passed into
    /// `lower_branch_group()`.
    Block(BlockIndex),
    /// A resolved reference to another instruction, after
    /// `Inst::with_block_offsets()`.
    ResolvedOffset(isize),
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

fn memarg_regs(memarg: &MemArg, used: &mut Set<Reg>, modified: &mut Set<Reg>) {
    match memarg {
        &MemArg::Base(reg) | &MemArg::BaseSImm9(reg, ..) | &MemArg::BaseUImm12Scaled(reg, ..) => {
            used.insert(reg);
        }
        &MemArg::BasePlusReg(r1, r2) | &MemArg::BasePlusRegScaled(r1, r2, ..) => {
            used.insert(r1);
            used.insert(r2);
        }
        &MemArg::Label(..) => {}
        &MemArg::PreIndexed(reg, ..) | &MemArg::PostIndexed(reg, ..) => {
            modified.insert(reg);
        }
    }
}

fn pairmemarg_regs(pairmemarg: &PairMemArg, used: &mut Set<Reg>, modified: &mut Set<Reg>) {
    match pairmemarg {
        &PairMemArg::Base(reg) | &PairMemArg::BaseSImm7(reg, ..) => {
            used.insert(reg);
        }
        &PairMemArg::PreIndexed(reg, ..) | &PairMemArg::PostIndexed(reg, ..) => {
            modified.insert(reg);
        }
    }
}

impl Inst {
    /// Create a move instruction.
    pub fn mov(to_reg: Reg, from_reg: Reg) -> Inst {
        Inst::Mov {
            rd: to_reg,
            rm: from_reg,
        }
    }
}

impl MachInst for Inst {
    fn get_regs(&self) -> InstRegUses {
        // One thing we need to enforce here is that if a register is in the
        // modified set, then it may not be in either the use or def sets.  On
        // this target we don't expect to have any registers used in a
        // 'modify' role, so we do the obvious thing at the end and assert
        // that the modify set is empty.
        let mut iru = InstRegUses::new();

        match self {
            &Inst::AluRRR { rd, rn, rm, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
                iru.used.insert(rm);
            }
            &Inst::AluRRImm12 { rd, rn, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
            }
            &Inst::AluRRImmLogic { rd, rn, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
            }
            &Inst::AluRRImmShift { rd, rn, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
            }
            &Inst::AluRRRShift { rd, rn, rm, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
                iru.used.insert(rm);
            }
            &Inst::AluRRRExtend { rd, rn, rm, .. } => {
                iru.defined.insert(rd);
                iru.used.insert(rn);
                iru.used.insert(rm);
            }
            &Inst::ULoad8 { rd, ref mem, .. }
            | &Inst::SLoad8 { rd, ref mem, .. }
            | &Inst::ULoad16 { rd, ref mem, .. }
            | &Inst::SLoad16 { rd, ref mem, .. }
            | &Inst::ULoad32 { rd, ref mem, .. }
            | &Inst::SLoad32 { rd, ref mem, .. }
            | &Inst::ULoad64 { rd, ref mem, .. } => {
                iru.defined.insert(rd);
                memarg_regs(mem, &mut iru.used, &mut iru.modified);
            }
            &Inst::Store8 { rd, ref mem, .. }
            | &Inst::Store16 { rd, ref mem, .. }
            | &Inst::Store32 { rd, ref mem, .. }
            | &Inst::Store64 { rd, ref mem, .. } => {
                iru.used.insert(rd);
                memarg_regs(mem, &mut iru.used, &mut iru.modified);
            }
            &Inst::StoreP64 {
                rt, rt2, ref mem, ..
            } => {
                iru.used.insert(rt);
                iru.used.insert(rt2);
                pairmemarg_regs(mem, &mut iru.used, &mut iru.modified);
            }
            &Inst::LoadP64 {
                rt, rt2, ref mem, ..
            } => {
                iru.defined.insert(rt);
                iru.defined.insert(rt2);
                pairmemarg_regs(mem, &mut iru.used, &mut iru.modified);
            }
            &Inst::Mov { rd, rm } => {
                iru.defined.insert(rd);
                iru.used.insert(rm);
            }
            &Inst::MovZ { rd, .. } => {
                iru.defined.insert(rd);
            }
            &Inst::Jump { .. } | &Inst::Call { .. } | &Inst::Ret { .. } => {}
            &Inst::CallInd { rn, .. } => {
                iru.used.insert(rn);
            }
            &Inst::CondBr { ref kind, .. }
            | &Inst::CondBrLowered { ref kind, .. }
            | &Inst::CondBrLoweredCompound { ref kind, .. } => match kind {
                CondBrKind::Zero(rt) | CondBrKind::NotZero(rt) => {
                    iru.used.insert(*rt);
                }
                CondBrKind::Cond(_) => {}
            },
            &Inst::Nop | Inst::Nop4 => {}
        }

        debug_assert!(iru.modified.is_empty());
        iru
    }

    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        fn map(m: &RegallocMap<VirtualReg, RealReg>, r: Reg) -> Reg {
            if r.is_virtual() {
                m.get(&r.to_virtual_reg()).cloned().unwrap().to_reg()
            } else {
                r
            }
        }

        fn map_mem(u: &RegallocMap<VirtualReg, RealReg>, mem: &MemArg) -> MemArg {
            // N.B.: we take only the pre-map here, but this is OK because the
            // only addressing modes that update registers (pre/post-increment on
            // ARM64) both read and write registers, so they are "mods" rather
            // than "defs", so must be the same in both the pre- and post-map.
            match mem {
                &MemArg::Base(reg) => MemArg::Base(map(u, reg)),
                &MemArg::BaseSImm9(reg, simm9) => MemArg::BaseSImm9(map(u, reg), simm9),
                &MemArg::BaseUImm12Scaled(reg, uimm12) => {
                    MemArg::BaseUImm12Scaled(map(u, reg), uimm12)
                }
                &MemArg::BasePlusReg(r1, r2) => MemArg::BasePlusReg(map(u, r1), map(u, r2)),
                &MemArg::BasePlusRegScaled(r1, r2, ty) => {
                    MemArg::BasePlusRegScaled(map(u, r1), map(u, r2), ty)
                }
                &MemArg::Label(ref l) => MemArg::Label(l.clone()),
                &MemArg::PreIndexed(r, simm9) => MemArg::PreIndexed(map(u, r), simm9),
                &MemArg::PostIndexed(r, simm9) => MemArg::PostIndexed(map(u, r), simm9),
            }
        }

        fn map_pairmem(u: &RegallocMap<VirtualReg, RealReg>, mem: &PairMemArg) -> PairMemArg {
            match mem {
                &PairMemArg::Base(reg) => PairMemArg::Base(map(u, reg)),
                &PairMemArg::BaseSImm7(reg, simm7) => PairMemArg::BaseSImm7(map(u, reg), simm7),
                &PairMemArg::PreIndexed(reg, simm7) => PairMemArg::PreIndexed(map(u, reg), simm7),
                &PairMemArg::PostIndexed(reg, simm7) => PairMemArg::PostIndexed(map(u, reg), simm7),
            }
        }

        fn map_br(u: &RegallocMap<VirtualReg, RealReg>, br: &CondBrKind) -> CondBrKind {
            match br {
                &CondBrKind::Zero(reg) => CondBrKind::Zero(map(u, reg)),
                &CondBrKind::NotZero(reg) => CondBrKind::NotZero(map(u, reg)),
                &CondBrKind::Cond(c) => CondBrKind::Cond(c),
            }
        }

        let u = pre_map; // For brevity below.
        let d = post_map;

        let newval = match self {
            &mut Inst::AluRRR { alu_op, rd, rn, rm } => Inst::AluRRR {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                rm: map(u, rm),
            },
            &mut Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                ref imm12,
            } => Inst::AluRRImm12 {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                imm12: imm12.clone(),
            },
            &mut Inst::AluRRImmLogic {
                alu_op,
                rd,
                rn,
                ref imml,
            } => Inst::AluRRImmLogic {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                imml: imml.clone(),
            },
            &mut Inst::AluRRImmShift {
                alu_op,
                rd,
                rn,
                ref immshift,
            } => Inst::AluRRImmShift {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                immshift: immshift.clone(),
            },
            &mut Inst::AluRRRShift {
                alu_op,
                rd,
                rn,
                rm,
                ref shiftop,
            } => Inst::AluRRRShift {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                rm: map(u, rm),
                shiftop: shiftop.clone(),
            },
            &mut Inst::AluRRRExtend {
                alu_op,
                rd,
                rn,
                rm,
                ref extendop,
            } => Inst::AluRRRExtend {
                alu_op,
                rd: map(d, rd),
                rn: map(u, rn),
                rm: map(u, rm),
                extendop: extendop.clone(),
            },
            &mut Inst::ULoad8 { rd, ref mem } => Inst::ULoad8 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::SLoad8 { rd, ref mem } => Inst::SLoad8 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::ULoad16 { rd, ref mem } => Inst::ULoad16 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::SLoad16 { rd, ref mem } => Inst::SLoad16 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::ULoad32 { rd, ref mem } => Inst::ULoad32 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::SLoad32 { rd, ref mem } => Inst::SLoad32 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::ULoad64 { rd, ref mem } => Inst::ULoad64 {
                rd: map(d, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::Store8 { rd, ref mem } => Inst::Store8 {
                rd: map(u, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::Store16 { rd, ref mem } => Inst::Store16 {
                rd: map(u, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::Store32 { rd, ref mem } => Inst::Store32 {
                rd: map(u, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::Store64 { rd, ref mem } => Inst::Store64 {
                rd: map(u, rd),
                mem: map_mem(u, mem),
            },
            &mut Inst::StoreP64 { rt, rt2, ref mem } => Inst::StoreP64 {
                rt: map(u, rt),
                rt2: map(u, rt2),
                mem: map_pairmem(u, mem),
            },
            &mut Inst::LoadP64 { rt, rt2, ref mem } => Inst::LoadP64 {
                rt: map(d, rt),
                rt2: map(d, rt2),
                mem: map_pairmem(u, mem),
            },
            &mut Inst::Mov { rd, rm } => Inst::Mov {
                rd: map(d, rd),
                rm: map(u, rm),
            },
            &mut Inst::MovZ { rd, ref imm } => Inst::MovZ {
                rd: map(d, rd),
                imm: imm.clone(),
            },
            &mut Inst::Jump { dest } => Inst::Jump { dest },
            &mut Inst::Call { dest } => Inst::Call { dest },
            &mut Inst::Ret {} => Inst::Ret {},
            &mut Inst::CallInd { rn } => Inst::CallInd { rn: map(u, rn) },
            &mut Inst::CondBr {
                taken,
                not_taken,
                kind,
            } => Inst::CondBr {
                taken,
                not_taken,
                kind: map_br(u, &kind),
            },
            &mut Inst::CondBrLowered {
                target,
                inverted,
                kind,
            } => Inst::CondBrLowered {
                target,
                inverted,
                kind: map_br(u, &kind),
            },
            &mut Inst::CondBrLoweredCompound {
                taken,
                not_taken,
                kind,
            } => Inst::CondBrLoweredCompound {
                taken,
                not_taken,
                kind: map_br(u, &kind),
            },
            &mut Inst::Nop => Inst::Nop,
            &mut Inst::Nop4 => Inst::Nop4,
        };
        *self = newval;
    }

    fn is_move(&self) -> Option<(Reg, Reg)> {
        match self {
            &Inst::Mov { rd, rm } => Some((rd, rm)),
            _ => None,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            &Inst::Ret {} => MachTerminator::Ret,
            &Inst::Jump { dest } => MachTerminator::Uncond(dest.as_block_index().unwrap()),
            &Inst::CondBr {
                taken, not_taken, ..
            } => MachTerminator::Cond(
                taken.as_block_index().unwrap(),
                not_taken.as_block_index().unwrap(),
            ),
            &Inst::CondBrLowered { .. } | &Inst::CondBrLoweredCompound { .. } => {
                panic!("is_term() called after lowering branches");
            }
            _ => MachTerminator::None,
        }
    }

    fn get_spillslot_size(rc: RegClass, ty: Type) -> u32 {
        // We allocate in terms of 8-byte slots.
        match (rc, ty) {
            (RegClass::I64, _) => 1,
            (RegClass::V128, F32) | (RegClass::V128, F64) => 1,
            (RegClass::V128, _) => 2,
            _ => panic!("Unexpected register class!"),
        }
    }

    fn gen_spill(_to_slot: SpillSlot, _from_reg: RealReg, _ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_reload(_to_reg: RealReg, _from_slot: SpillSlot, _ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_move(to_reg: Reg, from_reg: Reg) -> Inst {
        Inst::mov(to_reg, from_reg)
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        // We can't give a NOP (or any insn) < 4 bytes.
        assert!(preferred_size >= 4);
        Inst::Nop4
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
        Inst::Jump {
            dest: BranchTarget::Block(blockindex),
        }
    }

    fn with_block_rewrites(&mut self, block_target_map: &[BlockIndex]) {
        match self {
            &mut Inst::Jump { ref mut dest } => {
                dest.map(block_target_map);
            }
            &mut Inst::CondBr {
                ref mut taken,
                ref mut not_taken,
                ..
            } => {
                taken.map(block_target_map);
                not_taken.map(block_target_map);
            }
            &mut Inst::CondBrLowered { .. } | &mut Inst::CondBrLoweredCompound { .. } => {
                panic!("with_block_rewrites called after branch lowering!");
            }
            _ => {}
        }
    }

    fn with_fallthrough_block(&mut self, fallthrough: Option<BlockIndex>) {
        match self {
            &mut Inst::CondBr {
                taken,
                not_taken,
                kind,
            } => {
                if taken.as_block_index() == fallthrough {
                    *self = Inst::CondBrLowered {
                        target: not_taken,
                        inverted: true,
                        kind,
                    };
                } else if not_taken.as_block_index() == fallthrough {
                    *self = Inst::CondBrLowered {
                        target: taken,
                        inverted: false,
                        kind,
                    };
                } else {
                    // We need a compound sequence (condbr / uncond-br).
                    *self = Inst::CondBrLoweredCompound {
                        taken,
                        not_taken,
                        kind,
                    };
                }
            }
            &mut Inst::Jump { dest } => {
                if dest.as_block_index() == fallthrough {
                    *self = Inst::Nop;
                }
            }
            _ => {}
        }
    }

    fn with_block_offsets(&mut self, my_offset: CodeOffset, targets: &[CodeOffset]) {
        match self {
            &mut Inst::CondBrLowered { ref mut target, .. } => {
                target.lower(targets, my_offset);
            }
            &mut Inst::CondBrLoweredCompound {
                ref mut taken,
                ref mut not_taken,
                ..
            } => {
                taken.lower(targets, my_offset);
                not_taken.lower(targets, my_offset);
            }
            &mut Inst::Jump { ref mut dest } => {
                dest.lower(targets, my_offset);
            }
            _ => {}
        }
    }

    fn reg_universe() -> RealRegUniverse {
        create_reg_universe()
    }
}

impl BranchTarget {
    fn lower(&mut self, targets: &[CodeOffset], my_offset: CodeOffset) {
        match self {
            &mut BranchTarget::Block(bix) => {
                let bix = bix as usize;
                assert!(bix < targets.len());
                let block_offset_in_func = targets[bix];
                let branch_offset = (block_offset_in_func as isize) - (my_offset as isize);
                *self = BranchTarget::ResolvedOffset(branch_offset);
            }
            &mut BranchTarget::ResolvedOffset(..) => {}
        }
    }

    fn as_block_index(&self) -> Option<BlockIndex> {
        match self {
            &BranchTarget::Block(bix) => Some(bix),
            _ => None,
        }
    }

    fn as_offset_words(&self) -> Option<isize> {
        match self {
            &BranchTarget::ResolvedOffset(off) => Some(off >> 2),
            _ => None,
        }
    }

    fn as_off26(&self) -> Option<u32> {
        self.as_offset_words().and_then(|i| {
            if (i < (1 << 25)) && (i >= -(1 << 25)) {
                Some((i as u32) & ((1 << 26) - 1))
            } else {
                None
            }
        })
    }

    fn as_off19(&self) -> Option<u32> {
        self.as_offset_words().and_then(|i| {
            if (i < (1 << 18)) && (i >= -(1 << 18)) {
                Some((i as u32) & ((1 << 19) - 1))
            } else {
                None
            }
        })
    }

    fn map(&mut self, block_index_map: &[BlockIndex]) {
        match self {
            &mut BranchTarget::Block(ref mut bix) => {
                let n = block_index_map[*bix as usize];
                *bix = n;
            }
            _ => panic!("BranchTarget::map() called on already-lowered BranchTarget!"),
        }
    }
}

fn machreg_to_gpr(m: Reg) -> u32 {
    assert!(m.is_real());
    m.to_real_reg().get_hw_encoding() as u32
}

fn enc_arith_rrr(bits_31_21: u16, bits_15_10: u8, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    ((bits_31_21 as u32) << 21)
        | ((bits_15_10 as u32) << 10)
        | machreg_to_gpr(rd)
        | (machreg_to_gpr(rn) << 5)
        | (machreg_to_gpr(rm) << 16)
}

fn enc_arith_rr_imm12(bits_31_24: u8, immshift: u8, imm12: u16, rn: Reg, rd: Reg) -> u32 {
    ((bits_31_24 as u32) << 24)
        | ((immshift as u32) << 22)
        | ((imm12 as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

fn enc_arith_rr_imml(bits_31_23: u16, imm_bits: u16, rn: Reg, rd: Reg) -> u32 {
    ((bits_31_23 as u32) << 23)
        | ((imm_bits as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

fn enc_jump26(op_31_26: u32, off_26_0: u32) -> u32 {
    assert!(off_26_0 < (1 << 26));
    (op_31_26 << 26) | off_26_0
}

fn enc_cmpbr(op_31_24: u32, off_18_0: u32, reg: Reg) -> u32 {
    assert!(off_18_0 < (1 << 19));
    (op_31_24 << 24) | (off_18_0 << 5) | machreg_to_gpr(reg)
}

fn enc_cbr(op_31_24: u32, off_18_0: u32, op_4: u32, cond: u32) -> u32 {
    assert!(off_18_0 < (1 << 19));
    assert!(cond < (1 << 4));
    (op_31_24 << 24) | (off_18_0 << 5) | (op_4 << 4) | cond
}

const MOVE_WIDE_FIXED: u32 = 0x92800000;

#[repr(u32)]
enum MoveWideOpcode {
    MOVZ = 0b10,
}

// TODO: Pass imm_shift / imm16 in a struct?
fn enc_move_wide(opc: MoveWideOpcode, rd: Reg, imm_shift: u8, imm16: u16) -> u32 {
    assert!(imm_shift <= 0b11);
    MOVE_WIDE_FIXED
        | (opc as u32) << 29
        | (imm_shift as u32) << 21
        | (imm16 as u32) << 5
        | machreg_to_gpr(rd)
}

impl<CS: CodeSink> MachInstEmit<CS> for Inst {
    fn emit(&self, sink: &mut CS) {
        match self {
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let top11 = match alu_op {
                    ALUOp::Add32 => 0b00001011_000,
                    ALUOp::Add64 => 0b10001011_000,
                    ALUOp::Sub32 => 0b01001011_000,
                    ALUOp::Sub64 => 0b11001011_000,
                    ALUOp::Orr32 => 0b00101010_000,
                    ALUOp::Orr64 => 0b10101010_000,
                    ALUOp::SubS32 => 0b01101011_000,
                    ALUOp::SubS64 => 0b11101011_000,
                    _ => unimplemented!(),
                };
                sink.put4(enc_arith_rrr(top11, 0b000_000, rd, rn, rm));
            }
            &Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                ref imm12,
            } => {
                let top8 = match alu_op {
                    ALUOp::Add32 => 0b000_10001,
                    ALUOp::Add64 => 0b100_10001,
                    ALUOp::Sub32 => 0b010_10001,
                    ALUOp::Sub64 => 0b010_10001,
                    _ => unimplemented!(),
                };
                sink.put4(enc_arith_rr_imm12(
                    top8,
                    imm12.shift_bits(),
                    imm12.imm_bits(),
                    rn,
                    rd,
                ));
            }
            &Inst::AluRRImmLogic {
                alu_op,
                rd,
                rn,
                ref imml,
            } => {
                let top9 = match alu_op {
                    ALUOp::Orr32 => 0b001_100100,
                    ALUOp::Orr64 => 0b101_100100,
                    ALUOp::And32 => 0b000_100100,
                    ALUOp::And64 => 0b100_100100,
                    _ => unimplemented!(),
                };
                sink.put4(enc_arith_rr_imml(top9, imml.enc_bits(), rn, rd));
            }
            &Inst::AluRRImmShift { rd: _, rn: _, .. } => unimplemented!(),
            &Inst::AluRRRShift {
                rd: _,
                rn: _,
                rm: _,
                ..
            } => unimplemented!(),
            &Inst::AluRRRExtend {
                rd: _,
                rn: _,
                rm: _,
                ..
            } => unimplemented!(),
            &Inst::ULoad8 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::SLoad8 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::ULoad16 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::SLoad16 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::ULoad32 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::SLoad32 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::ULoad64 {
                rd: _,
                /*ref*/ mem: _,
                ..
            } => unimplemented!(),
            &Inst::Store8 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::Store16 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::Store32 {
                rd: _,
                /*ref*/ mem: _,
                ..
            }
            | &Inst::Store64 {
                rd: _,
                /*ref*/ mem: _,
                ..
            } => unimplemented!(),
            &Inst::StoreP64 { .. } => unimplemented!(),
            &Inst::LoadP64 { .. } => unimplemented!(),
            &Inst::Mov { rd, rm } => {
                // Encoded as ORR rd, rm, zero.
                sink.put4(enc_arith_rrr(0b10101010_000, 0b000_000, rd, zero_reg(), rm));
            }
            &Inst::MovZ { rd, imm } => {
                sink.put4(enc_move_wide(MoveWideOpcode::MOVZ, rd, imm.shift, imm.bits))
            }
            &Inst::Jump { ref dest } => {
                // TODO: differentiate between as_off26() returning `None` for
                // out-of-range vs. not-yet-finalized. The latter happens when we
                // do early (fake) emission for size computation.
                sink.put4(enc_jump26(0b000101, dest.as_off26().unwrap_or(0)));
            }
            &Inst::Ret {} => {
                sink.put4(0xd65f03c0);
            }
            &Inst::Call { .. } => unimplemented!(),
            &Inst::CallInd { rn: _, .. } => unimplemented!(),
            &Inst::CondBr { .. } => panic!("Unlowered CondBr during binemit!"),
            &Inst::CondBrLowered {
                target,
                inverted,
                kind,
            } => {
                let kind = if inverted { kind.invert() } else { kind };
                match kind {
                    CondBrKind::Zero(reg) => {
                        sink.put4(enc_cmpbr(0b1_011010_0, target.as_off19().unwrap_or(0), reg));
                    }
                    CondBrKind::NotZero(reg) => {
                        sink.put4(enc_cmpbr(0b1_011010_1, target.as_off19().unwrap_or(0), reg));
                    }
                    CondBrKind::Cond(c) => {
                        sink.put4(enc_cbr(
                            0b01010100,
                            target.as_off19().unwrap_or(0),
                            0b0,
                            c.bits(),
                        ));
                    }
                }
            }
            &Inst::CondBrLoweredCompound {
                taken,
                not_taken,
                kind,
            } => {
                // Conditional part first.
                match kind {
                    CondBrKind::Zero(reg) => {
                        sink.put4(enc_cmpbr(0b1_011010_0, taken.as_off19().unwrap_or(0), reg));
                    }
                    CondBrKind::NotZero(reg) => {
                        sink.put4(enc_cmpbr(0b1_011010_1, taken.as_off19().unwrap_or(0), reg));
                    }
                    CondBrKind::Cond(c) => {
                        sink.put4(enc_cbr(
                            0b01010100,
                            taken.as_off19().unwrap_or(0),
                            0b0,
                            c.bits(),
                        ));
                    }
                }
                // Unconditional part.
                sink.put4(enc_jump26(0b000101, not_taken.as_off26().unwrap_or(0)));
            }
            &Inst::Nop => {}
            &Inst::Nop4 => {
                sink.put4(0xd503201f);
            }
        }
    }
}
