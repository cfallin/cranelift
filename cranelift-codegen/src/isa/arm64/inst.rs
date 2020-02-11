//! This module defines arm64-specific machine instruction types.

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use crate::binemit::{CodeOffset, CodeSink};
use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::{B1, B128, B16, B32, B64, B8, F32, F64, I128, I16, I32, I64, I8};
use crate::ir::{FuncRef, GlobalValue, Type};
use crate::machinst::*;

use regalloc::Map as RegallocMap;
use regalloc::{InstRegUses, Set};
use regalloc::{
    RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, WritableReg, NUM_REG_CLASSES,
};

use alloc::vec::Vec;
use smallvec::SmallVec;
use std::mem;
use std::string::{String, ToString};

//=============================================================================
// Registers, the Universe thereof, and printing

#[rustfmt::skip]
const XREG_INDICES: [u8; 31] = [
    // X0 - X7
    32, 33, 34, 35, 36, 37, 38, 39,
    // X8 - X14
    40, 41, 42, 43, 44, 45, 46,
    // X15
    59,
    // X16, X17
    47, 48,
    // X18
    60,
    // X19 - X28
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    // X29
    61,
    // X30
    62,
];

const ZERO_REG_INDEX: u8 = 63;

const SP_REG_INDEX: u8 = 64;

/// Get a reference to an X-register (integer register).
pub fn xreg(num: u8) -> Reg {
    assert!(num < 31);
    Reg::new_real(
        RegClass::I64,
        /* enc = */ num,
        /* index = */ XREG_INDICES[num as usize],
    )
}

/// Get a writable reference to an X-register.
pub fn writable_xreg(num: u8) -> WritableReg<Reg> {
    WritableReg::from_reg(xreg(num))
}

/// Get a reference to a V-register (vector/FP register).
pub fn vreg(num: u8) -> Reg {
    assert!(num < 32);
    Reg::new_real(RegClass::V128, /* enc = */ num, /* index = */ num)
}

/// Get a writable reference to a V-register.
pub fn writable_vreg(num: u8) -> WritableReg<Reg> {
    WritableReg::from_reg(vreg(num))
}

/// Get a reference to the zero-register.
pub fn zero_reg() -> Reg {
    // This should be the same as what xreg(31) returns, except that
    // we use the special index into the register index space.
    Reg::new_real(
        RegClass::I64,
        /* enc = */ 31,
        /* index = */ ZERO_REG_INDEX,
    )
}

/// Get a writable reference to the zero-register (this discards a result).
pub fn writable_zero_reg() -> WritableReg<Reg> {
    WritableReg::from_reg(zero_reg())
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
        /* index = */ SP_REG_INDEX,
    )
}

/// Get a writable reference to the stack-pointer register.
pub fn writable_stack_reg() -> WritableReg<Reg> {
    WritableReg::from_reg(stack_reg())
}

/// Get a reference to the link register (x30).
pub fn link_reg() -> Reg {
    xreg(30)
}

/// Get a writable reference to the link register.
pub fn writable_link_reg() -> WritableReg<Reg> {
    WritableReg::from_reg(link_reg())
}

/// Get a reference to the frame pointer (x29).
pub fn fp_reg() -> Reg {
    xreg(29)
}

/// Get a writable reference to the frame pointer.
pub fn writable_fp_reg() -> WritableReg<Reg> {
    WritableReg::from_reg(fp_reg())
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

/// Get a writable reference to the spilltmp reg.
pub fn writable_spilltmp_reg() -> WritableReg<Reg> {
    WritableReg::from_reg(spilltmp_reg())
}

/// Create the register universe for ARM64.
pub fn create_reg_universe() -> RealRegUniverse {
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

    // Add the X registers. N.B.: the order here must match the order implied
    // by XREG_INDICES, ZERO_REG_INDEX, and SP_REG_INDEX above.

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
    regs.push((stack_reg().to_real_reg(), "sp".to_string()));
    // FIXME JRS 2020Feb06: unfortunately this pushes the number of real regs
    // to 65, which is potentially inconvenient from a compiler performance
    // standpoint.  We could possibly drop back to 64 by "losing" a vector
    // register in future.

    // Assert sanity: the indices in the register structs must match their
    // actual indices in the array.
    for (i, reg) in regs.iter().enumerate() {
        assert_eq!(i, reg.0.get_index());
    }

    RealRegUniverse {
        regs,
        allocable,
        allocable_by_class,
    }
}

// If |ireg| denotes an I64-classed reg, make a best-effort attempt to show
// its name at the 32-bit size.
fn show_ireg_sized(reg: Reg, mb_rru: Option<&RealRegUniverse>, is32: bool) -> String {
    let mut s = reg.show_rru(mb_rru);
    if reg.get_class() != RegClass::I64 || !is32 {
        // We can't do any better.
        return s;
    }

    if reg.is_real() {
        // Change (eg) "x42" into "w42" as appropriate
        if reg.get_class() == RegClass::I64 && is32 && s.starts_with("x") {
            s = "w".to_string() + &s[1..];
        }
    } else {
        // Add a "w" suffix to RegClass::I64 vregs used in a 32-bit role
        if reg.get_class() == RegClass::I64 && is32 {
            s = s + &"w";
        }
    }
    s
}

//=============================================================================
// Instruction sub-components (immediates and offsets): definitions

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

    /// Create a zero immediate of this format.
    pub fn zero() -> SImm7 {
        SImm7 { bits: 0 }
    }

    /// Bits for encoding.
    pub fn bits(&self) -> u32 {
        (self.bits as u32) & 0x7f
    }
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

    // Create a zero immediate of this format.
    pub fn zero() -> SImm9 {
        SImm9 { bits: 0 }
    }

    /// Bits for encoding.
    pub fn bits(&self) -> u32 {
        (self.bits as u32) & 0x1ff
    }
}

/// an unsigned, scaled 12-bit offset.
#[derive(Clone, Copy, Debug)]
pub struct UImm12Scaled {
    bits: u16,
    scale_ty: Type, // multiplied by the size of this type
}

impl UImm12Scaled {
    /// Create a UImm12Scaled from a raw offset and the known scale type, if
    /// possible.
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

    /// Encoded bits.
    pub fn bits(&self) -> u32 {
        (self.bits as u32) & 0xfff
    }
}

/// A shifted immediate value in 'imm12' format: supports 12 bits, shifted
/// left by 0 or 12 places.
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

    /// Returns the value that this immediate represents.
    pub fn value(&self) -> u64 {
        unimplemented!()
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

/// A 16-bit immediate for a MOVZ instruction, with a {0,16,32,48}-bit shift.
#[derive(Clone, Copy, Debug)]
pub struct MoveWideConst {
    bits: u16,
    shift: u8, // shifted 16*shift bits to the left.
}

impl MoveWideConst {
    /// Construct a MoveWideConst from an arbitrary 64-bit constant if possible.
    pub fn maybe_from_u64(value: u64) -> Option<MoveWideConst> {
        let mask0 = 0x0000_0000_0000_ffffu64;
        let mask1 = 0x0000_0000_ffff_0000u64;
        let mask2 = 0x0000_ffff_0000_0000u64;
        let mask3 = 0xffff_0000_0000_0000u64;

        if value == (value & mask0) {
            return Some(MoveWideConst {
                bits: (value & mask0) as u16,
                shift: 0,
            });
        }
        if value == (value & mask1) {
            return Some(MoveWideConst {
                bits: ((value >> 16) & mask0) as u16,
                shift: 1,
            });
        }
        if value == (value & mask2) {
            return Some(MoveWideConst {
                bits: ((value >> 32) & mask0) as u16,
                shift: 2,
            });
        }
        if value == (value & mask3) {
            return Some(MoveWideConst {
                bits: ((value >> 48) & mask0) as u16,
                shift: 3,
            });
        }
        None
    }

    /// Returns the value that this constant represents.
    pub fn value(&self) -> u64 {
        (self.bits as u64) << (16 * self.shift)
    }
}

//=============================================================================
// Instruction sub-components (shifted and extended operands): definitions

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

    /// Return the shift amount.
    pub fn value(&self) -> u8 {
        self.0
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

//=============================================================================
// Instruction sub-components (memory addresses): definitions

/// A reference to some memory address.
#[derive(Clone, Debug)]
pub enum MemLabel {
    /// A value in a constant pool, already emitted.
    ConstantPool(ConstantOffset),
    /// A value in a constant pool, to be emitted during binemit.
    ConstantData(ConstantData),
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
    PreIndexed(WritableReg<Reg>, SImm9),
    PostIndexed(WritableReg<Reg>, SImm9),
    /// Offset from the frame pointer.
    StackOffset(i64),
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

/// A memory argument to a load/store-pair.
#[derive(Clone, Debug)]
pub enum PairMemArg {
    Base(Reg),
    BaseSImm7(Reg, SImm7),
    PreIndexed(WritableReg<Reg>, SImm7),
    PostIndexed(WritableReg<Reg>, SImm7),
}

//=============================================================================
// Instruction sub-components (conditions, branches and branch targets):
// definitions

/// Condition for conditional branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A branch target. Either unresolved (basic-block index) or resolved (offset
/// from end of current instruction).
#[derive(Clone, Copy, Debug)]
pub enum BranchTarget {
    /// An unresolved reference to a BlockIndex, as passed into
    /// `lower_branch_group()`.
    Block(BlockIndex),
    /// A resolved reference to another instruction, after
    /// `Inst::with_block_offsets()`.
    ResolvedOffset(BlockIndex, isize),
}

impl BranchTarget {
    fn lower(&mut self, targets: &[CodeOffset], my_offset: CodeOffset) {
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

    fn as_block_index(&self) -> Option<BlockIndex> {
        match self {
            &BranchTarget::Block(bix) => Some(bix),
            _ => None,
        }
    }

    fn as_offset_words(&self) -> Option<isize> {
        match self {
            &BranchTarget::ResolvedOffset(_, off) => Some(off >> 2),
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

//=============================================================================
// Instructions (top level): definition

/// An ALU operation. This can be paired with several instruction formats
/// below (see `Inst`) in any combination.
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
        rd: WritableReg<Reg>,
        rn: Reg,
        rm: Reg,
    },
    /// An ALU operation with a register source and an immediate-12 source, and a register
    /// destination.
    AluRRImm12 {
        alu_op: ALUOp,
        rd: WritableReg<Reg>,
        rn: Reg,
        imm12: Imm12,
    },
    /// An ALU operation with a register source and an immediate-logic source, and a register destination.
    AluRRImmLogic {
        alu_op: ALUOp,
        rd: WritableReg<Reg>,
        rn: Reg,
        imml: ImmLogic,
    },
    /// An ALU operation with a register source and an immediate-shiftamt source, and a register destination.
    AluRRImmShift {
        alu_op: ALUOp,
        rd: WritableReg<Reg>,
        rn: Reg,
        immshift: ImmShift,
    },
    /// An ALU operation with two register sources, one of which can be shifted, and a register
    /// destination.
    AluRRRShift {
        alu_op: ALUOp,
        rd: WritableReg<Reg>,
        rn: Reg,
        rm: Reg,
        shiftop: ShiftOpAndAmt,
    },
    /// An ALU operation with two register sources, one of which can be {zero,sign}-extended and
    /// shifted, and a register destination.
    AluRRRExtend {
        alu_op: ALUOp,
        rd: WritableReg<Reg>,
        rn: Reg,
        rm: Reg,
        extendop: ExtendOp,
    },
    /// An unsigned (zero-extending) 8-bit load.
    ULoad8 { rd: WritableReg<Reg>, mem: MemArg },
    /// A signed (sign-extending) 8-bit load.
    SLoad8 { rd: WritableReg<Reg>, mem: MemArg },
    /// An unsigned (zero-extending) 16-bit load.
    ULoad16 { rd: WritableReg<Reg>, mem: MemArg },
    /// A signed (sign-extending) 16-bit load.
    SLoad16 { rd: WritableReg<Reg>, mem: MemArg },
    /// An unsigned (zero-extending) 32-bit load.
    ULoad32 { rd: WritableReg<Reg>, mem: MemArg },
    /// A signed (sign-extending) 32-bit load.
    SLoad32 { rd: WritableReg<Reg>, mem: MemArg },
    /// A 64-bit load.
    ULoad64 { rd: WritableReg<Reg>, mem: MemArg },

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
    LoadP64 {
        rt: WritableReg<Reg>,
        rt2: WritableReg<Reg>,
        mem: PairMemArg,
    },

    /// A MOV instruction. These are encoded as ORR's (AluRRR form) but we
    /// keep them separate at the `Inst` level for better pretty-printing
    /// and faster `is_move()` logic.
    Mov { rd: WritableReg<Reg>, rm: Reg },

    /// A MOVZ with a 16-bit immediate.
    MovZ {
        rd: WritableReg<Reg>,
        imm: MoveWideConst,
    },

    /// A MOVN with a 16-bit immediate.
    MovN {
        rd: WritableReg<Reg>,
        imm: MoveWideConst,
    },

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

impl Inst {
    /// Create a move instruction.
    pub fn mov(to_reg: WritableReg<Reg>, from_reg: Reg) -> Inst {
        Inst::Mov {
            rd: to_reg,
            rm: from_reg,
        }
    }
}

//=============================================================================
// Instructions: get_regs

fn memarg_regs(memarg: &MemArg, used: &mut Set<Reg>, modified: &mut Set<WritableReg<Reg>>) {
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
        &MemArg::StackOffset(..) => {
            used.insert(fp_reg());
        }
    }
}

fn pairmemarg_regs(
    pairmemarg: &PairMemArg,
    used: &mut Set<Reg>,
    modified: &mut Set<WritableReg<Reg>>,
) {
    match pairmemarg {
        &PairMemArg::Base(reg) | &PairMemArg::BaseSImm7(reg, ..) => {
            used.insert(reg);
        }
        &PairMemArg::PreIndexed(reg, ..) | &PairMemArg::PostIndexed(reg, ..) => {
            modified.insert(reg);
        }
    }
}

fn arm64_get_regs(inst: &Inst) -> InstRegUses {
    // One thing we need to enforce here is that if a register is in the
    // modified set, then it may not be in either the use or def sets.  On
    // this target we don't expect to have any registers used in a 'modify'
    // role, so we do the obvious thing at the end and assert that the modify
    // set is empty.
    let mut iru = InstRegUses::new();

    match inst {
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
        &Inst::MovZ { rd, .. } | &Inst::MovN { rd, .. } => {
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

//=============================================================================
// Instructions: map_regs

fn arm64_map_regs(
    inst: &mut Inst,
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

    fn map_wr(m: &RegallocMap<VirtualReg, RealReg>, r: WritableReg<Reg>) -> WritableReg<Reg> {
        WritableReg::from_reg(map(m, r.to_reg()))
    }

    fn map_mem(u: &RegallocMap<VirtualReg, RealReg>, mem: &MemArg) -> MemArg {
        // N.B.: we take only the pre-map here, but this is OK because the
        // only addressing modes that update registers (pre/post-increment on
        // ARM64) both read and write registers, so they are "mods" rather
        // than "defs", so must be the same in both the pre- and post-map.
        match mem {
            &MemArg::Base(reg) => MemArg::Base(map(u, reg)),
            &MemArg::BaseSImm9(reg, simm9) => MemArg::BaseSImm9(map(u, reg), simm9),
            &MemArg::BaseUImm12Scaled(reg, uimm12) => MemArg::BaseUImm12Scaled(map(u, reg), uimm12),
            &MemArg::BasePlusReg(r1, r2) => MemArg::BasePlusReg(map(u, r1), map(u, r2)),
            &MemArg::BasePlusRegScaled(r1, r2, ty) => {
                MemArg::BasePlusRegScaled(map(u, r1), map(u, r2), ty)
            }
            &MemArg::Label(ref l) => MemArg::Label(l.clone()),
            &MemArg::PreIndexed(r, simm9) => MemArg::PreIndexed(map_wr(u, r), simm9),
            &MemArg::PostIndexed(r, simm9) => MemArg::PostIndexed(map_wr(u, r), simm9),
            &MemArg::StackOffset(off) => MemArg::StackOffset(off),
        }
    }

    fn map_pairmem(u: &RegallocMap<VirtualReg, RealReg>, mem: &PairMemArg) -> PairMemArg {
        match mem {
            &PairMemArg::Base(reg) => PairMemArg::Base(map(u, reg)),
            &PairMemArg::BaseSImm7(reg, simm7) => PairMemArg::BaseSImm7(map(u, reg), simm7),
            &PairMemArg::PreIndexed(reg, simm7) => PairMemArg::PreIndexed(map_wr(u, reg), simm7),
            &PairMemArg::PostIndexed(reg, simm7) => PairMemArg::PostIndexed(map_wr(u, reg), simm7),
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

    let newval = match inst {
        &mut Inst::AluRRR { alu_op, rd, rn, rm } => Inst::AluRRR {
            alu_op,
            rd: map_wr(d, rd),
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
            rd: map_wr(d, rd),
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
            rd: map_wr(d, rd),
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
            rd: map_wr(d, rd),
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
            rd: map_wr(d, rd),
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
            rd: map_wr(d, rd),
            rn: map(u, rn),
            rm: map(u, rm),
            extendop: extendop.clone(),
        },
        &mut Inst::ULoad8 { rd, ref mem } => Inst::ULoad8 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::SLoad8 { rd, ref mem } => Inst::SLoad8 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::ULoad16 { rd, ref mem } => Inst::ULoad16 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::SLoad16 { rd, ref mem } => Inst::SLoad16 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::ULoad32 { rd, ref mem } => Inst::ULoad32 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::SLoad32 { rd, ref mem } => Inst::SLoad32 {
            rd: map_wr(d, rd),
            mem: map_mem(u, mem),
        },
        &mut Inst::ULoad64 { rd, ref mem } => Inst::ULoad64 {
            rd: map_wr(d, rd),
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
            rt: map_wr(d, rt),
            rt2: map_wr(d, rt2),
            mem: map_pairmem(u, mem),
        },
        &mut Inst::Mov { rd, rm } => Inst::Mov {
            rd: map_wr(d, rd),
            rm: map(u, rm),
        },
        &mut Inst::MovZ { rd, ref imm } => Inst::MovZ {
            rd: map_wr(d, rd),
            imm: imm.clone(),
        },
        &mut Inst::MovN { rd, ref imm } => Inst::MovN {
            rd: map_wr(d, rd),
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
    *inst = newval;
}

//============================================================================
// Memory addressing mode finalization: convert "special" modes (e.g.,
// generic arbitrary stack offset) into real addressing modes, possibly by
// emitting some helper instructions that come immediately before the use
// of this amod.

fn mem_finalize(mem: &MemArg) -> (Vec<Inst>, MemArg) {
    match mem {
        &MemArg::StackOffset(fp_offset) => {
            if let Some(simm9) = SImm9::maybe_from_i64(fp_offset) {
                let mem = MemArg::BaseSImm9(fp_reg(), simm9);
                (vec![], mem)
            } else {
                let tmp = writable_spilltmp_reg();
                let const_data = u64_constant(fp_offset as u64);
                let const_inst = Inst::ULoad64 {
                    rd: tmp,
                    mem: MemArg::label(MemLabel::ConstantData(const_data)),
                };
                let add_inst = Inst::AluRRR {
                    alu_op: ALUOp::Add64,
                    rd: tmp,
                    rn: tmp.to_reg(),
                    rm: fp_reg(),
                };
                (vec![const_inst, add_inst], MemArg::Base(tmp.to_reg()))
            }
        }
        _ => (vec![], mem.clone()),
    }
}

//=============================================================================
// Instructions and subcomponents: emission

fn machreg_to_gpr(m: Reg) -> u32 {
    assert!(m.is_real());
    m.to_real_reg().get_hw_encoding() as u32
}

fn enc_arith_rrr(bits_31_21: u16, bits_15_10: u8, rd: WritableReg<Reg>, rn: Reg, rm: Reg) -> u32 {
    ((bits_31_21 as u32) << 21)
        | ((bits_15_10 as u32) << 10)
        | machreg_to_gpr(rd.to_reg())
        | (machreg_to_gpr(rn) << 5)
        | (machreg_to_gpr(rm) << 16)
}

fn enc_arith_rr_imm12(
    bits_31_24: u8,
    immshift: u8,
    imm12: u16,
    rn: Reg,
    rd: WritableReg<Reg>,
) -> u32 {
    ((bits_31_24 as u32) << 24)
        | ((immshift as u32) << 22)
        | ((imm12 as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd.to_reg())
}

fn enc_arith_rr_imml(bits_31_23: u16, imm_bits: u16, rn: Reg, rd: WritableReg<Reg>) -> u32 {
    ((bits_31_23 as u32) << 23)
        | ((imm_bits as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd.to_reg())
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
    MOVN = 0b00,
    MOVZ = 0b10,
}

fn enc_move_wide(op: MoveWideOpcode, rd: WritableReg<Reg>, imm: MoveWideConst) -> u32 {
    assert!(imm.shift <= 0b11);
    MOVE_WIDE_FIXED
        | (op as u32) << 29
        | (imm.shift as u32) << 21
        | (imm.bits as u32) << 5
        | machreg_to_gpr(rd.to_reg())
}

fn enc_ldst_pair(op_31_22: u32, simm7: SImm7, rn: Reg, rt: Reg, rt2: Reg) -> u32 {
    (op_31_22 << 22)
        | (simm7.bits() << 15)
        | (machreg_to_gpr(rt2) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rt)
}

fn enc_ldst_simm9(op_31_22: u32, simm9: SImm9, op_11_10: u32, rn: Reg, rd: Reg) -> u32 {
    (op_31_22 << 22)
        | (simm9.bits() << 12)
        | (op_11_10 << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

fn enc_ldst_uimm12(op_31_22: u32, uimm12: UImm12Scaled, rn: Reg, rd: Reg) -> u32 {
    (op_31_22 << 22) | (uimm12.bits() << 10) | (machreg_to_gpr(rn) << 5) | machreg_to_gpr(rd)
}

fn enc_ldst_reg(op_31_22: u32, rn: Reg, rm: Reg, s_bit: bool, rd: Reg) -> u32 {
    let s_bit = if s_bit { 1 } else { 0 };
    (op_31_22 << 22)
        | (1 << 21)
        | (machreg_to_gpr(rm) << 16)
        | (0b011 << 13)
        | (s_bit << 12)
        | (0b10 << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

fn enc_ldst_imm19(op_31_24: u32, imm19: u32, rd: Reg) -> u32 {
    (op_31_24 << 24) | (imm19 << 5) | machreg_to_gpr(rd)
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
                    ALUOp::And32 => 0b00001010_000,
                    ALUOp::And64 => 0b10001010_000,
                    ALUOp::SubS32 => 0b01101011_000,
                    ALUOp::SubS64 => 0b11101011_000,
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
                    ALUOp::Sub64 => 0b110_10001,
                    ALUOp::SubS32 => 0b011_10001,
                    ALUOp::SubS64 => 0b111_10001,
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

            &Inst::ULoad8 { rd, ref mem }
            | &Inst::SLoad8 { rd, ref mem }
            | &Inst::ULoad16 { rd, ref mem }
            | &Inst::SLoad16 { rd, ref mem }
            | &Inst::ULoad32 { rd, ref mem }
            | &Inst::SLoad32 { rd, ref mem }
            | &Inst::ULoad64 { rd, ref mem } => {
                let (mem_insts, mem) = mem_finalize(mem);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink);
                }

                // ldst encoding helpers take Reg, not WritableReg.
                let rd = rd.to_reg();

                // This is the base opcode (top 10 bits) for the "unscaled
                // immediate" form (BaseSImm9). Other addressing modes will OR in
                // other values for bits 24/25 (bits 1/2 of this constant).
                let op = match self {
                    &Inst::ULoad8 { .. } => 0b0011100001,
                    &Inst::SLoad8 { .. } => 0b0011100010,
                    &Inst::ULoad16 { .. } => 0b0111100001,
                    &Inst::SLoad16 { .. } => 0b0111100010,
                    &Inst::ULoad32 { .. } => 0b1011100001,
                    &Inst::SLoad32 { .. } => 0b1011100010,
                    &Inst::ULoad64 { .. } => 0b1111100001,
                    _ => unreachable!(),
                };
                match &mem {
                    &MemArg::Base(reg) => {
                        sink.put4(enc_ldst_simm9(op, SImm9::zero(), 0b00, reg, rd));
                    }
                    &MemArg::BaseSImm9(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b00, reg, rd));
                    }
                    &MemArg::BaseUImm12Scaled(reg, uimm12scaled) => {
                        sink.put4(enc_ldst_uimm12(op | 0b101, uimm12scaled, reg, rd));
                    }
                    &MemArg::BasePlusReg(r1, r2) => {
                        sink.put4(enc_ldst_reg(op | 0b01, r1, r2, /* S = */ false, rd));
                    }
                    &MemArg::BasePlusRegScaled(r1, r2, _ty) => {
                        sink.put4(enc_ldst_reg(op | 0b01, r1, r2, /* S = */ true, rd));
                    }
                    &MemArg::Label(ref label) => {
                        let offset = match label {
                            &MemLabel::ConstantPool(off) => off,
                            &MemLabel::ConstantData(..) => {
                                // Should only happen when computing size --
                                // ConstantData refs are converted to
                                // ConstantPool refs once the data itself is
                                // collected and allocated into the constant
                                // pool.
                                0
                            }
                        } / 4;
                        assert!(offset < (1 << 19));
                        match self {
                            &Inst::ULoad32 { .. } => {
                                sink.put4(enc_ldst_imm19(0b00011000, offset, rd));
                            }
                            &Inst::SLoad32 { .. } => {
                                sink.put4(enc_ldst_imm19(0b10011000, offset, rd));
                            }
                            &Inst::ULoad64 { .. } => {
                                sink.put4(enc_ldst_imm19(0b01011000, offset, rd));
                            }
                            _ => panic!("Unspported size for LDR from constant pool!"),
                        }
                    }
                    &MemArg::PreIndexed(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b11, reg.to_reg(), rd));
                    }
                    &MemArg::PostIndexed(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b01, reg.to_reg(), rd));
                    }
                    // Eliminated by `mem_finalize()` above.
                    &MemArg::StackOffset(..) => panic!("Should not see StackOffset here!"),
                }
            }

            &Inst::Store8 { rd, ref mem }
            | &Inst::Store16 { rd, ref mem }
            | &Inst::Store32 { rd, ref mem }
            | &Inst::Store64 { rd, ref mem } => {
                let (mem_insts, mem) = mem_finalize(mem);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink);
                }

                let op = match self {
                    &Inst::Store8 { .. } => 0b0011100000,
                    &Inst::Store16 { .. } => 0b0111100000,
                    &Inst::Store32 { .. } => 0b1011100000,
                    &Inst::Store64 { .. } => 0b1111100000,
                    _ => unreachable!(),
                };
                match &mem {
                    &MemArg::Base(reg) => {
                        sink.put4(enc_ldst_simm9(op, SImm9::zero(), 0b00, reg, rd));
                    }
                    &MemArg::BaseSImm9(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b00, reg, rd));
                    }
                    &MemArg::BaseUImm12Scaled(reg, uimm12scaled) => {
                        sink.put4(enc_ldst_uimm12(op | 0b100, uimm12scaled, reg, rd));
                    }
                    &MemArg::BasePlusReg(r1, r2) => {
                        sink.put4(enc_ldst_reg(op, r1, r2, /* S = */ false, rd));
                    }
                    &MemArg::BasePlusRegScaled(r1, r2, _ty) => {
                        sink.put4(enc_ldst_reg(op, r1, r2, /* S = */ true, rd));
                    }
                    &MemArg::Label(..) => {
                        panic!("Store to a constant-pool entry not allowed!");
                    }
                    &MemArg::PreIndexed(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b11, reg.to_reg(), rd));
                    }
                    &MemArg::PostIndexed(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b01, reg.to_reg(), rd));
                    }
                    // Eliminated by `mem_finalize()` above.
                    &MemArg::StackOffset(..) => panic!("Should not see StackOffset here!"),
                }
            }

            &Inst::StoreP64 { rt, rt2, ref mem } => match mem {
                &PairMemArg::Base(reg) => {
                    sink.put4(enc_ldst_pair(0b1010100100, SImm7::zero(), reg, rt, rt2));
                }
                &PairMemArg::BaseSImm7(reg, simm7) => {
                    sink.put4(enc_ldst_pair(0b1010100100, simm7, reg, rt, rt2));
                }
                &PairMemArg::PreIndexed(reg, simm7) => {
                    sink.put4(enc_ldst_pair(0b1010100110, simm7, reg.to_reg(), rt, rt2));
                }
                &PairMemArg::PostIndexed(reg, simm7) => {
                    sink.put4(enc_ldst_pair(0b1010100010, simm7, reg.to_reg(), rt, rt2));
                }
            },
            &Inst::LoadP64 { rt, rt2, ref mem } => {
                let rt = rt.to_reg();
                let rt2 = rt2.to_reg();
                match mem {
                    &PairMemArg::Base(reg) => {
                        sink.put4(enc_ldst_pair(0b1010100101, SImm7::zero(), reg, rt, rt2));
                    }
                    &PairMemArg::BaseSImm7(reg, simm7) => {
                        sink.put4(enc_ldst_pair(0b1010100101, simm7, reg, rt, rt2));
                    }
                    &PairMemArg::PreIndexed(reg, simm7) => {
                        sink.put4(enc_ldst_pair(0b1010100111, simm7, reg.to_reg(), rt, rt2));
                    }
                    &PairMemArg::PostIndexed(reg, simm7) => {
                        sink.put4(enc_ldst_pair(0b1010100011, simm7, reg.to_reg(), rt, rt2));
                    }
                }
            }
            &Inst::Mov { rd, rm } => {
                // Encoded as ORR rd, rm, zero.
                sink.put4(enc_arith_rrr(0b10101010_000, 0b000_000, rd, zero_reg(), rm));
            }
            &Inst::MovZ { rd, imm } => sink.put4(enc_move_wide(MoveWideOpcode::MOVZ, rd, imm)),
            &Inst::MovN { rd, imm } => sink.put4(enc_move_wide(MoveWideOpcode::MOVN, rd, imm)),
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

//=============================================================================
// Instructions: misc functions and external interface

/// Helper: get a ConstantData from a u32.
// Currently unused
//pub fn u32_constant(bits: u32) -> ConstantData {
//    let data = [
//        (bits & 0xff) as u8,
//        ((bits >> 8) & 0xff) as u8,
//        ((bits >> 16) & 0xff) as u8,
//        ((bits >> 24) & 0xff) as u8,
//    ];
//    ConstantData::from(&data[..])
//}

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

impl MachInst for Inst {
    fn get_regs(&self) -> InstRegUses {
        arm64_get_regs(self)
    }

    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        arm64_map_regs(self, pre_map, post_map);
    }

    fn is_move(&self) -> Option<(WritableReg<Reg>, Reg)> {
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

    fn gen_move(to_reg: WritableReg<Reg>, from_reg: Reg) -> Inst {
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

//=============================================================================
// Pretty-printing of instructions.

impl ShowWithRRU for Imm12 {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        let shift = if self.shift12 { 12 } else { 0 };
        let value = self.bits << shift;
        format!("#{}", value)
    }
}

impl ShowWithRRU for SImm7 {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.bits)
    }

    fn show_rru_sized(&self, _mb_rru: Option<&RealRegUniverse>, size: u8) -> String {
        format!("#{}", self.bits * (size as i16))
    }
}

impl ShowWithRRU for SImm9 {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.bits)
    }
}

impl ShowWithRRU for UImm12Scaled {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        let scale = self.scale_ty.bytes();
        format!("#{}", self.bits * scale as u16)
    }
}

impl ShowWithRRU for ImmLogic {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value())
    }
}

impl ShowWithRRU for ImmShift {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.imm)
    }
}

impl ShowWithRRU for MoveWideConst {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value())
    }
}

impl ShowWithRRU for ShiftOpAndAmt {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("{:?} {}", self.op(), self.amt().value())
    }
}

impl ShowWithRRU for ExtendOp {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("{:?}", self)
    }
}

impl ShowWithRRU for MemLabel {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &MemLabel::ConstantPool(off) => format!("{}", off),
            // Should be resolved into an offset before we pretty-print.
            &MemLabel::ConstantData(..) => "!!constant!!".to_string(),
        }
    }
}

fn shift_for_type(ty: Type) -> usize {
    match ty.bytes() {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => panic!("unknown type"),
    }
}

impl ShowWithRRU for MemArg {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &MemArg::Base(reg) => format!("[{}]", reg.show_rru(mb_rru)),
            &MemArg::BaseSImm9(reg, simm9) => {
                format!("[{}, {}]", reg.show_rru(mb_rru), simm9.show_rru(mb_rru))
            }
            &MemArg::BaseUImm12Scaled(reg, uimm12scaled) => format!(
                "[{}, {}]",
                reg.show_rru(mb_rru),
                uimm12scaled.show_rru(mb_rru)
            ),
            &MemArg::BasePlusReg(r1, r2) => {
                format!("[{}, {}]", r1.show_rru(mb_rru), r2.show_rru(mb_rru))
            }
            &MemArg::BasePlusRegScaled(r1, r2, ty) => {
                let shift = shift_for_type(ty);
                format!(
                    "[{}, {}, lsl #{}]",
                    r1.show_rru(mb_rru),
                    r2.show_rru(mb_rru),
                    shift,
                )
            }
            &MemArg::Label(ref label) => label.show_rru(mb_rru),
            &MemArg::PreIndexed(r, simm9) => format!(
                "[{}, {}]!",
                r.to_reg().show_rru(mb_rru),
                simm9.show_rru(mb_rru)
            ),
            &MemArg::PostIndexed(r, simm9) => format!(
                "[{}], {}",
                r.to_reg().show_rru(mb_rru),
                simm9.show_rru(mb_rru)
            ),
            // Eliminated by `mem_finalize()`.
            &MemArg::StackOffset(..) => panic!("Unexpected StackOffset mem-arg mode!"),
        }
    }
}

impl ShowWithRRU for PairMemArg {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &PairMemArg::Base(reg) => format!("[{}]", reg.show_rru(mb_rru)),
            &PairMemArg::BaseSImm7(reg, simm7) => {
                format!("[{}, {}]", reg.show_rru(mb_rru), simm7.show_rru(mb_rru))
            }
            &PairMemArg::PreIndexed(reg, simm7) => format!(
                "[{}, {}]!",
                reg.to_reg().show_rru(mb_rru),
                simm7.show_rru(mb_rru)
            ),
            &PairMemArg::PostIndexed(reg, simm7) => format!(
                "[{}], {}",
                reg.to_reg().show_rru(mb_rru),
                simm7.show_rru(mb_rru)
            ),
        }
    }

    fn show_rru_sized(&self, mb_rru: Option<&RealRegUniverse>, size: u8) -> String {
        match self {
            &PairMemArg::Base(reg) => format!("[{}]", reg.show_rru(mb_rru)),
            &PairMemArg::BaseSImm7(reg, simm7) => format!(
                "[{}, {}]",
                reg.show_rru(mb_rru),
                simm7.show_rru_sized(mb_rru, size)
            ),
            &PairMemArg::PreIndexed(reg, simm7) => format!(
                "[{}, {}]!",
                reg.to_reg().show_rru(mb_rru),
                simm7.show_rru_sized(mb_rru, size)
            ),
            &PairMemArg::PostIndexed(reg, simm7) => format!(
                "[{}], {}",
                reg.to_reg().show_rru(mb_rru),
                simm7.show_rru_sized(mb_rru, size)
            ),
        }
    }
}

impl ShowWithRRU for Cond {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        let mut s = format!("{:?}", self);
        s.make_ascii_lowercase();
        s
    }
}

impl ShowWithRRU for BranchTarget {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &BranchTarget::Block(block) => format!("block{}", block),
            &BranchTarget::ResolvedOffset(block, _) => format!("block{}", block),
        }
    }
}
fn mem_finalize_for_show(mem: &MemArg, mb_rru: Option<&RealRegUniverse>) -> (String, MemArg) {
    let (mem_insts, mem) = mem_finalize(mem);
    let mut mem_str = mem_insts
        .into_iter()
        .map(|inst| inst.show_rru(mb_rru))
        .collect::<Vec<_>>()
        .join(" ; ");
    if !mem_str.is_empty() {
        mem_str += " ; ";
    }

    (mem_str, mem)
}

impl ShowWithRRU for Inst {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        fn op_is32(alu_op: ALUOp) -> (&'static str, bool) {
            match alu_op {
                ALUOp::Add32 => ("add", true),
                ALUOp::Add64 => ("add", false),
                ALUOp::Sub32 => ("sub", true),
                ALUOp::Sub64 => ("sub", false),
                ALUOp::Orr32 => ("orr", true),
                ALUOp::Orr64 => ("orr", false),
                ALUOp::And32 => ("and", true),
                ALUOp::And64 => ("and", false),
                ALUOp::SubS32 => ("subs", true),
                ALUOp::SubS64 => ("subs", false),
            }
        }

        match self {
            &Inst::Nop => "".to_string(),
            &Inst::Nop4 => "nop".to_string(),
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);
                let rm = show_ireg_sized(rm, mb_rru, is32);
                format!("{} {}, {}, {}", op, rd, rn, rm)
            }
            &Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                ref imm12,
            } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);

                if imm12.bits == 0 && alu_op == ALUOp::Add64 {
                    // special-case MOV (used for moving into SP).
                    format!("mov {}, {}", rd, rn)
                } else {
                    let imm12 = imm12.show_rru(mb_rru);
                    format!("{} {}, {}, {}", op, rd, rn, imm12)
                }
            }
            &Inst::AluRRImmLogic {
                alu_op,
                rd,
                rn,
                ref imml,
            } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);
                let imml = imml.show_rru(mb_rru);
                format!("{} {}, {}, {}", op, rd, rn, imml)
            }
            &Inst::AluRRImmShift {
                alu_op,
                rd,
                rn,
                ref immshift,
            } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);
                let immshift = immshift.show_rru(mb_rru);
                format!("{} {}, {}, {}", op, rd, rn, immshift)
            }
            &Inst::AluRRRShift {
                alu_op,
                rd,
                rn,
                rm,
                ref shiftop,
            } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);
                let rm = show_ireg_sized(rm, mb_rru, is32);
                let shiftop = shiftop.show_rru(mb_rru);
                format!("{} {}, {}, {}, {}", op, rd, rn, rm, shiftop)
            }
            &Inst::AluRRRExtend {
                alu_op,
                rd,
                rn,
                rm,
                ref extendop,
            } => {
                let (op, is32) = op_is32(alu_op);
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let rn = show_ireg_sized(rn, mb_rru, is32);
                let rm = show_ireg_sized(rm, mb_rru, is32);
                let extendop = extendop.show_rru(mb_rru);
                format!("{} {}, {}, {}, {}", op, rd, rn, rm, extendop)
            }
            &Inst::ULoad8 { rd, ref mem }
            | &Inst::SLoad8 { rd, ref mem }
            | &Inst::ULoad16 { rd, ref mem }
            | &Inst::SLoad16 { rd, ref mem }
            | &Inst::ULoad32 { rd, ref mem }
            | &Inst::SLoad32 { rd, ref mem }
            | &Inst::ULoad64 { rd, ref mem } => {
                let (mem_str, mem) = mem_finalize_for_show(mem, mb_rru);

                let is_unscaled_base = match &mem {
                    &MemArg::Base(..) | &MemArg::BaseSImm9(..) => true,
                    _ => false,
                };
                let (op, is32) = match (self, is_unscaled_base) {
                    (&Inst::ULoad8 { .. }, false) => ("ldrb", true),
                    (&Inst::ULoad8 { .. }, true) => ("ldurb", true),
                    (&Inst::SLoad8 { .. }, false) => ("ldrsb", false),
                    (&Inst::SLoad8 { .. }, true) => ("ldursb", false),
                    (&Inst::ULoad16 { .. }, false) => ("ldrh", true),
                    (&Inst::ULoad16 { .. }, true) => ("ldurh", true),
                    (&Inst::SLoad16 { .. }, false) => ("ldrsh", false),
                    (&Inst::SLoad16 { .. }, true) => ("ldursh", false),
                    (&Inst::ULoad32 { .. }, false) => ("ldr", true),
                    (&Inst::ULoad32 { .. }, true) => ("ldur", true),
                    (&Inst::SLoad32 { .. }, false) => ("ldrsw", false),
                    (&Inst::SLoad32 { .. }, true) => ("ldursw", false),
                    (&Inst::ULoad64 { .. }, false) => ("ldr", false),
                    (&Inst::ULoad64 { .. }, true) => ("ldur", false),
                    _ => unreachable!(),
                };
                let rd = show_ireg_sized(rd.to_reg(), mb_rru, is32);
                let mem = mem.show_rru(mb_rru);
                format!("{}{} {}, {}", mem_str, op, rd, mem)
            }
            &Inst::Store8 { rd, ref mem }
            | &Inst::Store16 { rd, ref mem }
            | &Inst::Store32 { rd, ref mem }
            | &Inst::Store64 { rd, ref mem } => {
                let (mem_str, mem) = mem_finalize_for_show(mem, mb_rru);

                let is_unscaled_base = match &mem {
                    &MemArg::Base(..) | &MemArg::BaseSImm9(..) => true,
                    _ => false,
                };
                let (op, is32) = match (self, is_unscaled_base) {
                    (&Inst::Store8 { .. }, false) => ("strb", true),
                    (&Inst::Store8 { .. }, true) => ("sturb", true),
                    (&Inst::Store16 { .. }, false) => ("strh", true),
                    (&Inst::Store16 { .. }, true) => ("sturh", true),
                    (&Inst::Store32 { .. }, false) => ("str", true),
                    (&Inst::Store32 { .. }, true) => ("stur", true),
                    (&Inst::Store64 { .. }, false) => ("str", false),
                    (&Inst::Store64 { .. }, true) => ("stur", false),
                    _ => unreachable!(),
                };
                let rd = show_ireg_sized(rd, mb_rru, is32);
                let mem = mem.show_rru(mb_rru);
                format!("{}{} {}, {}", mem_str, op, rd, mem)
            }
            &Inst::StoreP64 { rt, rt2, ref mem } => {
                let rt = rt.show_rru(mb_rru);
                let rt2 = rt2.show_rru(mb_rru);
                let mem = mem.show_rru_sized(mb_rru, /* size = */ 8);
                format!("stp {}, {}, {}", rt, rt2, mem)
            }
            &Inst::LoadP64 { rt, rt2, ref mem } => {
                let rt = rt.to_reg().show_rru(mb_rru);
                let rt2 = rt2.to_reg().show_rru(mb_rru);
                let mem = mem.show_rru_sized(mb_rru, /* size = */ 8);
                format!("ldp {}, {}, {}", rt, rt2, mem)
            }
            &Inst::Mov { rd, rm } => {
                let rd = rd.to_reg().show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("mov {}, {}", rd, rm)
            }
            &Inst::MovZ { rd, ref imm } => {
                let rd = rd.to_reg().show_rru(mb_rru);
                let imm = imm.show_rru(mb_rru);
                format!("movz {}, {}", rd, imm)
            }
            &Inst::MovN { rd, ref imm } => {
                let rd = rd.to_reg().show_rru(mb_rru);
                let imm = imm.show_rru(mb_rru);
                format!("movn {}, {}", rd, imm)
            }
            &Inst::Call { dest: _ } => {
                let dest = "!!".to_string(); // TODO
                format!("bl {}", dest)
            }
            &Inst::CallInd { rn } => {
                let rn = rn.show_rru(mb_rru);
                format!("bl {}", rn)
            }
            &Inst::Ret {} => "ret".to_string(),
            &Inst::Jump { ref dest } => {
                let dest = dest.show_rru(mb_rru);
                format!("b {}", dest)
            }
            &Inst::CondBr {
                ref taken,
                ref not_taken,
                ref kind,
            } => {
                let taken = taken.show_rru(mb_rru);
                let not_taken = not_taken.show_rru(mb_rru);
                match kind {
                    &CondBrKind::Zero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbz {}, {} ; b {}", reg, taken, not_taken)
                    }
                    &CondBrKind::NotZero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbnz {}, {} ; b {}", reg, taken, not_taken)
                    }
                    &CondBrKind::Cond(c) => {
                        let c = c.show_rru(mb_rru);
                        format!("b.{} {} ; b {}", c, taken, not_taken)
                    }
                }
            }
            &Inst::CondBrLowered {
                ref target,
                inverted,
                ref kind,
            } => {
                let target = target.show_rru(mb_rru);
                let kind = if inverted {
                    kind.invert()
                } else {
                    kind.clone()
                };
                match &kind {
                    &CondBrKind::Zero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbz {}, {}", reg, target)
                    }
                    &CondBrKind::NotZero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbnz {}, {}", reg, target)
                    }
                    &CondBrKind::Cond(c) => {
                        let c = c.show_rru(mb_rru);
                        format!("b.{} {}", c, target)
                    }
                }
            }
            &Inst::CondBrLoweredCompound {
                ref taken,
                ref not_taken,
                ref kind,
            } => {
                let first = Inst::CondBrLowered {
                    target: taken.clone(),
                    inverted: false,
                    kind: kind.clone(),
                };
                let second = Inst::Jump {
                    dest: not_taken.clone(),
                };
                first.show_rru(mb_rru) + " ; " + &second.show_rru(mb_rru)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::isa::test_utils;

    #[test]
    fn test_arm64_binemit() {
        let mut insns = Vec::<(Inst, &str, &str)>::new();

        // N.B.: the architecture is little-endian, so when transcribing the 32-bit
        // hex instructions from e.g. objdump disassembly, one must swap the bytes
        // seen below. (E.g., a `ret` is normally written as the u32 `D65F03C0`,
        // but we write it here as C0035FD6.)

        // Useful helper script to produce the encodings from the text:
        //
        //      #!/bin/sh
        //      tmp=`mktemp /tmp/XXXXXXXX.o`
        //      aarch64-linux-gnu-as /dev/stdin -o $tmp
        //      aarch64-linux-gnu-objdump -d $tmp
        //      rm -f $tmp
        //
        // Then:
        //
        //      $ echo "mov x1, x2" | arm64inst.sh

        insns.push((Inst::Ret {}, "C0035FD6", "ret"));
        insns.push((Inst::Nop {}, "", ""));
        insns.push((Inst::Nop4 {}, "1F2003D5", "nop"));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Add32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100030B",
            "add w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Add64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400068B",
            "add x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Sub32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100034B",
            "sub w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Sub64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40006CB",
            "sub x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Orr32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100032A",
            "orr w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Orr64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40006AA",
            "orr x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::And32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100030A",
            "and w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::And64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400068A",
            "and x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::SubS32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100036B",
            "subs w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::SubS64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40006EB",
            "subs x4, x5, x6",
        ));

        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::Add32,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D0411",
            "add w7, w8, #291",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::Add32,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: true,
                },
            },
            "078D4411",
            "add w7, w8, #1191936",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::Add64,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D0491",
            "add x7, x8, #291",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::Sub32,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D0451",
            "sub w7, w8, #291",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::Sub64,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D04D1",
            "sub x7, x8, #291",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::SubS32,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D0471",
            "subs w7, w8, #291",
        ));
        insns.push((
            Inst::AluRRImm12 {
                alu_op: ALUOp::SubS64,
                rd: writable_xreg(7),
                rn: xreg(8),
                imm12: Imm12 {
                    bits: 0x123,
                    shift12: false,
                },
            },
            "078D04F1",
            "subs x7, x8, #291",
        ));

        // TODO: ImmLogic forms (once logic-immediate encoding/decoding exists).

        // TODO: AluRRRShift and ALURRRExtend forms.

        insns.push((
            Inst::ULoad8 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41004038",
            "ldurb w1, [x2]",
        ));
        insns.push((
            Inst::SLoad8 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41008038",
            "ldursb x1, [x2]",
        ));
        insns.push((
            Inst::ULoad16 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41004078",
            "ldurh w1, [x2]",
        ));
        insns.push((
            Inst::SLoad16 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41008078",
            "ldursh x1, [x2]",
        ));
        insns.push((
            Inst::ULoad32 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "410040B8",
            "ldur w1, [x2]",
        ));
        insns.push((
            Inst::SLoad32 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "410080B8",
            "ldursw x1, [x2]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "410040F8",
            "ldur x1, [x2]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::BaseSImm9(xreg(2), SImm9::maybe_from_i64(-256).unwrap()),
            },
            "410050F8",
            "ldur x1, [x2, #-256]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::BaseSImm9(xreg(2), SImm9::maybe_from_i64(255).unwrap()),
            },
            "41F04FF8",
            "ldur x1, [x2, #255]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::BaseUImm12Scaled(
                    xreg(2),
                    UImm12Scaled::maybe_from_i64(32760, I64).unwrap(),
                ),
            },
            "41FC7FF9",
            "ldr x1, [x2, #32760]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::BasePlusReg(xreg(2), xreg(3)),
            },
            "416863F8",
            "ldr x1, [x2, x3]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::BasePlusRegScaled(xreg(2), xreg(3), I64),
            },
            "417863F8",
            "ldr x1, [x2, x3, lsl #3]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Label(MemLabel::ConstantPool(64)),
            },
            "01020058",
            "ldr x1, 64",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::PreIndexed(writable_xreg(2), SImm9::maybe_from_i64(16).unwrap()),
            },
            "410C41F8",
            "ldr x1, [x2, #16]!",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::PostIndexed(writable_xreg(2), SImm9::maybe_from_i64(16).unwrap()),
            },
            "410441F8",
            "ldr x1, [x2], #16",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::StackOffset(32768),
            },
            "0F000058EF011D8BE10140F8",
            "ldr x15, !!constant!! ; add x15, x15, fp ; ldur x1, [x15]",
        ));

        insns.push((
            Inst::Store8 {
                rd: xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41000038",
            "sturb w1, [x2]",
        ));
        insns.push((
            Inst::Store8 {
                rd: xreg(1),
                mem: MemArg::BaseUImm12Scaled(
                    xreg(2),
                    UImm12Scaled::maybe_from_i64(4095, I8).unwrap(),
                ),
            },
            "41FC3F39",
            "strb w1, [x2, #4095]",
        ));
        insns.push((
            Inst::Store16 {
                rd: xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "41000078",
            "sturh w1, [x2]",
        ));
        insns.push((
            Inst::Store16 {
                rd: xreg(1),
                mem: MemArg::BaseUImm12Scaled(
                    xreg(2),
                    UImm12Scaled::maybe_from_i64(8190, I16).unwrap(),
                ),
            },
            "41FC3F79",
            "strh w1, [x2, #8190]",
        ));
        insns.push((
            Inst::Store32 {
                rd: xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "410000B8",
            "stur w1, [x2]",
        ));
        insns.push((
            Inst::Store32 {
                rd: xreg(1),
                mem: MemArg::BaseUImm12Scaled(
                    xreg(2),
                    UImm12Scaled::maybe_from_i64(16380, I32).unwrap(),
                ),
            },
            "41FC3FB9",
            "str w1, [x2, #16380]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::Base(xreg(2)),
            },
            "410000F8",
            "stur x1, [x2]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::BaseUImm12Scaled(
                    xreg(2),
                    UImm12Scaled::maybe_from_i64(32760, I64).unwrap(),
                ),
            },
            "41FC3FF9",
            "str x1, [x2, #32760]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::BasePlusReg(xreg(2), xreg(3)),
            },
            "416823F8",
            "str x1, [x2, x3]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::BasePlusRegScaled(xreg(2), xreg(3), I64),
            },
            "417823F8",
            "str x1, [x2, x3, lsl #3]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::PreIndexed(writable_xreg(2), SImm9::maybe_from_i64(16).unwrap()),
            },
            "410C01F8",
            "str x1, [x2, #16]!",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::PostIndexed(writable_xreg(2), SImm9::maybe_from_i64(16).unwrap()),
            },
            "410401F8",
            "str x1, [x2], #16",
        ));

        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::Base(xreg(10)),
            },
            "482500A9",
            "stp x8, x9, [x10]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::BaseSImm7(xreg(10), SImm7::maybe_from_i64(63).unwrap()),
            },
            "48A51FA9",
            "stp x8, x9, [x10, #504]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::BaseSImm7(xreg(10), SImm7::maybe_from_i64(-64).unwrap()),
            },
            "482520A9",
            "stp x8, x9, [x10, #-512]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::PreIndexed(writable_xreg(10), SImm7::maybe_from_i64(-64).unwrap()),
            },
            "4825A0A9",
            "stp x8, x9, [x10, #-512]!",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::PostIndexed(writable_xreg(10), SImm7::maybe_from_i64(63).unwrap()),
            },
            "48A59FA8",
            "stp x8, x9, [x10], #504",
        ));

        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::Base(xreg(10)),
            },
            "482540A9",
            "ldp x8, x9, [x10]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::BaseSImm7(xreg(10), SImm7::maybe_from_i64(63).unwrap()),
            },
            "48A55FA9",
            "ldp x8, x9, [x10, #504]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::BaseSImm7(xreg(10), SImm7::maybe_from_i64(-64).unwrap()),
            },
            "482560A9",
            "ldp x8, x9, [x10, #-512]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::PreIndexed(writable_xreg(10), SImm7::maybe_from_i64(-64).unwrap()),
            },
            "4825E0A9",
            "ldp x8, x9, [x10, #-512]!",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::PostIndexed(writable_xreg(10), SImm7::maybe_from_i64(63).unwrap()),
            },
            "48A5DFA8",
            "ldp x8, x9, [x10], #504",
        ));

        insns.push((
            Inst::Mov {
                rd: writable_xreg(8),
                rm: xreg(9),
            },
            "E80309AA",
            "mov x8, x9",
        ));

        insns.push((
            Inst::MovZ {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_0000_0000_ffff).unwrap(),
            },
            "E8FF9FD2",
            "movz x8, #65535",
        ));
        insns.push((
            Inst::MovZ {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_0000_ffff_0000).unwrap(),
            },
            "E8FFBFD2",
            "movz x8, #4294901760",
        ));
        insns.push((
            Inst::MovZ {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_ffff_0000_0000).unwrap(),
            },
            "E8FFDFD2",
            "movz x8, #281470681743360",
        ));
        insns.push((
            Inst::MovZ {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0xffff_0000_0000_0000).unwrap(),
            },
            "E8FFFFD2",
            "movz x8, #18446462598732840960",
        ));

        insns.push((
            Inst::MovN {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_0000_0000_ffff).unwrap(),
            },
            "E8FF9F92",
            "movn x8, #65535",
        ));
        insns.push((
            Inst::MovN {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_0000_ffff_0000).unwrap(),
            },
            "E8FFBF92",
            "movn x8, #4294901760",
        ));
        insns.push((
            Inst::MovN {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0x0000_ffff_0000_0000).unwrap(),
            },
            "E8FFDF92",
            "movn x8, #281470681743360",
        ));
        insns.push((
            Inst::MovN {
                rd: writable_xreg(8),
                imm: MoveWideConst::maybe_from_u64(0xffff_0000_0000_0000).unwrap(),
            },
            "E8FFFF92",
            "movn x8, #18446462598732840960",
        ));

        insns.push((
            Inst::Jump {
                dest: BranchTarget::ResolvedOffset(0, 64),
            },
            "10000014",
            "b block0",
        ));

        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Zero(xreg(8)),
            },
            "080200B4",
            "cbz x8, block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: true,
                kind: CondBrKind::Zero(xreg(8)),
            },
            "080200B5",
            "cbnz x8, block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::NotZero(xreg(8)),
            },
            "080200B5",
            "cbnz x8, block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: true,
                kind: CondBrKind::NotZero(xreg(8)),
            },
            "080200B4",
            "cbz x8, block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Eq),
            },
            "00020054",
            "b.eq block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Ne),
            },
            "01020054",
            "b.ne block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: true,
                kind: CondBrKind::Cond(Cond::Ne),
            },
            "00020054",
            "b.eq block0",
        ));

        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Hs),
            },
            "02020054",
            "b.hs block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Lo),
            },
            "03020054",
            "b.lo block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Mi),
            },
            "04020054",
            "b.mi block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Pl),
            },
            "05020054",
            "b.pl block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Vs),
            },
            "06020054",
            "b.vs block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Vc),
            },
            "07020054",
            "b.vc block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Hi),
            },
            "08020054",
            "b.hi block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Ls),
            },
            "09020054",
            "b.ls block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Ge),
            },
            "0A020054",
            "b.ge block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Lt),
            },
            "0B020054",
            "b.lt block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Gt),
            },
            "0C020054",
            "b.gt block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Le),
            },
            "0D020054",
            "b.le block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Al),
            },
            "0E020054",
            "b.al block0",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(0, 64),
                inverted: false,
                kind: CondBrKind::Cond(Cond::Nv),
            },
            "0F020054",
            "b.nv block0",
        ));

        insns.push((
            Inst::CondBrLoweredCompound {
                taken: BranchTarget::ResolvedOffset(0, 64),
                not_taken: BranchTarget::ResolvedOffset(1, 128),
                kind: CondBrKind::Cond(Cond::Le),
            },
            "0D02005420000014",
            "b.le block0 ; b block1",
        ));

        let rru = create_reg_universe();
        for (insn, expected_encoding, expected_printing) in insns {
            println!(
                "ARM64: {:?}, {}, {}",
                insn, expected_encoding, expected_printing
            );

            // Check the printed text is as expected.
            let actual_printing = insn.show_rru(Some(&rru));
            assert_eq!(expected_printing, actual_printing);

            // Check the encoding is as expected.
            let mut sink = test_utils::TestCodeSink::new();
            insn.emit(&mut sink);
            let actual_encoding = &sink.stringify();
            assert_eq!(expected_encoding, actual_encoding);
        }
    }

    #[test]
    fn test_cond_invert() {
        for cond in vec![
            Cond::Eq,
            Cond::Ne,
            Cond::Hs,
            Cond::Lo,
            Cond::Mi,
            Cond::Pl,
            Cond::Vs,
            Cond::Vc,
            Cond::Hi,
            Cond::Ls,
            Cond::Ge,
            Cond::Lt,
            Cond::Gt,
            Cond::Le,
            Cond::Al,
            Cond::Nv,
        ]
        .into_iter()
        {
            assert_eq!(cond.invert().invert(), cond);
        }
    }
}

// TODO (test): lowering
// - simple and complex addressing modes
// - immediate in second arg
// - extend op in second arg
// - shift op in second arg
// - constants of various sizes
// - values of different widths
