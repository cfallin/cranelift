//! ARM64 ISA definitions: instruction arguments.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::binemit::{CodeOffset, CodeSink, ConstantPoolSink, NullConstantPoolSink};
use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::Type;
use crate::isa::arm64::inst::*;
use crate::machinst::*;

use regalloc::{
    RealReg, RealRegUniverse, Reg, RegClass, RegClassInfo, SpillSlot, VirtualReg, Writable,
    NUM_REG_CLASSES,
};

use std::string::{String, ToString};

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
    pub const MAX_SHIFT: u64 = 63;

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
    /// A value in a constant pool, already emitted, with relative offset from
    /// this instruction. This form must be used just before emission.
    ConstantPoolRel(ConstantOffset),
    /// A value in a constant pool, to be emitted during binemit. This form is
    /// created during isel and is lowered during emission.
    ConstantData(ConstantData),
}

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    Label(MemLabel),
    PostIndexed(Writable<Reg>, SImm9),
    PreIndexed(Writable<Reg>, SImm9),
    // TODO: Support more than LSL.
    RegScaled(Reg, Reg, Type, bool),
    Unscaled(Reg, SImm9),
    UnsignedOffset(Reg, UImm12Scaled),
    /// Offset from the frame pointer.
    StackOffset(i64),
}

impl MemArg {
    /// Memory reference using an address in a register.
    pub fn reg(reg: Reg) -> MemArg {
        MemArg::Unscaled(reg, SImm9::zero())
    }

    /// Memory reference using an address in a register and an offset, if possible.
    pub fn reg_maybe_offset(reg: Reg, offset: i64, value_type: Type) -> Option<MemArg> {
        if offset == 0 {
            Some(MemArg::Unscaled(reg, SImm9::zero()))
        } else if let Some(simm9) = SImm9::maybe_from_i64(offset) {
            Some(MemArg::Unscaled(reg, simm9))
        } else if let Some(uimm12s) = UImm12Scaled::maybe_from_i64(offset, value_type) {
            Some(MemArg::UnsignedOffset(reg, uimm12s))
        } else {
            None
        }
    }

    /// Memory reference using the sum of two registers as an address.
    pub fn reg_reg(reg1: Reg, reg2: Reg) -> MemArg {
        MemArg::RegScaled(reg1, reg2, I64, false)
    }

    /// Memory reference to a label: a global function or value, or data in the constant pool.
    pub fn label(label: MemLabel) -> MemArg {
        MemArg::Label(label)
    }
}

/// A memory argument to a load/store-pair.
#[derive(Clone, Debug)]
pub enum PairMemArg {
    SignedOffset(Reg, SImm7Scaled),
    PreIndexed(Writable<Reg>, SImm7Scaled),
    PostIndexed(Writable<Reg>, SImm7Scaled),
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

    /// Return the machine encoding of this condition.
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

    /// Get the offset as 4-byte words (instructions).
    pub fn as_offset_words(&self) -> Option<isize> {
        match self {
            &BranchTarget::ResolvedOffset(_, off) => Some(off >> 2),
            _ => None,
        }
    }

    /// Get the offset as a 26-bit offset suitable for a 26-bit jump.
    pub fn as_off26(&self) -> Option<u32> {
        self.as_offset_words().and_then(|i| {
            if (i < (1 << 25)) && (i >= -(1 << 25)) {
                Some((i as u32) & ((1 << 26) - 1))
            } else {
                None
            }
        })
    }

    /// Get the offset as a 16-bit offset.
    pub fn as_off19(&self) -> Option<u32> {
        self.as_offset_words().and_then(|i| {
            if (i < (1 << 18)) && (i >= -(1 << 18)) {
                Some((i as u32) & ((1 << 19) - 1))
            } else {
                None
            }
        })
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
            &MemLabel::ConstantPoolRel(off) => format!("{}", off),
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
            &MemArg::Unscaled(reg, simm9) => {
                if simm9.value != 0 {
                    format!("[{}, {}]", reg.show_rru(mb_rru), simm9.show_rru(mb_rru))
                } else {
                    format!("[{}]", reg.show_rru(mb_rru))
                }
            }
            &MemArg::UnsignedOffset(reg, uimm12scaled) => format!(
                "[{}, {}]",
                reg.show_rru(mb_rru),
                uimm12scaled.show_rru(mb_rru)
            ),
            &MemArg::RegScaled(r1, r2, ty, scaled) => {
                if scaled {
                    let shift = shift_for_type(ty);
                    format!(
                        "[{}, {}, lsl #{}]",
                        r1.show_rru(mb_rru),
                        r2.show_rru(mb_rru),
                        shift,
                    )
                } else {
                    format!("[{}, {}]", r1.show_rru(mb_rru), r2.show_rru(mb_rru),)
                }
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
            &PairMemArg::SignedOffset(reg, simm7) => {
                if simm7.value != 0 {
                    format!("[{}, {}]", reg.show_rru(mb_rru), simm7.show_rru(mb_rru))
                } else {
                    format!("[{}]", reg.show_rru(mb_rru))
                }
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
