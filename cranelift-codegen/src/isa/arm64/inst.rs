//! This module defines `Op` and `Arg` and friends for Arm64, along with constructors/helpers for
//! lowering code that constructs `MachInst<Op, Arg>` values.

/*
 * TODO:
 * - Support lowering table keying on "controlling typevar".
 *
 * - Support (Iadd (Uextend ...) ...) and (Iadd (Ishl_imm ...) ...) using extended-register and
 *   shifted-register forms.
 */

use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{FuncRef, GlobalValue, Inst, InstructionData, Type};
use crate::ir::types::*;
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::lower::LowerCtx;
use crate::machinst::*;
use crate::{mach_args, mach_ops};

/// An ALU operation. This can be paired with several instruction formats below (see `Inst`) in any
/// combination.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ALUOp {
    Add32,
    Add64,
    Sub32,
    Sub64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    /// An ALU operation with two register sources and a register destination.
    AluRRR { alu_op: ALUOp, rd: RegRef, rn: RegRef, rm: RegRef },
    /// An ALU operation with a register source and an immediate source, and a register destionat.
    AluRRI { alu_op: ALUOp, rd: RegRef, rn: RegRef, imm12: ShiftedImm, },
    /// An ALU operation with two register sources, one of which can be shifted, and a register
    /// destination.
    AluRRRShift { alu_op: ALUOp, rd: RegRef, rn: RegRef, rm: RegRef, shiftop: ShiftOpAndAmt },
    /// An ALU operation with two register sources, one of which can be {zero,sign}-extended and
    /// shifted, and a register destination.
    AluRRRExtend { alu_op: ALUOp, rd: RegRef, rn: RegRef, rm: RegRef, extendop: ExtendOpAndAmt },
    /// A load with a register destination and a memory source.
    Load { rd: RegRef, mem: MemArg },
    /// A store with a register source and a memory destination.
    Store { rd: RegRef, mem: MemArg },

    // TODO: control flow ops.
}

/// A shifted immediate value in 'imm12' format: supports 12 bits, shifted left by 0 or 12 places.
#[derive(Clone, Debug)]
pub struct ShiftedImm {
    /// The immediate bits.
    pub bits: usize,
    /// Whether the immediate bits are shifted left by 12 or not.
    pub shift12: bool,
}

impl ShiftedImm {
    /// Compute a ShiftedImm from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<ShiftedImm> {
        if val == 0 {
            Some(ShiftedImm { bits: 0, shift12: false })
        } else if val < 0xfff {
            Some(ShiftedImm { bits: val as usize, shift12: false })
        } else if val < 0xfff_000 && (val & 0xfff == 0) {
            Some(ShiftedImm { bits: (val as usize) >> 12, shift12: true })
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
}

/// A shift operator for a register or immediate.
#[derive(Clone, Debug)]
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
    shift: usize,
}

impl ShiftOpAndAmt {
    /// Maximum shift for shifted-register operands.
    pub const MAX_SHIFT: usize = 7;

    /// Create a new shiftop-with-amount.
    pub fn new(op: ShiftOp, shift: usize) -> ShiftOpAndAmt {
        assert!(shift <= Self::MAX_SHIFT);
        ShiftOpAndAmt {
            op,
            shift,
        }
    }

    /// Get the shift op.
    pub fn op(&self) -> ShiftOp {
        self.op.clone()
    }

    /// Get the shift amount.
    pub fn amt(&self) -> usize {
        self.shift
    }
}

/// An extend operator for a register.
#[derive(Clone, Debug)]
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

/// A register-extend operation paired with a shift amount, as accepted by many ALU instructions.
#[derive(Clone, Debug)]
pub struct ExtendOpAndAmt {
    op: ExtendOp,
    amt: usize,
}

impl ExtendOpAndAmt {
    /// Maximum shift value.
    pub const MAX_SHIFT: usize = 63;

    /// Create a new extend-op-with-shift-amount.
    pub fn new(op: ExtendOp, amt: usize) -> ExtendOpAndAmt {
        assert!(amt <= Self::MAX_SHIFT);
        ExtendOpAndAmt {
            op,
            amt,
        }
    }

    /// Get the extend operator.
    pub fn op(&self) -> ExtendOp {
        self.op.clone()
    }

    /// Get the shift amount.
    pub fn amt(&self) -> usize {
        self.amt
    }
}

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    Base(RegRef),
    BaseImm(RegRef, usize),
    BaseOffsetShifted(RegRef, usize),
    BaseImmPreIndexed(RegRef, usize),
    BaseImmPostIndexed(RegRef, usize),
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
