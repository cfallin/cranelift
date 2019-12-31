//! This module defines arm64-specific machine instruction types.

use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{FuncRef, GlobalValue};
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::*;

/// An ALU operation. This can be paired with several instruction formats below (see `Inst`) in any
/// combination.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ALUOp {
    Add32,
    Add64,
    Sub32,
    Sub64,
}

/// Instruction formats.
#[derive(Clone, Debug)]
pub enum Inst {
    /// An ALU operation with two register sources and a register destination.
    AluRRR {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        rm: MachReg,
    },
    /// An ALU operation with a register source and an immediate-12 source, and a register
    /// destination.
    AluRRImm12 {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        imm12: Imm12,
    },
    /// An ALU operation with a register source and an immediate-logic source, and a register destination.
    AluRRImmLogic {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        imml: ImmLogic,
    },
    /// An ALU operation with a register source and an immediate-shiftamt source, and a register destination.
    AluRRImmShift {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        immshift: ImmShift,
    },
    /// An ALU operation with two register sources, one of which can be shifted, and a register
    /// destination.
    AluRRRShift {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        rm: MachReg,
        shiftop: ShiftOpAndAmt,
    },
    /// An ALU operation with two register sources, one of which can be {zero,sign}-extended and
    /// shifted, and a register destination.
    AluRRRExtend {
        alu_op: ALUOp,
        rd: MachReg,
        rn: MachReg,
        rm: MachReg,
        extendop: ExtendOpAndAmt,
    },
    /// A load with a register destination and a memory source.
    Load { rd: MachReg, mem: MemArg },
    /// A store with a register source and a memory destination.
    Store { rd: MachReg, mem: MemArg },
    // TODO: control flow ops.
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
    pub fn maybe_from_u64(val: u64) -> Option<ImmLogic> {
        // TODO: implement.
        None
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
    shift: usize,
}

impl ShiftOpAndAmt {
    /// Maximum shift for shifted-register operands.
    pub const MAX_SHIFT: usize = 7;

    /// Create a new shiftop-with-amount.
    pub fn new(op: ShiftOp, shift: usize) -> ShiftOpAndAmt {
        assert!(shift <= Self::MAX_SHIFT);
        ShiftOpAndAmt { op, shift }
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
        ExtendOpAndAmt { op, amt }
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
    Base(MachReg),
    BaseImm(MachReg, usize),
    BaseOffsetShifted(MachReg, usize),
    BaseImmPreIndexed(MachReg, usize),
    BaseImmPostIndexed(MachReg, usize),
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
