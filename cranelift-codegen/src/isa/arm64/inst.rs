//! This module defines arm64-specific machine instruction types.

use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{Ebb, FuncRef, GlobalValue, Type};
use crate::isa::arm64::registers::*;
use crate::isa::registers::RegClass;
use crate::machinst::*;

use smallvec::SmallVec;
use std::mem;

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
    /// An unsigned (zero-extending) 8-bit load.
    ULoad8 { rd: MachReg, mem: MemArg },
    /// A signed (sign-extending) 8-bit load.
    SLoad8 { rd: MachReg, mem: MemArg },
    /// An unsigned (zero-extending) 16-bit load.
    ULoad16 { rd: MachReg, mem: MemArg },
    /// A signed (sign-extending) 16-bit load.
    SLoad16 { rd: MachReg, mem: MemArg },
    /// An unsigned (zero-extending) 32-bit load.
    ULoad32 { rd: MachReg, mem: MemArg },
    /// A signed (sign-extending) 32-bit load.
    SLoad32 { rd: MachReg, mem: MemArg },
    /// A 64-bit load.
    ULoad64 { rd: MachReg, mem: MemArg },

    /// An 8-bit store.
    Store8 { rd: MachReg, mem: MemArg },
    /// A 16-bit store.
    Store16 { rd: MachReg, mem: MemArg },
    /// A 32-bit store.
    Store32 { rd: MachReg, mem: MemArg },
    /// A 64-bit store.
    Store64 { rd: MachReg, mem: MemArg },

    /// An unconditional branch.
    Jump { dest: Ebb },
    /// A machine call instruction.
    Call { dest: FuncRef },
    /// A machine return instruction.
    Ret {},
    /// A machine indirect-branch instruction.
    JumpInd { rn: MachReg },
    /// A machine indirect-call instruction.
    CallInd { rn: MachReg },

    /// A conditional branch on zero.
    CondBrZ { dest: Ebb, rt: MachReg },
    /// A conditional branch on nonzero.
    CondBrNZ { dest: Ebb, rt: MachReg },
    /// A compare / conditional branch sequence.
    CmpCondBr {
        dest: Ebb,
        rn: MachReg,
        rm: MachReg,
        cond: Cond,
    },

    /// Virtual instruction: a move for regalloc.
    RegallocMove { dst: MachReg, src: MachReg },
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
    BaseSImm9(MachReg, SImm9),
    BaseUImm12Scaled(MachReg, UImm12Scaled),
    BasePlusReg(MachReg, MachReg),
    BasePlusRegScaled(MachReg, MachReg, Type),
    Label(MemLabel),
    // TODO: use pre-indexed and post-indexed modes
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

impl MemArg {
    /// Memory reference using an address in a register.
    pub fn reg(reg: MachReg) -> MemArg {
        MemArg::Base(reg)
    }

    /// Memory reference using an address in a register and an offset, if possible.
    pub fn reg_maybe_offset(reg: MachReg, offset: i64, value_type: Type) -> Option<MemArg> {
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
    pub fn reg_reg(reg1: MachReg, reg2: MachReg) -> MemArg {
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
#[derive(Clone, Debug)]
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

impl MachInst for Inst {
    fn regs(&self) -> MachInstRegs {
        // TODO: return all regs in the insn args (including the MemArg).
        SmallVec::new()
    }

    fn map_virtregs(&mut self, locs: &MachLocations) {
        // TODO.
    }

    fn is_move(&self) -> Option<(MachReg, MachReg)> {
        match self {
            &Inst::RegallocMove { dst, src } => Some((src.clone(), dst.clone())),
            _ => None,
        }
    }

    fn finalize(&mut self) {
        match self {
            &mut Inst::RegallocMove { dst, src } => {
                let rd = dst.clone();
                let rn = src.clone();
                let rm = MachReg::zero();
                mem::replace(
                    self,
                    Inst::AluRRR {
                        alu_op: ALUOp::Add64,
                        rd,
                        rn,
                        rm,
                    },
                );
            }
            _ => {}
        }
    }
}
