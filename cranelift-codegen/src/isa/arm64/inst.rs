//! This module defines arm64-specific machine instruction types.

use crate::binemit::CodeSink;
use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::{Ebb, FuncRef, GlobalValue, Type};
use crate::isa::arm64::registers::*;
use crate::isa::registers::{RegClass, RegUnit};
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
    And32,
    And64,
    SubS32,
    SubS64,
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
        extendop: ExtendOp,
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

    /// A MOVZ with a 16-bit immediate.
    MovZ { rd: MachReg, imm: MovZConst },

    /// An unconditional branch.
    Jump { dest: Ebb },
    /// A machine call instruction.
    Call { dest: FuncRef },
    /// A machine return instruction.
    Ret {},
    /// A machine indirect-call instruction.
    CallInd { rn: MachReg },

    /// A conditional branch on zero.
    CondBrZ { dest: Ebb, rt: MachReg },
    /// A conditional branch on nonzero.
    CondBrNZ { dest: Ebb, rt: MachReg },
    /// A conditional branch based on machine flags.
    CondBr { dest: Ebb, cond: Cond },

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
    pub fn maybe_from_u64(val: u64) -> Option<ImmLogic> {
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

fn memarg_regs(memarg: &MemArg, regs: &mut MachInstRegs) {
    match memarg {
        &MemArg::Base(reg) | &MemArg::BaseSImm9(reg, ..) | &MemArg::BaseUImm12Scaled(reg, ..) => {
            regs.push((reg.clone(), MachRegMode::Use));
        }
        &MemArg::BasePlusReg(r1, r2) | &MemArg::BasePlusRegScaled(r1, r2, ..) => {
            regs.push((r1.clone(), MachRegMode::Use));
            regs.push((r2.clone(), MachRegMode::Use));
        }
        &MemArg::Label(..) => {}
    }
}

impl MachInst for Inst {
    fn regs(&self) -> MachInstRegs {
        let mut ret = SmallVec::new();
        match self {
            &Inst::AluRRR { rd, rn, rm, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
                ret.push((rm.clone(), MachRegMode::Use));
            }
            &Inst::AluRRImm12 { rd, rn, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
            }
            &Inst::AluRRImmLogic { rd, rn, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
            }
            &Inst::AluRRImmShift { rd, rn, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
            }
            &Inst::AluRRRShift { rd, rn, rm, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
                ret.push((rm.clone(), MachRegMode::Use));
            }
            &Inst::AluRRRExtend { rd, rn, rm, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                ret.push((rn.clone(), MachRegMode::Use));
                ret.push((rm.clone(), MachRegMode::Use));
            }
            &Inst::ULoad8 { rd, ref mem, .. }
            | &Inst::SLoad8 { rd, ref mem, .. }
            | &Inst::ULoad16 { rd, ref mem, .. }
            | &Inst::SLoad16 { rd, ref mem, .. }
            | &Inst::ULoad32 { rd, ref mem, .. }
            | &Inst::SLoad32 { rd, ref mem, .. }
            | &Inst::ULoad64 { rd, ref mem, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
                memarg_regs(mem, &mut ret);
            }
            &Inst::Store8 { rd, ref mem, .. }
            | &Inst::Store16 { rd, ref mem, .. }
            | &Inst::Store32 { rd, ref mem, .. }
            | &Inst::Store64 { rd, ref mem, .. } => {
                ret.push((rd.clone(), MachRegMode::Use));
                memarg_regs(mem, &mut ret);
            }
            &Inst::MovZ { rd, .. } => {
                ret.push((rd.clone(), MachRegMode::Def));
            }
            &Inst::Jump { .. } | &Inst::Call { .. } | &Inst::Ret { .. } => {}
            &Inst::CallInd { rn, .. } => {
                ret.push((rn.clone(), MachRegMode::Use));
            }
            &Inst::CondBrZ { rt, .. } | &Inst::CondBrNZ { rt, .. } => {
                ret.push((rt.clone(), MachRegMode::Use));
            }
            &Inst::CondBr { .. } => {}
            &Inst::RegallocMove { dst, src, .. } => {
                ret.push((dst.clone(), MachRegMode::Def));
                ret.push((src.clone(), MachRegMode::Use));
            }
        }
        ret
    }

    fn reg_constraints(&self) -> MachInstRegConstraints {
        SmallVec::new()
    }

    fn map_virtregs(&mut self, locs: &MachLocations) {
        let map = |r| match r {
            MachReg::Virtual(num) => MachReg::Allocated(locs[num]),
            _ => r.clone(),
        };
        let mapmem = |mem| match mem {
            &MemArg::Base(reg) => MemArg::Base(map(reg)),
            &MemArg::BaseSImm9(reg, simm9) => MemArg::BaseSImm9(map(reg), simm9),
            &MemArg::BaseUImm12Scaled(reg, uimm12) => MemArg::BaseUImm12Scaled(map(reg), uimm12),
            &MemArg::BasePlusReg(r1, r2) => MemArg::BasePlusReg(map(r1), map(r2)),
            &MemArg::BasePlusRegScaled(r1, r2, ty) => {
                MemArg::BasePlusRegScaled(map(r1), map(r2), ty)
            }
            &MemArg::Label(ref l) => MemArg::Label(l.clone()),
        };

        let newval = match self {
            &mut Inst::AluRRR { alu_op, rd, rn, rm } => Inst::AluRRR {
                alu_op,
                rd: map(rd),
                rn: map(rn),
                rm: map(rm),
            },
            &mut Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                ref imm12,
            } => Inst::AluRRImm12 {
                alu_op,
                rd: map(rd),
                rn: map(rn),
                imm12: imm12.clone(),
            },
            &mut Inst::AluRRImmLogic {
                alu_op,
                rd,
                rn,
                ref imml,
            } => Inst::AluRRImmLogic {
                alu_op,
                rd: map(rd),
                rn: map(rn),
                imml: imml.clone(),
            },
            &mut Inst::AluRRImmShift {
                alu_op,
                rd,
                rn,
                ref immshift,
            } => Inst::AluRRImmShift {
                alu_op,
                rd: map(rd),
                rn: map(rn),
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
                rd: map(rd),
                rn: map(rn),
                rm: map(rm),
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
                rd: map(rd),
                rn: map(rn),
                rm: map(rm),
                extendop: extendop.clone(),
            },
            &mut Inst::ULoad8 { rd, ref mem } => Inst::ULoad8 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::SLoad8 { rd, ref mem } => Inst::SLoad8 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::ULoad16 { rd, ref mem } => Inst::ULoad16 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::SLoad16 { rd, ref mem } => Inst::SLoad16 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::ULoad32 { rd, ref mem } => Inst::ULoad32 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::SLoad32 { rd, ref mem } => Inst::SLoad32 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::ULoad64 { rd, ref mem } => Inst::ULoad64 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::Store8 { rd, ref mem } => Inst::Store8 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::Store16 { rd, ref mem } => Inst::Store16 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::Store32 { rd, ref mem } => Inst::Store32 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::Store64 { rd, ref mem } => Inst::Store64 {
                rd: map(rd),
                mem: mapmem(mem),
            },
            &mut Inst::MovZ { rd, ref imm } => Inst::MovZ {
                rd: map(rd),
                imm: imm.clone(),
            },
            &mut Inst::Jump { dest } => Inst::Jump { dest },
            &mut Inst::Call { dest } => Inst::Call { dest },
            &mut Inst::Ret {} => Inst::Ret {},
            &mut Inst::CallInd { rn } => Inst::CallInd { rn: map(rn) },
            &mut Inst::CondBrZ { rt, dest } => Inst::CondBrZ { rt: map(rt), dest },
            &mut Inst::CondBrNZ { rt, dest } => Inst::CondBrNZ { rt: map(rt), dest },
            &mut Inst::CondBr { dest, ref cond } => Inst::CondBr {
                dest,
                cond: cond.clone(),
            },
            &mut Inst::RegallocMove { dst, src } => Inst::RegallocMove {
                dst: map(dst),
                src: map(src),
            },
        };
        *self = newval;
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
            // TODO: convert virtual branch forms into actual branches
            _ => {}
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            // TODO: insts in terms of machine BB indices (vcode::BlockIndex)
            // TODO: virtual two-dest condbr form that finalizes into two
            // insns (or one, if fallthrough)
            _ => MachTerminator::None,
        }
    }
}

fn regunit_to_gpr(ru: RegUnit) -> u8 {
    let bank = &GPR.info.banks[GPR.index as usize];
    assert!(ru >= bank.first_unit);
    assert!(ru < bank.first_unit + bank.units);
    return (ru - bank.first_unit) as u8;
}

fn machreg_to_gpr(m: MachReg) -> u32 {
    match m {
        MachReg::Allocated(ru) => regunit_to_gpr(ru) as u32,
        MachReg::Zero => 31,
        _ => panic!("Non-allocated register at binemit time!"),
    }
}

fn enc_arith_rrr(bits_31_21: u16, bits_15_10: u8, rd: MachReg, rn: MachReg, rm: MachReg) -> u32 {
    ((bits_31_21 as u32) << 21)
        | ((bits_15_10 as u32) << 10)
        | machreg_to_gpr(rd)
        | (machreg_to_gpr(rn) << 5)
        | (machreg_to_gpr(rm) << 16)
}

fn enc_arith_rr_imm12(bits_31_24: u8, immshift: u8, imm12: u16, rn: MachReg, rd: MachReg) -> u32 {
    ((bits_31_24 as u32) << 24)
        | ((immshift as u32) << 22)
        | ((imm12 as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

fn enc_arith_rr_imml(bits_31_23: u16, imm_bits: u16, rn: MachReg, rd: MachReg) -> u32 {
    ((bits_31_23 as u32) << 23)
        | ((imm_bits as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
}

impl<CS: CodeSink> MachInstEmit<CS> for Inst {
    fn size(&self) -> usize {
        4 // RISC!
    }

    fn emit(&self, sink: &mut CS) {
        match self {
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let top11 = match alu_op {
                    ALUOp::Add32 => 0b00001011_001,
                    ALUOp::Add64 => 0b10001011_001,
                    ALUOp::Sub32 => 0b01001011_001,
                    ALUOp::Sub64 => 0b11001011_001,
                    ALUOp::Orr32 => 0b00101010_000,
                    ALUOp::Orr64 => 0b10101010_000,
                    ALUOp::SubS32 => 0b01101011_001,
                    ALUOp::SubS64 => 0b11101011_001,
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
            &Inst::AluRRImmShift { rd, rn, .. } => unimplemented!(),
            &Inst::AluRRRShift { rd, rn, rm, .. } => unimplemented!(),
            &Inst::AluRRRExtend { rd, rn, rm, .. } => unimplemented!(),
            &Inst::ULoad8 { rd, ref mem, .. }
            | &Inst::SLoad8 { rd, ref mem, .. }
            | &Inst::ULoad16 { rd, ref mem, .. }
            | &Inst::SLoad16 { rd, ref mem, .. }
            | &Inst::ULoad32 { rd, ref mem, .. }
            | &Inst::SLoad32 { rd, ref mem, .. }
            | &Inst::ULoad64 { rd, ref mem, .. } => unimplemented!(),
            &Inst::Store8 { rd, ref mem, .. }
            | &Inst::Store16 { rd, ref mem, .. }
            | &Inst::Store32 { rd, ref mem, .. }
            | &Inst::Store64 { rd, ref mem, .. } => unimplemented!(),
            &Inst::MovZ { rd, .. } => unimplemented!(),
            &Inst::Jump { .. } | &Inst::Call { .. } | &Inst::Ret { .. } => unimplemented!(),
            &Inst::CallInd { rn, .. } => unimplemented!(),
            &Inst::CondBrZ { rt, .. } | &Inst::CondBrNZ { rt, .. } => unimplemented!(),
            &Inst::CondBr { .. } => unimplemented!(),
            &Inst::RegallocMove { dst, src, .. } => {
                panic!("RegallocMove reached binemit!");
            }
        }
    }
}
