//! This module defines the encoding for `MachInst<Op, Arg>`.

use crate::binemit::CodeSink;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::*;
use crate::isa::RegUnit;
use crate::machinst::*;

// ----- ARM64 instruction formats: encoding bit-manipulation helpers. -----

fn encode_reg_reg_reg(top8: u8, bit_23_21: u8, bit_15_10: u8, rm: u8, rn: u8, rd: u8) -> u32 {
    (top8 as u32) << 24
        | (bit_23_21 as u32) << 21
        | (bit_15_10 as u32) << 10
        | (rm as u32) << 16
        | (rn as u32) << 5
        | (rd as u32)
}

fn encode_reg_reg_imm12(top8: u8, bit_23_22: u8, imm12: u16, rn: u8, rd: u8) -> u32 {
    (top8 as u32) << 24
        | (bit_23_22 as u32) << 22
        | (imm12 as u32) << 10
        | (rn as u32) << 5
        | (rd as u32)
}

// ----- Convert a RegUnit to a physical register number in one of the regfiles. -----

fn regunit_to_gpr(ru: RegUnit) -> u8 {
    let idx = ru - GPR_DATA.first;
    assert!(idx < 32);
    idx as u8
}

fn regunit_to_fpr(ru: RegUnit) -> u8 {
    let idx = ru - FPR_DATA.first;
    assert!(idx < 32);
    idx as u8
}

// ----- Helper: encode an arithmetic instruction that accepts a Reg, ShiftedReg, or ExtendedReg.

fn arith_rrr(top8: u8, inst: &MachInst<Op, Arg>, locs: &MachLocations) -> u32 {
    match (&inst.args[0], &inst.args[1], &inst.args[2]) {
        (&Arg::Reg(ref rd), &Arg::Reg(ref rn), &Arg::Reg(ref rm)) => {
            let rd = regunit_to_gpr(*locs.get(rd).unwrap());
            let rn = regunit_to_gpr(*locs.get(rn).unwrap());
            let rm = regunit_to_gpr(*locs.get(rm).unwrap());
            encode_reg_reg_reg(top8, 0b000, 0, rm, rn, rd)
        }
        (&Arg::Reg(ref rd), &Arg::Reg(ref rn), &Arg::ShiftedReg(ref rm, ref shiftop, shiftamt)) => {
            let rd = regunit_to_gpr(*locs.get(rd).unwrap());
            let rn = regunit_to_gpr(*locs.get(rn).unwrap());
            let rm = regunit_to_gpr(*locs.get(rm).unwrap());
            encode_reg_reg_reg(
                top8,
                shiftop.bits() << 1,
                (shiftamt & 0x1f) as u8,
                rm,
                rn,
                rd,
            )
        }
        (
            &Arg::Reg(ref rd),
            &Arg::Reg(ref rn),
            &Arg::ExtendedReg(ref rm, ref extendop, shiftamt),
        ) => {
            let rd = regunit_to_gpr(*locs.get(rd).unwrap());
            let rn = regunit_to_gpr(*locs.get(rn).unwrap());
            let rm = regunit_to_gpr(*locs.get(rm).unwrap());
            encode_reg_reg_reg(
                top8,
                0b001,
                (extendop.bits() << 3) | ((shiftamt & 0x07) as u8),
                rm,
                rn,
                rd,
            )
        }
        _ => unimplemented!(),
    }
}

fn arith_rr_imm12(top8: u8, inst: &MachInst<Op, Arg>, locs: &MachLocations) -> u32 {
    match (&inst.args[0], &inst.args[1], &inst.args[2]) {
        (&Arg::Reg(ref rd), &Arg::Reg(ref rn), &Arg::Imm(ref imm)) => {
            let rd = regunit_to_gpr(*locs.get(rd).unwrap());
            let rn = regunit_to_gpr(*locs.get(rn).unwrap());
            encode_reg_reg_imm12(top8, imm.shift_bits(), (imm.bits & 0xfff) as u16, rn, rd)
        }
        _ => unimplemented!(),
    }
}

// ----- Main dispatch on opcode. -----

impl<CS: CodeSink> MachInstEncode<Op, Arg, CS> for MachInst<Op, Arg> {
    fn size(&self, locs: &MachLocations) -> usize {
        4 // RISC! Every (non-THUMB) instruction on ARM is 4 bytes.
    }

    fn encode(&self, locs: &MachLocations, sink: &mut CS) {
        match self.op {
            Op::Add32 => {
                sink.put4(arith_rrr(0b000_01011, self, locs));
            }
            Op::Add64 => {
                sink.put4(arith_rrr(0b100_01011, self, locs));
            }
            Op::AddI32 => {
                sink.put4(arith_rr_imm12(0b000_10001, self, locs));
            }
            Op::AddI64 => {
                sink.put4(arith_rr_imm12(0b100_10001, self, locs));
            }
            Op::Sub32 => {
                sink.put4(arith_rrr(0b010_01011, self, locs));
            }
            Op::Sub64 => {
                sink.put4(arith_rrr(0b110_01011, self, locs));
            }
            Op::SubI32 => {
                sink.put4(arith_rr_imm12(0b010_10001, self, locs));
            }
            Op::SubI64 => {
                sink.put4(arith_rr_imm12(0b110_10001, self, locs));
            }
            _ => unimplemented!(),
        }
    }
}
