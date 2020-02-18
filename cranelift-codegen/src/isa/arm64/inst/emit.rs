//! ARM64 ISA: binary code emission.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::binemit::{CodeOffset, CodeSink, ConstantPoolSink, NullConstantPoolSink, Reloc};
use crate::ir::constant::{ConstantData, ConstantOffset};
use crate::ir::types::*;
use crate::ir::Type;
use crate::isa::arm64::inst::*;
use crate::machinst::*;

use regalloc::{
    RealReg, RealRegUniverse, Reg, RegClass, RegClassInfo, SpillSlot, VirtualReg, Writable,
    NUM_REG_CLASSES,
};

use alloc::vec::Vec;

/// Memory addressing mode finalization: convert "special" modes (e.g.,
/// generic arbitrary stack offset) into real addressing modes, possibly by
/// emitting some helper instructions that come immediately before the use
/// of this amod.
pub fn mem_finalize<CPS: ConstantPoolSink>(
    insn_off: CodeOffset,
    mem: &MemArg,
    consts: &mut CPS,
) -> (Vec<Inst>, MemArg) {
    match mem {
        &MemArg::StackOffset(fp_offset) => {
            if let Some(simm9) = SImm9::maybe_from_i64(fp_offset) {
                let mem = MemArg::BaseSImm9(fp_reg(), simm9);
                (vec![], mem)
            } else {
                let tmp = writable_spilltmp_reg();
                let const_data = u64_constant(fp_offset as u64);
                let (_, const_mem) = mem_finalize(
                    insn_off,
                    &MemArg::label(MemLabel::ConstantData(const_data)),
                    consts,
                );
                let const_inst = Inst::ULoad64 {
                    rd: tmp,
                    mem: const_mem,
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
        &MemArg::Label(MemLabel::ConstantData(ref data)) => {
            let len = data.len();
            let alignment = if len <= 4 {
                4
            } else if len <= 8 {
                8
            } else {
                16
            };
            consts.align_to(alignment);
            let off = consts.get_offset_from_code_start();
            consts.add_data(data.iter().as_slice());
            let rel_off = off - insn_off;
            (vec![], MemArg::Label(MemLabel::ConstantPoolRel(rel_off)))
        }
        _ => (vec![], mem.clone()),
    }
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

//=============================================================================
// Instructions and subcomponents: emission

fn machreg_to_gpr(m: Reg) -> u32 {
    assert!(m.is_real());
    m.to_real_reg().get_hw_encoding() as u32
}

fn enc_arith_rrr(bits_31_21: u16, bits_15_10: u8, rd: Writable<Reg>, rn: Reg, rm: Reg) -> u32 {
    ((bits_31_21 as u32) << 21)
        | ((bits_15_10 as u32) << 10)
        | machreg_to_gpr(rd.to_reg())
        | (machreg_to_gpr(rn) << 5)
        | (machreg_to_gpr(rm) << 16)
}

fn enc_arith_rr_imm12(bits_31_24: u8, immshift: u8, imm12: u16, rn: Reg, rd: Writable<Reg>) -> u32 {
    ((bits_31_24 as u32) << 24)
        | ((immshift as u32) << 22)
        | ((imm12 as u32) << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd.to_reg())
}

fn enc_arith_rr_imml(bits_31_23: u16, imm_bits: u16, rn: Reg, rd: Writable<Reg>) -> u32 {
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

fn enc_move_wide(op: MoveWideOpcode, rd: Writable<Reg>, imm: MoveWideConst) -> u32 {
    assert!(imm.shift <= 0b11);
    MOVE_WIDE_FIXED
        | (op as u32) << 29
        | (imm.shift as u32) << 21
        | (imm.bits as u32) << 5
        | machreg_to_gpr(rd.to_reg())
}

fn enc_ldst_pair(op_31_22: u32, simm7: SImm7Scaled, rn: Reg, rt: Reg, rt2: Reg) -> u32 {
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

fn enc_extend(top22: u32, rd: Writable<Reg>, rn: Reg) -> u32 {
    (top22 << 10) | (machreg_to_gpr(rn) << 5) | machreg_to_gpr(rd.to_reg())
}

impl<CS: CodeSink, CPS: ConstantPoolSink> MachInstEmit<CS, CPS> for Inst {
    fn emit(&self, sink: &mut CS, consts: &mut CPS) {
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
                    ALUOp::AddS32 => 0b00101011_000,
                    ALUOp::AddS64 => 0b10101011_000,
                    ALUOp::SubS32 => 0b01101011_000,
                    ALUOp::SubS64 => 0b11101011_000,
                    ALUOp::MAdd32 | ALUOp::MAdd64 => {
                        // multiply-add is of form RRRR (three-source).
                        panic!("Bad ALUOp in RRR form!");
                    }
                };
                sink.put4(enc_arith_rrr(top11, 0b000_000, rd, rn, rm));
            }
            &Inst::AluRRRR { .. } => {
                // TODO.
                unimplemented!();
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
                    ALUOp::AddS32 => 0b001_10001,
                    ALUOp::AddS64 => 0b101_10001,
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
                alu_op,
                rd,
                rn,
                rm,
                ref shiftop,
            } => {
                let top11: u16 = match alu_op {
                    ALUOp::Add32 => 0b000_01011000,
                    ALUOp::Add64 => 0b100_01011000,
                    ALUOp::AddS32 => 0b001_01011000,
                    ALUOp::AddS64 => 0b101_01011000,
                    ALUOp::Sub32 => 0b010_01011000,
                    ALUOp::Sub64 => 0b110_01011000,
                    ALUOp::SubS32 => 0b011_01011000,
                    ALUOp::SubS64 => 0b111_01011000,
                    _ => unimplemented!(),
                };
                let top11 = top11 | ((shiftop.op().bits() as u16) << 1);
                let bits_15_10 = shiftop.amt().value();
                sink.put4(enc_arith_rrr(top11, bits_15_10, rd, rn, rm));
            }

            &Inst::AluRRRExtend {
                alu_op,
                rd,
                rn,
                rm,
                extendop,
            } => {
                let top11 = match alu_op {
                    ALUOp::Add32 => 0b00001011001,
                    ALUOp::Add64 => 0b10001011001,
                    ALUOp::Sub32 => 0b01001011001,
                    ALUOp::Sub64 => 0b11001011001,
                    ALUOp::AddS32 => 0b00101011001,
                    ALUOp::AddS64 => 0b10101011001,
                    ALUOp::SubS32 => 0b01101011001,
                    ALUOp::SubS64 => 0b11101011001,
                    _ => unimplemented!(),
                };
                let bits_15_10 = extendop.bits() << 3;
                sink.put4(enc_arith_rrr(top11, bits_15_10, rd, rn, rm));
            }

            &Inst::ULoad8 { rd, ref mem }
            | &Inst::SLoad8 { rd, ref mem }
            | &Inst::ULoad16 { rd, ref mem }
            | &Inst::SLoad16 { rd, ref mem }
            | &Inst::ULoad32 { rd, ref mem }
            | &Inst::SLoad32 { rd, ref mem }
            | &Inst::ULoad64 { rd, ref mem } => {
                let (mem_insts, mem) = mem_finalize(sink.offset(), mem, consts);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink, consts);
                }

                // ldst encoding helpers take Reg, not Writable<Reg>.
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
                            &MemLabel::ConstantPoolRel(off) => off,
                            // Should be converted by `mem_finalize()` into `ConstantPool`.
                            &MemLabel::ConstantData(..) => {
                                panic!("Should not see ConstantData here!")
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
                let (mem_insts, mem) = mem_finalize(sink.offset(), mem, consts);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink, consts);
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
                &PairMemArg::SignedOffset(reg, simm7) => {
                    assert_eq!(simm7.scale_ty, I64);
                    sink.put4(enc_ldst_pair(0b1010100100, simm7, reg, rt, rt2));
                }
                &PairMemArg::PreIndexed(reg, simm7) => {
                    assert_eq!(simm7.scale_ty, I64);
                    sink.put4(enc_ldst_pair(0b1010100110, simm7, reg.to_reg(), rt, rt2));
                }
                &PairMemArg::PostIndexed(reg, simm7) => {
                    assert_eq!(simm7.scale_ty, I64);
                    sink.put4(enc_ldst_pair(0b1010100010, simm7, reg.to_reg(), rt, rt2));
                }
            },
            &Inst::LoadP64 { rt, rt2, ref mem } => {
                let rt = rt.to_reg();
                let rt2 = rt2.to_reg();
                match mem {
                    &PairMemArg::SignedOffset(reg, simm7) => {
                        assert_eq!(simm7.scale_ty, I64);
                        sink.put4(enc_ldst_pair(0b1010100101, simm7, reg, rt, rt2));
                    }
                    &PairMemArg::PreIndexed(reg, simm7) => {
                        assert_eq!(simm7.scale_ty, I64);
                        sink.put4(enc_ldst_pair(0b1010100111, simm7, reg.to_reg(), rt, rt2));
                    }
                    &PairMemArg::PostIndexed(reg, simm7) => {
                        assert_eq!(simm7.scale_ty, I64);
                        sink.put4(enc_ldst_pair(0b1010100011, simm7, reg.to_reg(), rt, rt2));
                    }
                }
            }
            &Inst::Mov { rd, rm } => {
                // Encoded as ORR rd, rm, zero.
                sink.put4(enc_arith_rrr(0b10101010_000, 0b000_000, rd, zero_reg(), rm));
            }
            &Inst::Mov32 { rd, rm } => {
                // Encoded as ORR rd, rm, zero.
                sink.put4(enc_arith_rrr(0b00101010_000, 0b000_000, rd, zero_reg(), rm));
            }
            &Inst::MovZ { rd, imm } => sink.put4(enc_move_wide(MoveWideOpcode::MOVZ, rd, imm)),
            &Inst::MovN { rd, imm } => sink.put4(enc_move_wide(MoveWideOpcode::MOVN, rd, imm)),
            &Inst::Extend {
                rd,
                rn,
                signed,
                from_bits,
                to_bits,
            } => {
                let top22 = match (signed, from_bits, to_bits) {
                    (false, 8, 32) => 0b010_100110_0_000000_000111, // UXTB (32)
                    (false, 16, 32) => 0b010_100110_0_000000_001111, // UXTH (32)
                    (true, 8, 32) => 0b000_100110_0_000000_000111,  // SXTB (32)
                    (true, 16, 32) => 0b000_100110_0_000000_001111, // SXTH (32)
                    // The 64-bit unsigned variants are the same as the 32-bit ones,
                    // because writes to Wn zero out the top 32 bits of Xn
                    (false, 8, 64) => 0b010_100110_0_000000_000111, // UXTB (64)
                    (false, 16, 64) => 0b010_100110_0_000000_001111, // UXTH (64)
                    (true, 8, 64) => 0b100_100110_1_000000_000111,  // SXTB (64)
                    (true, 16, 64) => 0b100_100110_1_000000_001111, // SXTH (64)
                    // 32-to-64: the unsigned case is a 'mov' (special-cased below).
                    (false, 32, 64) => 0,                           // MOV
                    (true, 32, 64) => 0b100_100110_1_000000_011111, // SXTW (64)
                    _ => panic!(
                        "Unsupported extend combination: signed = {}, from_bits = {}, to_bits = {}",
                        signed, from_bits, to_bits
                    ),
                };
                if top22 != 0 {
                    sink.put4(enc_extend(top22, rd, rn));
                } else {
                    Inst::mov32(rd, rn).emit(sink, consts);
                }
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
            &Inst::Call { ref dest, .. } => {
                sink.reloc_external(Reloc::Arm64Call, dest, 0);
                sink.put4(enc_jump26(0b100101, 0));
            }
            &Inst::CallInd { rn, .. } => {
                sink.put4(0b1101011_0001_11111_000000_00000_00000 | (machreg_to_gpr(rn) << 5));
            }
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
            Inst::AluRRR {
                alu_op: ALUOp::AddS32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
            },
            "4100032B",
            "adds w1, w2, w3",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::AddS64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40006AB",
            "adds x4, x5, x6",
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

        insns.push((
            Inst::AluRRRExtend {
                alu_op: ALUOp::Add32,
                rd: writable_xreg(7),
                rn: xreg(8),
                rm: xreg(9),
                extendop: ExtendOp::SXTB,
            },
            "0781290B",
            "add w7, w8, w9, SXTB",
        ));

        insns.push((
            Inst::AluRRRExtend {
                alu_op: ALUOp::Add64,
                rd: writable_xreg(15),
                rn: xreg(16),
                rm: xreg(17),
                extendop: ExtendOp::UXTB,
            },
            "0F02318B",
            "add x15, x16, x17, UXTB",
        ));

        insns.push((
            Inst::AluRRRExtend {
                alu_op: ALUOp::Sub32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                extendop: ExtendOp::SXTH,
            },
            "41A0234B",
            "sub w1, w2, w3, SXTH",
        ));

        insns.push((
            Inst::AluRRRExtend {
                alu_op: ALUOp::Sub64,
                rd: writable_xreg(20),
                rn: xreg(21),
                rm: xreg(22),
                extendop: ExtendOp::UXTW,
            },
            "B44236CB",
            "sub x20, x21, x22, UXTW",
        ));

        insns.push((
            Inst::AluRRRShift {
                alu_op: ALUOp::Add64,
                rd: writable_xreg(10),
                rn: xreg(11),
                rm: xreg(12),
                shiftop: ShiftOpAndAmt::new(
                    ShiftOp::ASR,
                    ShiftOpShiftImm::maybe_from_shift(42).unwrap(),
                ),
            },
            "6AA98C8B",
            "add x10, x11, x12, ASR 42",
        ));
        insns.push((
            Inst::AluRRRShift {
                alu_op: ALUOp::Sub32,
                rd: writable_xreg(10),
                rn: xreg(11),
                rm: xreg(12),
                shiftop: ShiftOpAndAmt::new(
                    ShiftOp::LSL,
                    ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
                ),
            },
            "6A5D0C4B",
            "sub w10, w11, w12, LSL 23",
        ));

        // TODO: ImmLogic forms (once logic-immediate encoding/decoding exists).

        // TODO: AluRRRShift forms.

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
                mem: MemArg::Label(MemLabel::ConstantPoolRel(64)),
            },
            "01020058",
            "ldr x1, 64",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Label(MemLabel::ConstantData(u64_constant(0x0123456789abcdef))),
            },
            "01200058EFCDAB8967452301",
            "ldr x1, 1024",
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
            "0F200058EF011D8BE10140F80080000000000000",
            "ldr x15, 1024 ; add x15, x15, fp ; ldur x1, [x15]",
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
                mem: PairMemArg::SignedOffset(xreg(10), SImm7Scaled::zero(I64)),
            },
            "482500A9",
            "stp x8, x9, [x10]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::SignedOffset(
                    xreg(10),
                    SImm7Scaled::maybe_from_i64(504, I64).unwrap(),
                ),
            },
            "48A51FA9",
            "stp x8, x9, [x10, #504]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::SignedOffset(
                    xreg(10),
                    SImm7Scaled::maybe_from_i64(-64, I64).unwrap(),
                ),
            },
            "48253CA9",
            "stp x8, x9, [x10, #-64]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(21),
                rt2: xreg(28),
                mem: PairMemArg::SignedOffset(
                    xreg(1),
                    SImm7Scaled::maybe_from_i64(-512, I64).unwrap(),
                ),
            },
            "357020A9",
            "stp x21, x28, [x1, #-512]",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(8),
                rt2: xreg(9),
                mem: PairMemArg::PreIndexed(
                    writable_xreg(10),
                    SImm7Scaled::maybe_from_i64(-64, I64).unwrap(),
                ),
            },
            "4825BCA9",
            "stp x8, x9, [x10, #-64]!",
        ));
        insns.push((
            Inst::StoreP64 {
                rt: xreg(15),
                rt2: xreg(16),
                mem: PairMemArg::PostIndexed(
                    writable_xreg(20),
                    SImm7Scaled::maybe_from_i64(504, I64).unwrap(),
                ),
            },
            "8FC29FA8",
            "stp x15, x16, [x20], #504",
        ));

        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::SignedOffset(xreg(10), SImm7Scaled::zero(I64)),
            },
            "482540A9",
            "ldp x8, x9, [x10]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::SignedOffset(
                    xreg(10),
                    SImm7Scaled::maybe_from_i64(504, I64).unwrap(),
                ),
            },
            "48A55FA9",
            "ldp x8, x9, [x10, #504]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::SignedOffset(
                    xreg(10),
                    SImm7Scaled::maybe_from_i64(-64, I64).unwrap(),
                ),
            },
            "48257CA9",
            "ldp x8, x9, [x10, #-64]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::SignedOffset(
                    xreg(10),
                    SImm7Scaled::maybe_from_i64(-512, I64).unwrap(),
                ),
            },
            "482560A9",
            "ldp x8, x9, [x10, #-512]",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(9),
                mem: PairMemArg::PreIndexed(
                    writable_xreg(10),
                    SImm7Scaled::maybe_from_i64(-64, I64).unwrap(),
                ),
            },
            "4825FCA9",
            "ldp x8, x9, [x10, #-64]!",
        ));
        insns.push((
            Inst::LoadP64 {
                rt: writable_xreg(8),
                rt2: writable_xreg(25),
                mem: PairMemArg::PostIndexed(
                    writable_xreg(12),
                    SImm7Scaled::maybe_from_i64(504, I64).unwrap(),
                ),
            },
            "88E5DFA8",
            "ldp x8, x25, [x12], #504",
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
            Inst::Mov32 {
                rd: writable_xreg(8),
                rm: xreg(9),
            },
            "E803092A",
            "mov w8, w9",
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
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: false,
                from_bits: 8,
                to_bits: 32,
            },
            "411C0053",
            "uxtb w1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: true,
                from_bits: 8,
                to_bits: 32,
            },
            "411C0013",
            "sxtb w1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: false,
                from_bits: 16,
                to_bits: 32,
            },
            "413C0053",
            "uxth w1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: true,
                from_bits: 16,
                to_bits: 32,
            },
            "413C0013",
            "sxth w1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: false,
                from_bits: 8,
                to_bits: 64,
            },
            "411C0053",
            "uxtb x1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: true,
                from_bits: 8,
                to_bits: 64,
            },
            "411C4093",
            "sxtb x1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: false,
                from_bits: 16,
                to_bits: 64,
            },
            "413C0053",
            "uxth x1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: true,
                from_bits: 16,
                to_bits: 64,
            },
            "413C4093",
            "sxth x1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: false,
                from_bits: 32,
                to_bits: 64,
            },
            "E103022A",
            "mov w1, w2",
        ));
        insns.push((
            Inst::Extend {
                rd: writable_xreg(1),
                rn: xreg(2),
                signed: true,
                from_bits: 32,
                to_bits: 64,
            },
            "417C4093",
            "sxtw x1, w2",
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

        insns.push((
            Inst::Call {
                dest: ExternalName::testcase("test0"),
                uses: Set::empty(),
                defs: Set::empty(),
            },
            "00000094",
            "bl 0",
        ));

        insns.push((
            Inst::CallInd {
                rn: xreg(10),
                uses: Set::empty(),
                defs: Set::empty(),
            },
            "40013FD6",
            "blr x10",
        ));

        let rru = create_reg_universe();
        for (insn, expected_encoding, expected_printing) in insns {
            println!(
                "ARM64: {:?}, {}, {}",
                insn, expected_encoding, expected_printing
            );

            // Check the printed text is as expected.
            let mut consts = VCodeConstantPool::new(1024);
            let actual_printing = insn.show_rru_with_constsink(Some(&rru), &mut consts);
            assert_eq!(expected_printing, actual_printing);

            // Check the encoding is as expected.
            let mut sink = test_utils::TestCodeSink::new();
            let mut consts = VCodeConstantPool::new(1024);
            insn.emit(&mut sink, &mut consts);
            consts.emit(&mut sink);
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
