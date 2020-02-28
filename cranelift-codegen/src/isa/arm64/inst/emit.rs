//! ARM64 ISA: binary code emission.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::binemit::{CodeOffset, CodeSink, Reloc};
use crate::ir::constant::ConstantData;
use crate::ir::types::*;
use crate::ir::Type;
use crate::isa::arm64::inst::*;
use crate::machinst::*;
use cranelift_entity::EntityRef;

use regalloc::{
    RealReg, RealRegUniverse, Reg, RegClass, RegClassInfo, SpillSlot, VirtualReg, Writable,
    NUM_REG_CLASSES,
};

use alloc::vec::Vec;

/// Memory label/reference finalization: convert a MemLabel to a PC-relative
/// offset, possibly emitting relocation(s) as necessary.
pub fn memlabel_finalize<O: MachSectionOutput>(
    insn_off: CodeOffset,
    label: &MemLabel,
    consts: &mut O,
    jt_offsets: &[CodeOffset],
) -> i32 {
    match label {
        &MemLabel::PCRel(rel) => rel,
        &MemLabel::ConstantData(ref data) => {
            let len = data.len();
            let alignment = if len <= 4 {
                4
            } else if len <= 8 {
                8
            } else {
                16
            };
            consts.align_to(alignment);
            let off = consts.cur_offset_from_start();
            consts.put_data(data.iter().as_slice());
            (off as i32) - (insn_off as i32)
        }
        &MemLabel::JumpTable(jt) => {
            let jt_off = if jt.index() < jt_offsets.len() {
                jt_offsets[jt.index()]
            } else {
                // Can happen when invoked from show_rru().
                0
            };
            (jt_off as i32) - (insn_off as i32)
        }
        &MemLabel::CodeOffset(off) => (off as i32) - (insn_off as i32),
        &MemLabel::ExtName(ref name, offset) => {
            consts.align_to(8);
            let off = consts.cur_offset_from_start();
            consts.add_reloc(Reloc::Abs8, name, offset);
            consts.put_data(&[0, 0, 0, 0, 0, 0, 0, 0]);
            (off as i32) - (insn_off as i32)
        }
    }
}

/// Memory addressing mode finalization: convert "special" modes (e.g.,
/// generic arbitrary stack offset) into real addressing modes, possibly by
/// emitting some helper instructions that come immediately before the use
/// of this amod.
pub fn mem_finalize<O: MachSectionOutput>(
    insn_off: CodeOffset,
    mem: &MemArg,
    consts: &mut O,
    jt_offsets: &[CodeOffset],
) -> (Vec<Inst>, MemArg) {
    match mem {
        &MemArg::StackOffset(fp_offset) => {
            if let Some(simm9) = SImm9::maybe_from_i64(fp_offset) {
                let mem = MemArg::Unscaled(fp_reg(), simm9);
                (vec![], mem)
            } else {
                let tmp = writable_spilltmp_reg();
                let const_data = u64_constant(fp_offset as u64);
                let (_, const_mem) = mem_finalize(
                    insn_off,
                    &MemArg::label(MemLabel::ConstantData(const_data)),
                    consts,
                    jt_offsets,
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
                (vec![const_inst, add_inst], MemArg::reg(tmp.to_reg()))
            }
        }
        &MemArg::Label(ref label) => {
            let off: i32 = memlabel_finalize(insn_off, label, consts, jt_offsets);
            (vec![], MemArg::Label(MemLabel::PCRel(off)))
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
    assert!(m.get_class() == RegClass::I64);
    assert!(m.is_real());
    m.to_real_reg().get_hw_encoding() as u32
}

fn machreg_to_vec(m: Reg) -> u32 {
    assert!(m.get_class() == RegClass::V128);
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

fn enc_arith_rrrr(top11: u32, rm: Reg, bit15: u32, ra: Reg, rn: Reg, rd: Writable<Reg>) -> u32 {
    (top11 << 21)
        | (machreg_to_gpr(rm) << 16)
        | (bit15 << 15)
        | (machreg_to_gpr(ra) << 10)
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
    (op_31_22 << 22)
        | (0b1 << 24)
        | (uimm12.bits() << 10)
        | (machreg_to_gpr(rn) << 5)
        | machreg_to_gpr(rd)
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

fn enc_vec_rrr(top11: u32, rm: Reg, bit15_10: u32, rn: Reg, rd: Writable<Reg>) -> u32 {
    (top11 << 21)
        | (machreg_to_vec(rm) << 16)
        | (bit15_10 << 10)
        | (machreg_to_vec(rn) << 5)
        | machreg_to_vec(rd.to_reg())
}

fn enc_bit_rr(size: u32, opcode2: u32, opcode1: u32, rn: Reg, rd: Writable<Reg>) -> u32 {
    (0b01011010110 << 21)
        | size << 31
        | opcode2 << 16
        | opcode1 << 10
        | machreg_to_gpr(rn) << 5
        | machreg_to_gpr(rd.to_reg())
}

fn enc_br(rn: Reg) -> u32 {
    0b1101011_0000_11111_000000_00000_00000 | (machreg_to_gpr(rn) << 5)
}

fn enc_adr(off: i32, rd: Writable<Reg>) -> u32 {
    let off = off as u32;
    let immlo = off & 3;
    let immhi = (off >> 2) & ((1 << 19) - 1);
    (0b00010000 << 24) | (immlo << 29) | (immhi << 5) | machreg_to_gpr(rd.to_reg())
}

impl<O: MachSectionOutput> MachInstEmit<O> for Inst {
    fn emit(&self, sink: &mut O, consts: &mut O, jt_offsets: &[CodeOffset]) {
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
                    ALUOp::Eor32 => 0b01001010_000,
                    ALUOp::Eor64 => 0b11001010_000,
                    ALUOp::OrrNot32 => 0b00101010_001,
                    ALUOp::OrrNot64 => 0b10101010_001,
                    ALUOp::AndNot32 => 0b00001010_001,
                    ALUOp::AndNot64 => 0b10001010_001,
                    ALUOp::EorNot32 => 0b01001010_001,
                    ALUOp::EorNot64 => 0b11001010_001,
                    ALUOp::AddS32 => 0b00101011_000,
                    ALUOp::AddS64 => 0b10101011_000,
                    ALUOp::SubS32 => 0b01101011_000,
                    ALUOp::SubS64 => 0b11101011_000,
                    ALUOp::SDiv64 => 0b10011010_110,
                    ALUOp::UDiv64 => 0b10011010_110,
                    ALUOp::RotR32 | ALUOp::Lsr32 | ALUOp::Asr32 | ALUOp::Lsl32 => 0b00011010_110,
                    ALUOp::RotR64 | ALUOp::Lsr64 | ALUOp::Asr64 | ALUOp::Lsl64 => 0b10011010_110,

                    ALUOp::MAdd32
                    | ALUOp::MAdd64
                    | ALUOp::MSub32
                    | ALUOp::MSub64
                    | ALUOp::SMulH
                    | ALUOp::UMulH => {
                        //// RRRR ops.
                        panic!("Bad ALUOp in RRR form!");
                    }
                };
                let bit15_10 = match alu_op {
                    ALUOp::SDiv64 => 0b000011,
                    ALUOp::UDiv64 => 0b000010,
                    ALUOp::RotR32 | ALUOp::RotR64 => 0b001011,
                    ALUOp::Lsr32 | ALUOp::Lsr64 => 0b001001,
                    ALUOp::Asr32 | ALUOp::Asr64 => 0b001010,
                    ALUOp::Lsl32 | ALUOp::Lsl64 => 0b001000,
                    _ => 0b000000,
                };
                sink.put4(enc_arith_rrr(top11, bit15_10, rd, rn, rm));
            }
            &Inst::AluRRRR {
                alu_op,
                rd,
                rm,
                rn,
                ra,
            } => {
                let (top11, bit15) = match alu_op {
                    ALUOp::MAdd32 => (0b0_00_11011_000, 0),
                    ALUOp::MSub32 => (0b0_00_11011_000, 1),
                    ALUOp::MAdd64 => (0b1_00_11011_000, 0),
                    ALUOp::MSub64 => (0b1_00_11011_000, 1),
                    ALUOp::SMulH => (0b1_00_11011_010, 0),
                    ALUOp::UMulH => (0b1_00_11011_110, 0),
                    _ => unimplemented!(),
                };
                sink.put4(enc_arith_rrrr(top11, rm, bit15, ra, rn, rd));
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
                let (top9, inv) = match alu_op {
                    ALUOp::Orr32 => (0b001_100100, false),
                    ALUOp::Orr64 => (0b101_100100, false),
                    ALUOp::And32 => (0b000_100100, false),
                    ALUOp::And64 => (0b100_100100, false),
                    ALUOp::Eor32 => (0b010_100100, false),
                    ALUOp::Eor64 => (0b110_100100, false),
                    ALUOp::OrrNot32 => (0b001_100100, true),
                    ALUOp::OrrNot64 => (0b101_100100, true),
                    ALUOp::AndNot32 => (0b000_100100, true),
                    ALUOp::AndNot64 => (0b100_100100, true),
                    ALUOp::EorNot32 => (0b010_100100, true),
                    ALUOp::EorNot64 => (0b110_100100, true),
                    _ => unimplemented!(),
                };
                let imml = if inv { imml.invert() } else { imml.clone() };
                sink.put4(enc_arith_rr_imml(top9, imml.enc_bits(), rn, rd));
            }

            &Inst::AluRRImmShift {
                alu_op,
                rd,
                rn,
                ref immshift,
            } => {
                let amt = immshift.value();
                let (top10, immr, imms) = match alu_op {
                    ALUOp::RotR32 => (0b0001001110, machreg_to_gpr(rn), amt as u32),
                    ALUOp::RotR64 => (0b1001001111, machreg_to_gpr(rn), amt as u32),
                    ALUOp::Lsr32 => (0b0101001100, amt as u32, 0b011111),
                    ALUOp::Lsr64 => (0b1101001101, amt as u32, 0b111111),
                    ALUOp::Asr32 => (0b0001001100, amt as u32, 0b011111),
                    ALUOp::Asr64 => (0b1001001101, amt as u32, 0b111111),
                    ALUOp::Lsl32 => (0b0101001100, (32 - amt) as u32, (31 - amt) as u32),
                    ALUOp::Lsl64 => (0b1101001101, (64 - amt) as u32, (63 - amt) as u32),
                    _ => unimplemented!(),
                };
                sink.put4(
                    (top10 << 22)
                        | (immr << 16)
                        | (imms << 10)
                        | (machreg_to_gpr(rn) << 5)
                        | machreg_to_gpr(rd.to_reg()),
                );
            }

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

            &Inst::BitRR { op, rd, rn, .. } => {
                let size = if op.is_32_bit() { 0b0 } else { 0b1 };
                let (op1, op2) = match op {
                    BitOp::RBit32 | BitOp::RBit64 => (0b00000, 0b000000),
                    BitOp::Clz32 | BitOp::Clz64 => (0b00000, 0b000100),
                    BitOp::Cls32 | BitOp::Cls64 => (0b00000, 0b000101),
                };
                sink.put4(enc_bit_rr(size, op1, op2, rn, rd))
            }

            &Inst::ULoad8 { rd, ref mem }
            | &Inst::SLoad8 { rd, ref mem }
            | &Inst::ULoad16 { rd, ref mem }
            | &Inst::SLoad16 { rd, ref mem }
            | &Inst::ULoad32 { rd, ref mem }
            | &Inst::SLoad32 { rd, ref mem }
            | &Inst::ULoad64 { rd, ref mem } => {
                let (mem_insts, mem) =
                    mem_finalize(sink.cur_offset_from_start(), mem, consts, jt_offsets);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink, consts, jt_offsets);
                }

                // ldst encoding helpers take Reg, not Writable<Reg>.
                let rd = rd.to_reg();

                // This is the base opcode (top 10 bits) for the "unscaled
                // immediate" form (Unscaled). Other addressing modes will OR in
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
                    &MemArg::Unscaled(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b00, reg, rd));
                    }
                    &MemArg::UnsignedOffset(reg, uimm12scaled) => {
                        sink.put4(enc_ldst_uimm12(op, uimm12scaled, reg, rd));
                    }
                    &MemArg::RegScaled(r1, r2, ty, scaled) => {
                        match (ty, self) {
                            (I8, &Inst::ULoad8 { .. }) => {}
                            (I8, &Inst::SLoad8 { .. }) => {}
                            (I16, &Inst::ULoad16 { .. }) => {}
                            (I16, &Inst::SLoad16 { .. }) => {}
                            (I32, &Inst::ULoad32 { .. }) => {}
                            (I32, &Inst::SLoad32 { .. }) => {}
                            (I64, &Inst::ULoad64 { .. }) => {}
                            _ => panic!("Mismatching reg-scaling type in MemArg"),
                        }
                        sink.put4(enc_ldst_reg(op, r1, r2, scaled, rd));
                    }
                    &MemArg::Label(ref label) => {
                        let offset = match label {
                            &MemLabel::PCRel(off) => {
                                if off < 0 {
                                    // Happens only before computing final section
                                    // offsets.
                                    assert!(consts.start_offset() == 0);
                                    0
                                } else {
                                    off as u32
                                }
                            }
                            _ => panic!("Unlowered MemLabel at emission: {:?}", label),
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
                let (mem_insts, mem) =
                    mem_finalize(sink.cur_offset_from_start(), mem, consts, jt_offsets);

                for inst in mem_insts.into_iter() {
                    inst.emit(sink, consts, jt_offsets);
                }

                let op = match self {
                    &Inst::Store8 { .. } => 0b0011100000,
                    &Inst::Store16 { .. } => 0b0111100000,
                    &Inst::Store32 { .. } => 0b1011100000,
                    &Inst::Store64 { .. } => 0b1111100000,
                    _ => unreachable!(),
                };
                match &mem {
                    &MemArg::Unscaled(reg, simm9) => {
                        sink.put4(enc_ldst_simm9(op, simm9, 0b00, reg, rd));
                    }
                    &MemArg::UnsignedOffset(reg, uimm12scaled) => {
                        sink.put4(enc_ldst_uimm12(op, uimm12scaled, reg, rd));
                    }
                    &MemArg::RegScaled(r1, r2, _ty, scaled) => {
                        sink.put4(enc_ldst_reg(op, r1, r2, scaled, rd));
                    }
                    &MemArg::Label(..) => {
                        panic!("Store to a MemLabel not implemented!");
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
            &Inst::MovToVec64 { rd, rn } => {
                sink.put4(
                    0b010_01110000_01000_0_0011_1_00000_00000
                        | (machreg_to_gpr(rn) << 5)
                        | machreg_to_vec(rd.to_reg()),
                );
            }
            &Inst::MovFromVec64 { rd, rn } => {
                sink.put4(
                    0b010_01110000_01000_0_0111_1_00000_00000
                        | (machreg_to_vec(rn) << 5)
                        | machreg_to_gpr(rd.to_reg()),
                );
            }
            &Inst::VecRRR { rd, rn, rm, alu_op } => {
                let (top11, bit15_10) = match alu_op {
                    VecALUOp::SQAddScalar => (0b010_11110_11_1, 0b000011),
                    VecALUOp::SQSubScalar => (0b010_11110_11_1, 0b001011),
                    VecALUOp::UQAddScalar => (0b011_11110_11_1, 0b000011),
                    VecALUOp::UQSubScalar => (0b011_11110_11_1, 0b001011),
                };
                sink.put4(enc_vec_rrr(top11, rm, bit15_10, rn, rd));
            }
            &Inst::MovToNZCV { rn } => {
                sink.put4(0xd51b4200 | machreg_to_gpr(rn));
            }
            &Inst::MovFromNZCV { rd } => {
                sink.put4(0xd53b4200 | machreg_to_gpr(rd.to_reg()));
            }
            &Inst::CondSet { rd, cond } => {
                sink.put4(
                    0b100_11010100_11111_0000_01_11111_00000
                        | (cond.invert().bits() << 12)
                        | machreg_to_gpr(rd.to_reg()),
                );
            }
            &Inst::Extend {
                rd,
                rn,
                signed,
                from_bits,
                to_bits,
            } if from_bits >= 8 => {
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
                    Inst::mov32(rd, rn).emit(sink, consts, jt_offsets);
                }
            }
            &Inst::Extend {
                rd,
                rn,
                signed,
                from_bits,
                to_bits,
            } if from_bits == 1 && signed => {
                assert!(to_bits <= 64);
                // Reduce sign-extend-from-1-bit to:
                // - and rd, rn, #1
                // - sub rd, zr, rd

                // We don't have ImmLogic yet, so we just hardcode this. FIXME.
                sink.put4(0x92400000 | (machreg_to_gpr(rn) << 5) | machreg_to_gpr(rd.to_reg()));
                let sub_inst = Inst::AluRRR {
                    alu_op: ALUOp::Sub64,
                    rd,
                    rn: zero_reg(),
                    rm: rd.to_reg(),
                };
                sub_inst.emit(sink, consts, jt_offsets);
            }
            &Inst::Extend {
                rd,
                rn,
                signed,
                from_bits,
                to_bits,
            } if from_bits == 1 && !signed => {
                assert!(to_bits <= 64);
                // Reduce zero-extend-from-1-bit to:
                // - and rd, rn, #1

                // We don't have ImmLogic yet, so we just hardcode this. FIXME.
                sink.put4(0x92400000 | (machreg_to_gpr(rn) << 5) | machreg_to_gpr(rd.to_reg()));
            }
            &Inst::Extend { .. } => {
                panic!("Unsupported extend variant");
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
            &Inst::EpiloguePlaceholder {} => {
                // Noop; this is just a placeholder for epilogues.
            }
            &Inst::Call { ref dest, .. } => {
                sink.add_reloc(Reloc::Arm64Call, dest, 0);
                sink.put4(enc_jump26(0b100101, 0));
            }
            &Inst::CallInd { rn, .. } => {
                sink.put4(0b1101011_0001_11111_000000_00000_00000 | (machreg_to_gpr(rn) << 5));
            }
            &Inst::CondBr { .. } => panic!("Unlowered CondBr during binemit!"),
            &Inst::CondBrLowered { target, kind } => match kind {
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
            },
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
            &Inst::IndirectBr { rn, .. } => {
                sink.put4(enc_br(rn));
            }
            &Inst::Nop => {}
            &Inst::Nop4 => {
                sink.put4(0xd503201f);
            }
            &Inst::Brk { trap_info } => {
                if let Some((srcloc, code)) = trap_info {
                    sink.add_trap(srcloc, code);
                }
                sink.put4(0xd4200000);
            }
            &Inst::Adr { rd, ref label } => {
                let off =
                    memlabel_finalize(sink.cur_offset_from_start(), label, consts, jt_offsets);
                // TODO: support larger offsets with ADRP / ADR pair.
                assert!(off > -(1 << 20));
                assert!(off < (1 << 20));
                sink.put4(enc_adr(off, rd));
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
            Inst::AluRRR {
                alu_op: ALUOp::SDiv64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40CC69A",
            "sdiv x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::UDiv64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A408C69A",
            "udiv x4, x5, x6",
        ));

        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Eor32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400064A",
            "eor w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Eor64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40006CA",
            "eor x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::AndNot32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400260A",
            "bic w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::AndNot64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400268A",
            "bic x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::OrrNot32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400262A",
            "orn w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::OrrNot64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40026AA",
            "orn x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::EorNot32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A400264A",
            "eon w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::EorNot64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A40026CA",
            "eon x4, x5, x6",
        ));

        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::RotR32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A42CC61A",
            "ror w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::RotR64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A42CC69A",
            "ror x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Lsr32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A424C61A",
            "lsr w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Lsr64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A424C69A",
            "lsr x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Asr32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A428C61A",
            "asr w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Asr64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A428C69A",
            "asr x4, x5, x6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Lsl32,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A420C61A",
            "lsl w4, w5, w6",
        ));
        insns.push((
            Inst::AluRRR {
                alu_op: ALUOp::Lsl64,
                rd: writable_xreg(4),
                rn: xreg(5),
                rm: xreg(6),
            },
            "A420C69A",
            "lsl x4, x5, x6",
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

        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::MAdd32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: xreg(4),
            },
            "4110031B",
            "madd w1, w2, w3, w4",
        ));
        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::MAdd64,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: xreg(4),
            },
            "4110039B",
            "madd x1, x2, x3, x4",
        ));
        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::MSub32,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: xreg(4),
            },
            "4190031B",
            "msub w1, w2, w3, w4",
        ));
        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::MSub64,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: xreg(4),
            },
            "4190039B",
            "msub x1, x2, x3, x4",
        ));
        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::SMulH,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: zero_reg(),
            },
            "417C439B",
            "smulh x1, x2, x3",
        ));
        insns.push((
            Inst::AluRRRR {
                alu_op: ALUOp::UMulH,
                rd: writable_xreg(1),
                rn: xreg(2),
                rm: xreg(3),
                ra: zero_reg(),
            },
            "417CC39B",
            "umulh x1, x2, x3",
        ));

        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::RotR32,
                rd: writable_xreg(20),
                rn: xreg(21),
                immshift: ImmShift::maybe_from_u64(19).unwrap(),
            },
            "B44E9513",
            "ror w20, w21, #19",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::RotR64,
                rd: writable_xreg(20),
                rn: xreg(21),
                immshift: ImmShift::maybe_from_u64(42).unwrap(),
            },
            "B4AAD593",
            "ror x20, x21, #42",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Lsr32,
                rd: writable_xreg(10),
                rn: xreg(11),
                immshift: ImmShift::maybe_from_u64(13).unwrap(),
            },
            "6A7D0D53",
            "lsr w10, w11, #13",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Lsr64,
                rd: writable_xreg(10),
                rn: xreg(11),
                immshift: ImmShift::maybe_from_u64(57).unwrap(),
            },
            "6AFD79D3",
            "lsr x10, x11, #57",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Asr32,
                rd: writable_xreg(4),
                rn: xreg(5),
                immshift: ImmShift::maybe_from_u64(7).unwrap(),
            },
            "A47C0713",
            "asr w4, w5, #7",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Asr64,
                rd: writable_xreg(4),
                rn: xreg(5),
                immshift: ImmShift::maybe_from_u64(35).unwrap(),
            },
            "A4FC6393",
            "asr x4, x5, #35",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Lsl32,
                rd: writable_xreg(8),
                rn: xreg(9),
                immshift: ImmShift::maybe_from_u64(24).unwrap(),
            },
            "281D0853",
            "lsl w8, w9, #24",
        ));
        insns.push((
            Inst::AluRRImmShift {
                alu_op: ALUOp::Lsl64,
                rd: writable_xreg(8),
                rn: xreg(9),
                immshift: ImmShift::maybe_from_u64(63).unwrap(),
            },
            "280141D3",
            "lsl x8, x9, #63",
        ));

        // TODO: ImmLogic forms (once logic-immediate encoding/decoding exists).

        insns.push((
            Inst::BitRR {
                op: BitOp::RBit32,
                rd: writable_xreg(1),
                rn: xreg(10),
            },
            "4101C05A",
            "rbit w1, w10",
        ));

        insns.push((
            Inst::BitRR {
                op: BitOp::RBit64,
                rd: writable_xreg(1),
                rn: xreg(10),
            },
            "4101C0DA",
            "rbit x1, x10",
        ));

        insns.push((
            Inst::BitRR {
                op: BitOp::Clz32,
                rd: writable_xreg(15),
                rn: xreg(3),
            },
            "6F10C05A",
            "clz w15, w3",
        ));

        insns.push((
            Inst::BitRR {
                op: BitOp::Clz64,
                rd: writable_xreg(15),
                rn: xreg(3),
            },
            "6F10C0DA",
            "clz x15, x3",
        ));

        insns.push((
            Inst::BitRR {
                op: BitOp::Cls32,
                rd: writable_xreg(21),
                rn: xreg(16),
            },
            "1516C05A",
            "cls w21, w16",
        ));

        insns.push((
            Inst::BitRR {
                op: BitOp::Cls64,
                rd: writable_xreg(21),
                rn: xreg(16),
            },
            "1516C0DA",
            "cls x21, x16",
        ));

        insns.push((
            Inst::ULoad8 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "41004038",
            "ldurb w1, [x2]",
        ));
        insns.push((
            Inst::ULoad8 {
                rd: writable_xreg(1),
                mem: MemArg::UnsignedOffset(xreg(2), UImm12Scaled::zero(I8)),
            },
            "41004039",
            "ldrb w1, [x2]",
        ));
        insns.push((
            Inst::ULoad8 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(5), I8, false),
            },
            "41686538",
            "ldrb w1, [x2, x5]",
        ));
        insns.push((
            Inst::SLoad8 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "41008038",
            "ldursb x1, [x2]",
        ));
        insns.push((
            Inst::SLoad8 {
                rd: writable_xreg(1),
                mem: MemArg::UnsignedOffset(xreg(2), UImm12Scaled::maybe_from_i64(63, I8).unwrap()),
            },
            "41FC8039",
            "ldrsb x1, [x2, #63]",
        ));
        insns.push((
            Inst::SLoad8 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(5), I8, false),
            },
            "4168A538",
            "ldrsb x1, [x2, x5]",
        ));
        insns.push((
            Inst::ULoad16 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::maybe_from_i64(5).unwrap()),
            },
            "41504078",
            "ldurh w1, [x2, #5]",
        ));
        insns.push((
            Inst::ULoad16 {
                rd: writable_xreg(1),
                mem: MemArg::UnsignedOffset(xreg(2), UImm12Scaled::maybe_from_i64(8, I16).unwrap()),
            },
            "41104079",
            "ldrh w1, [x2, #8]",
        ));
        insns.push((
            Inst::ULoad16 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(3), I16, true),
            },
            "41786378",
            "ldrh w1, [x2, x3, lsl #1]",
        ));
        insns.push((
            Inst::SLoad16 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "41008078",
            "ldursh x1, [x2]",
        ));
        insns.push((
            Inst::SLoad16 {
                rd: writable_xreg(28),
                mem: MemArg::UnsignedOffset(
                    xreg(20),
                    UImm12Scaled::maybe_from_i64(24, I16).unwrap(),
                ),
            },
            "9C328079",
            "ldrsh x28, [x20, #24]",
        ));
        insns.push((
            Inst::SLoad16 {
                rd: writable_xreg(28),
                mem: MemArg::RegScaled(xreg(20), xreg(20), I16, true),
            },
            "9C7AB478",
            "ldrsh x28, [x20, x20, lsl #1]",
        ));
        insns.push((
            Inst::ULoad32 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "410040B8",
            "ldur w1, [x2]",
        ));
        insns.push((
            Inst::ULoad32 {
                rd: writable_xreg(12),
                mem: MemArg::UnsignedOffset(
                    xreg(0),
                    UImm12Scaled::maybe_from_i64(204, I32).unwrap(),
                ),
            },
            "0CCC40B9",
            "ldr w12, [x0, #204]",
        ));
        insns.push((
            Inst::ULoad32 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(12), I32, true),
            },
            "41786CB8",
            "ldr w1, [x2, x12, lsl #2]",
        ));
        insns.push((
            Inst::SLoad32 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "410080B8",
            "ldursw x1, [x2]",
        ));
        insns.push((
            Inst::SLoad32 {
                rd: writable_xreg(12),
                mem: MemArg::UnsignedOffset(
                    xreg(1),
                    UImm12Scaled::maybe_from_i64(16380, I32).unwrap(),
                ),
            },
            "2CFCBFB9",
            "ldrsw x12, [x1, #16380]",
        ));
        insns.push((
            Inst::SLoad32 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(5), xreg(1), I32, true),
            },
            "A178A1B8",
            "ldrsw x1, [x5, x1, lsl #2]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "410040F8",
            "ldur x1, [x2]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::maybe_from_i64(-256).unwrap()),
            },
            "410050F8",
            "ldur x1, [x2, #-256]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::maybe_from_i64(255).unwrap()),
            },
            "41F04FF8",
            "ldur x1, [x2, #255]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::UnsignedOffset(
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
                mem: MemArg::RegScaled(xreg(2), xreg(3), I64, false),
            },
            "416863F8",
            "ldr x1, [x2, x3]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(3), I64, true),
            },
            "417863F8",
            "ldr x1, [x2, x3, lsl #3]",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Label(MemLabel::PCRel(64)),
            },
            "01020058",
            "ldr x1, pc+64",
        ));
        insns.push((
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: MemArg::Label(MemLabel::ConstantData(u64_constant(0x0123456789abcdef))),
            },
            "81000058000000000000000000000000EFCDAB8967452301",
            "ldr x1, pc+0",
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
            "8F000058EF011D8BE10140F9000000000080000000000000",
            "ldr x15, pc+0 ; add x15, x15, fp ; ldr x1, [x15]",
        ));

        insns.push((
            Inst::Store8 {
                rd: xreg(1),
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "41000038",
            "sturb w1, [x2]",
        ));
        insns.push((
            Inst::Store8 {
                rd: xreg(1),
                mem: MemArg::UnsignedOffset(
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
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "41000078",
            "sturh w1, [x2]",
        ));
        insns.push((
            Inst::Store16 {
                rd: xreg(1),
                mem: MemArg::UnsignedOffset(
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
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "410000B8",
            "stur w1, [x2]",
        ));
        insns.push((
            Inst::Store32 {
                rd: xreg(1),
                mem: MemArg::UnsignedOffset(
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
                mem: MemArg::Unscaled(xreg(2), SImm9::zero()),
            },
            "410000F8",
            "stur x1, [x2]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::UnsignedOffset(
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
                mem: MemArg::RegScaled(xreg(2), xreg(3), I64, false),
            },
            "416823F8",
            "str x1, [x2, x3]",
        ));
        insns.push((
            Inst::Store64 {
                rd: xreg(1),
                mem: MemArg::RegScaled(xreg(2), xreg(3), I64, true),
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
            Inst::MovToVec64 {
                rd: writable_vreg(20),
                rn: xreg(21),
            },
            "B41E084E",
            "mov v20.d[0], x21",
        ));
        insns.push((
            Inst::MovFromVec64 {
                rd: writable_xreg(21),
                rn: vreg(20),
            },
            "953E084E",
            "mov x21, v20.d[0]",
        ));
        insns.push((
            Inst::MovToNZCV { rn: xreg(13) },
            "0D421BD5",
            "msr nzcv, x13",
        ));
        insns.push((
            Inst::MovFromNZCV {
                rd: writable_xreg(27),
            },
            "1B423BD5",
            "mrs x27, nzcv",
        ));
        insns.push((
            Inst::CondSet {
                rd: writable_xreg(5),
                cond: Cond::Hi,
            },
            "E5979F9A",
            "cset x5, hi",
        ));
        insns.push((
            Inst::VecRRR {
                rd: writable_vreg(21),
                rn: vreg(22),
                rm: vreg(23),
                alu_op: VecALUOp::UQAddScalar,
            },
            "D50EF77E",
            "uqadd d21, d22, d23",
        ));
        insns.push((
            Inst::VecRRR {
                rd: writable_vreg(21),
                rn: vreg(22),
                rm: vreg(23),
                alu_op: VecALUOp::SQAddScalar,
            },
            "D50EF75E",
            "sqadd d21, d22, d23",
        ));
        insns.push((
            Inst::VecRRR {
                rd: writable_vreg(21),
                rn: vreg(22),
                rm: vreg(23),
                alu_op: VecALUOp::UQSubScalar,
            },
            "D52EF77E",
            "uqsub d21, d22, d23",
        ));
        insns.push((
            Inst::VecRRR {
                rd: writable_vreg(21),
                rn: vreg(22),
                rm: vreg(23),
                alu_op: VecALUOp::SQSubScalar,
            },
            "D52EF75E",
            "sqsub d21, d22, d23",
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
                dest: BranchTarget::ResolvedOffset(64),
            },
            "10000014",
            "b 64",
        ));

        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Zero(xreg(8)),
            },
            "080200B4",
            "cbz x8, 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::NotZero(xreg(8)),
            },
            "080200B5",
            "cbnz x8, 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Eq),
            },
            "00020054",
            "b.eq 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Ne),
            },
            "01020054",
            "b.ne 64",
        ));

        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Hs),
            },
            "02020054",
            "b.hs 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Lo),
            },
            "03020054",
            "b.lo 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Mi),
            },
            "04020054",
            "b.mi 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Pl),
            },
            "05020054",
            "b.pl 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Vs),
            },
            "06020054",
            "b.vs 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Vc),
            },
            "07020054",
            "b.vc 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Hi),
            },
            "08020054",
            "b.hi 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Ls),
            },
            "09020054",
            "b.ls 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Ge),
            },
            "0A020054",
            "b.ge 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Lt),
            },
            "0B020054",
            "b.lt 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Gt),
            },
            "0C020054",
            "b.gt 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Le),
            },
            "0D020054",
            "b.le 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Al),
            },
            "0E020054",
            "b.al 64",
        ));
        insns.push((
            Inst::CondBrLowered {
                target: BranchTarget::ResolvedOffset(64),
                kind: CondBrKind::Cond(Cond::Nv),
            },
            "0F020054",
            "b.nv 64",
        ));

        insns.push((
            Inst::CondBrLoweredCompound {
                taken: BranchTarget::ResolvedOffset(64),
                not_taken: BranchTarget::ResolvedOffset(128),
                kind: CondBrKind::Cond(Cond::Le),
            },
            "0D02005420000014",
            "b.le 64 ; b 128",
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

        insns.push((
            Inst::IndirectBr {
                rn: xreg(3),
                targets: vec![1, 2, 3],
            },
            "60001FD6",
            "br x3",
        ));

        insns.push((Inst::Brk { trap_info: None }, "000020D4", "brk #0"));

        insns.push((
            Inst::Adr {
                rd: writable_xreg(15),
                label: MemLabel::PCRel((1 << 20) - 4),
            },
            "EFFF7F10",
            "adr x15, pc+1048572",
        ));

        let rru = create_reg_universe();
        for (insn, expected_encoding, expected_printing) in insns {
            println!(
                "ARM64: {:?}, {}, {}",
                insn, expected_encoding, expected_printing
            );

            // Check the printed text is as expected.
            let mut const_sec = MachSectionSize::new(0);
            let actual_printing = insn.show_rru_with_constsec(Some(&rru), &mut const_sec, &[]);
            assert_eq!(expected_printing, actual_printing);

            // Check the encoding is as expected.
            let (text_size, rodata_size) = {
                let mut code_sec = MachSectionSize::new(0);
                let mut const_sec = MachSectionSize::new(0);
                insn.emit(&mut code_sec, &mut const_sec, &[]);
                (code_sec.size(), const_sec.size())
            };

            let mut sink = test_utils::TestCodeSink::new();
            let mut sections = MachSections::new();
            sections.add_section(0, text_size);
            sections.add_section(Inst::align_constant_pool(text_size), rodata_size);
            let (code_sec, const_sec) = sections.two_sections(0, 1);
            insn.emit(code_sec, const_sec, &[]);
            sections.emit(&mut sink);
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
