//! Lowering rules for ARM64.

#![allow(dead_code)]

//zz use crate::ir::condcodes::IntCC;
//zz use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{Ebb, InstructionData, Opcode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::x64::inst::*;
use crate::isa::x64::X64Backend;

use regalloc::{RealReg, Reg, RegClass, VirtualReg};

//zz use smallvec::SmallVec;
//zz
//zz fn op_to_aluop(op: Opcode, ty: Type) -> Option<ALUOp> {
//zz     match (op, ty) {
//zz         (Opcode::Iadd, I32) => Some(ALUOp::Add32),
//zz         (Opcode::Iadd, I64) => Some(ALUOp::Add64),
//zz         (Opcode::Isub, I32) => Some(ALUOp::Sub32),
//zz         (Opcode::Isub, I64) => Some(ALUOp::Sub64),
//zz         _ => None,
//zz     }
//zz }
//zz
//zz fn is_alu_op(op: Opcode, ctrl_typevar: Type) -> bool {
//zz     op_to_aluop(op, ctrl_typevar).is_some()
//zz }
//zz
//zz /// A lowering result: register, register-shift, register-extend.  An SSA value can always be
//zz /// lowered into one of these options; the register form is the fallback.
//zz #[derive(Clone, Debug)]
//zz enum ResultRSE {
//zz     Reg(Reg),
//zz     RegShift(Reg, ShiftOpAndAmt),
//zz     RegExtend(Reg, ExtendOp),
//zz }
//zz
//zz /// A lowering result: register, register-shift, register-extend, or 12-bit immediate form.
//zz /// An SSA value can always be lowered into one of these options; the register form is the
//zz /// fallback.
//zz #[derive(Clone, Debug)]
//zz enum ResultRSEImm12 {
//zz     Reg(Reg),
//zz     RegShift(Reg, ShiftOpAndAmt),
//zz     RegExtend(Reg, ExtendOp),
//zz     Imm12(Imm12),
//zz }
//zz
//zz impl ResultRSEImm12 {
//zz     fn from_rse(rse: ResultRSE) -> ResultRSEImm12 {
//zz         match rse {
//zz             ResultRSE::Reg(r) => ResultRSEImm12::Reg(r),
//zz             ResultRSE::RegShift(r, s) => ResultRSEImm12::RegShift(r, s),
//zz             ResultRSE::RegExtend(r, e) => ResultRSEImm12::RegExtend(r, e),
//zz         }
//zz     }
//zz }
//zz
//zz /// A lowering result: register, register-shift, register-extend, or logical immediate form.
//zz /// An SSA value can always be lowered into one of these options; the register form is the
//zz /// fallback.
//zz #[derive(Clone, Debug)]
//zz enum ResultRSEImmLogic {
//zz     Reg(Reg),
//zz     RegShift(Reg, ShiftOpAndAmt),
//zz     RegExtend(Reg, ExtendOp),
//zz     ImmLogic(ImmLogic),
//zz }
//zz
//zz impl ResultRSEImmLogic {
//zz     fn from_rse(rse: ResultRSE) -> ResultRSEImmLogic {
//zz         match rse {
//zz             ResultRSE::Reg(r) => ResultRSEImmLogic::Reg(r),
//zz             ResultRSE::RegShift(r, s) => ResultRSEImmLogic::RegShift(r, s),
//zz             ResultRSE::RegExtend(r, e) => ResultRSEImmLogic::RegExtend(r, e),
//zz         }
//zz     }
//zz }
//zz
//zz /// Identifier for a particular output of an instruction.
//zz #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//zz struct InsnOutput {
//zz     insn: IRInst,
//zz     output: usize,
//zz }
//zz
//zz /// Identifier for a particular input of an instruction.
//zz #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//zz struct InsnInput {
//zz     insn: IRInst,
//zz     input: usize,
//zz }
//zz
//zz /// Producer of a value: either a previous instruction's output, or a register that will be
//zz /// codegen'd separately.
//zz #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//zz enum InsnInputSource {
//zz     Output(InsnOutput),
//zz     Reg(Reg),
//zz }
//zz
//zz impl InsnInputSource {
//zz     fn as_output(self) -> Option<InsnOutput> {
//zz         match self {
//zz             InsnInputSource::Output(o) => Some(o),
//zz             _ => None,
//zz         }
//zz     }
//zz }
//zz
//zz /// Context passed to all lowering functions.
//zz type Ctx<'a> = &'a mut dyn LowerCtx<Inst>;
//zz
//zz fn get_input<'a>(ctx: Ctx<'a>, output: InsnOutput, num: usize) -> InsnInput {
//zz     assert!(num <= ctx.num_inputs(output.insn));
//zz     InsnInput {
//zz         insn: output.insn,
//zz         input: num,
//zz     }
//zz }
//zz
//zz /// Convert an instruction input to a producing instruction's output if possible (in same BB), or a
//zz /// register otherwise.
//zz fn input_source<'a>(ctx: Ctx<'a>, input: InsnInput) -> InsnInputSource {
//zz     if let Some((input_inst, result_num)) = ctx.input_inst(input.insn, input.input) {
//zz         let out = InsnOutput {
//zz             insn: input_inst,
//zz             output: result_num,
//zz         };
//zz         InsnInputSource::Output(out)
//zz     } else {
//zz         let reg = ctx.input(input.insn, input.input);
//zz         InsnInputSource::Reg(reg)
//zz     }
//zz }
//zz
//zz /// Lower an instruction output to a 64-bit constant, if possible.
//zz fn output_to_const<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<u64> {
//zz     if out.output > 0 {
//zz         None
//zz     } else {
//zz         let inst_data = ctx.data(out.insn);
//zz         match inst_data {
//zz             &InstructionData::UnaryImm { opcode: _, imm } => {
//zz                 // Only has Into for i64; we use u64 elsewhere, so we cast.
//zz                 let imm: i64 = imm.into();
//zz                 Some(imm as u64)
//zz             }
//zz             &InstructionData::UnaryIeee32 { opcode: _, imm } => Some(imm.bits() as u64),
//zz             &InstructionData::UnaryIeee64 { opcode: _, imm } => Some(imm.bits()),
//zz             _ => None,
//zz         }
//zz     }
//zz }
//zz
//zz /// Lower an instruction output to a constant register-shift amount, if possible.
//zz fn output_to_shiftimm<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Option<ShiftOpShiftImm> {
//zz     output_to_const(ctx, out).and_then(ShiftOpShiftImm::maybe_from_shift)
//zz }
//zz
//zz /// Lower an instruction input to a reg.
//zz fn input_to_reg<'a>(ctx: Ctx<'a>, input: InsnInput) -> Reg {
//zz     ctx.input(input.insn, input.input)
//zz }
//zz
//zz /// Lower an instruction output to a reg.
//zz fn output_to_reg<'a>(ctx: Ctx<'a>, out: InsnOutput) -> Reg {
//zz     ctx.output(out.insn, out.output)
//zz }
//zz
//zz /// Lower an instruction output to a reg, reg/shift, or reg/extend operand.  This does not actually
//zz /// codegen the source instruction; it just uses the vreg into which the source instruction will
//zz /// generate its value.
//zz fn output_to_rse<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSE {
//zz     assert!(out.output <= ctx.num_outputs(out.insn));
//zz     let insn = out.insn;
//zz     let op = ctx.data(out.insn).opcode();
//zz     let out_ty = ctx.output_ty(out.insn, out.output);
//zz
//zz     if op == Opcode::Ishl {
//zz         let shiftee = get_input(ctx, out, 0);
//zz         let _shift_amt = get_input(ctx, out, 1);
//zz
//zz         // Can we get the shift amount as an immediate?
//zz         if let Some(out) = input_source(ctx, shiftee).as_output() {
//zz             if let Some(shiftimm) = output_to_shiftimm(ctx, out) {
//zz                 ctx.dec_use(insn);
//zz                 let reg = input_to_reg(ctx, shiftee);
//zz                 return ResultRSE::RegShift(reg, ShiftOpAndAmt::new(ShiftOp::LSL, shiftimm));
//zz             }
//zz         }
//zz     }
//zz
//zz     // Is this a zero-extend or sign-extend and can we handle that with a register-mode operator?
//zz     if (op == Opcode::Uextend || op == Opcode::Sextend) && out_ty == I64 {
//zz         let sign_extend = op == Opcode::Sextend;
//zz         let extendee = get_input(ctx, out, 0);
//zz         if let Some(out) = input_source(ctx, extendee).as_output() {
//zz             let inner_ty = ctx.output_ty(out.insn, out.output);
//zz             if inner_ty == I32 || inner_ty == I16 || inner_ty == I8 {
//zz                 let extendop = match (sign_extend, inner_ty) {
//zz                     (true, I8) => ExtendOp::SXTB,
//zz                     (false, I8) => ExtendOp::UXTB,
//zz                     (true, I16) => ExtendOp::SXTH,
//zz                     (false, I16) => ExtendOp::UXTH,
//zz                     (true, I32) => ExtendOp::SXTW,
//zz                     (false, I32) => ExtendOp::UXTW,
//zz                     _ => unreachable!(),
//zz                 };
//zz                 let reg = input_to_reg(ctx, extendee);
//zz                 ctx.dec_use(insn);
//zz                 return ResultRSE::RegExtend(reg, extendop);
//zz             }
//zz         }
//zz     }
//zz
//zz     // Otherwise, just return the register corresponding to the output.
//zz     ResultRSE::Reg(output_to_reg(ctx, out))
//zz }
//zz
//zz /// Lower an instruction output to a reg, reg/shift, reg/extend, or 12-bit immediate operand.
//zz fn output_to_rse_imm12<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSEImm12 {
//zz     if let Some(imm_value) = output_to_const(ctx, out) {
//zz         if let Some(i) = Imm12::maybe_from_u64(imm_value) {
//zz             ctx.dec_use(out.insn);
//zz             return ResultRSEImm12::Imm12(i);
//zz         }
//zz     }
//zz
//zz     ResultRSEImm12::from_rse(output_to_rse(ctx, out))
//zz }
//zz
//zz /// Lower an instruction output to a reg, reg/shift, reg/extend, or logic-immediate operand.
//zz fn output_to_rse_immlogic<'a>(ctx: Ctx<'a>, out: InsnOutput) -> ResultRSEImmLogic {
//zz     if let Some(imm_value) = output_to_const(ctx, out) {
//zz         if let Some(i) = ImmLogic::maybe_from_u64(imm_value) {
//zz             ctx.dec_use(out.insn);
//zz             return ResultRSEImmLogic::ImmLogic(i);
//zz         }
//zz     }
//zz
//zz     ResultRSEImmLogic::from_rse(output_to_rse(ctx, out))
//zz }
//zz
//zz fn input_to_rse<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSE {
//zz     match input_source(ctx, input) {
//zz         InsnInputSource::Output(out) => output_to_rse(ctx, out),
//zz         InsnInputSource::Reg(reg) => ResultRSE::Reg(reg),
//zz     }
//zz }
//zz
//zz fn input_to_rse_imm12<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSEImm12 {
//zz     match input_source(ctx, input) {
//zz         InsnInputSource::Output(out) => output_to_rse_imm12(ctx, out),
//zz         InsnInputSource::Reg(reg) => ResultRSEImm12::Reg(reg),
//zz     }
//zz }
//zz
//zz fn input_to_rse_immlogic<'a>(ctx: Ctx<'a>, input: InsnInput) -> ResultRSEImmLogic {
//zz     match input_source(ctx, input) {
//zz         InsnInputSource::Output(out) => output_to_rse_immlogic(ctx, out),
//zz         InsnInputSource::Reg(reg) => ResultRSEImmLogic::Reg(reg),
//zz     }
//zz }
//zz
//zz fn alu_inst_imm12(op: ALUOp, rd: Reg, rn: Reg, rm: ResultRSEImm12) -> Inst {
//zz     match rm {
//zz         ResultRSEImm12::Imm12(imm12) => Inst::AluRRImm12 {
//zz             alu_op: op,
//zz             rd,
//zz             rn,
//zz             imm12,
//zz         },
//zz         ResultRSEImm12::Reg(rm) => Inst::AluRRR {
//zz             alu_op: op,
//zz             rd,
//zz             rn,
//zz             rm,
//zz         },
//zz         ResultRSEImm12::RegShift(rm, shiftop) => Inst::AluRRRShift {
//zz             alu_op: op,
//zz             rd,
//zz             rn,
//zz             rm,
//zz             shiftop,
//zz         },
//zz         ResultRSEImm12::RegExtend(rm, extendop) => Inst::AluRRRExtend {
//zz             alu_op: op,
//zz             rd,
//zz             rn,
//zz             rm,
//zz             extendop,
//zz         },
//zz     }
//zz }
//zz
//zz /// Lower the address of a load or store.
//zz fn lower_address<'a>(ctx: Ctx<'a>, elem_ty: Type, addends: &[InsnInput], offset: i32) -> MemArg {
//zz     // TODO: support base_reg + scale * index_reg. For this, we would need to pattern-match shl or
//zz     // mul instructions (Load/StoreComplex don't include scale factors).
//zz
//zz     // Handle one reg and offset that fits in immediate, if possible.
//zz     if addends.len() == 1 {
//zz         let reg = input_to_reg(ctx, addends[0]);
//zz         if let Some(memarg) = MemArg::reg_maybe_offset(reg, offset as i64, elem_ty) {
//zz             return memarg;
//zz         }
//zz     }
//zz
//zz     // Handle two regs and a zero offset, if possible.
//zz     if addends.len() == 2 && offset == 0 {
//zz         let ra = input_to_reg(ctx, addends[0]);
//zz         let rb = input_to_reg(ctx, addends[1]);
//zz         return MemArg::BasePlusReg(ra, rb);
//zz     }
//zz
//zz     // Otherwise, generate add instructions.
//zz     let addr = ctx.tmp(RegClass::I64);
//zz
//zz     // Get the const into a reg.
//zz     lower_constant(ctx, addr.clone(), offset as u64);
//zz
//zz     // Add each addend to the address.
//zz     for addend in addends {
//zz         let reg = input_to_reg(ctx, *addend);
//zz         ctx.emit(Inst::AluRRR {
//zz             alu_op: ALUOp::Add64,
//zz             rd: addr.clone(),
//zz             rn: addr.clone(),
//zz             rm: reg.clone(),
//zz         });
//zz     }
//zz
//zz     MemArg::Base(addr)
//zz }
//zz
//zz fn lower_constant<'a>(ctx: Ctx<'a>, rd: Reg, value: u64) {
//zz     if let Some(imm12) = Imm12::maybe_from_u64(value) {
//zz         // 12-bit immediate (shifted by 0 or 12 bits) in ADDI using zero register
//zz         ctx.emit(Inst::AluRRImm12 {
//zz             alu_op: ALUOp::Add64,
//zz             rd,
//zz             rn: zero_reg(),
//zz             imm12,
//zz         });
//zz     } else if let Some(imml) = ImmLogic::maybe_from_u64(value) {
//zz         // Weird logical-instruction immediate in ORI using zero register
//zz         ctx.emit(Inst::AluRRImmLogic {
//zz             alu_op: ALUOp::Orr64,
//zz             rd,
//zz             rn: zero_reg(),
//zz             imml,
//zz         });
//zz     } else if let Some(imm) = MovZConst::maybe_from_u64(value) {
//zz         // 16-bit immediate (shifted by 0, 16, 32 or 48 bits) in MOVZ
//zz         ctx.emit(Inst::MovZ { rd, imm });
//zz     } else {
//zz         // 64-bit constant in constant pool
//zz         let const_data = u64_constant(value);
//zz         ctx.emit(Inst::ULoad64 {
//zz             rd,
//zz             mem: MemArg::label(MemLabel::ConstantData(const_data)),
//zz         });
//zz     }
//zz }
//zz
//zz fn lower_condcode(cc: IntCC) -> Cond {
//zz     match cc {
//zz         IntCC::Equal => Cond::Eq,
//zz         IntCC::NotEqual => Cond::Ne,
//zz         IntCC::SignedGreaterThanOrEqual => Cond::Ge,
//zz         IntCC::SignedGreaterThan => Cond::Gt,
//zz         IntCC::SignedLessThanOrEqual => Cond::Le,
//zz         IntCC::SignedLessThan => Cond::Lt,
//zz         IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
//zz         IntCC::UnsignedGreaterThan => Cond::Hi,
//zz         IntCC::UnsignedLessThanOrEqual => Cond::Ls,
//zz         IntCC::UnsignedLessThan => Cond::Lo,
//zz         IntCC::Overflow => Cond::Vs,
//zz         IntCC::NotOverflow => Cond::Vc,
//zz     }
//zz }
//zz
//zz /// Actually codegen an instruction's results into registers.
//zz fn lower_insn_to_regs<'a>(ctx: Ctx<'a>, insn: IRInst) {
//zz     let op = ctx.data(insn).opcode();
//zz     let inputs: SmallVec<[InsnInput; 4]> = (0..ctx.num_inputs(insn))
//zz         .map(|i| InsnInput { insn, input: i })
//zz         .collect();
//zz     let outputs: SmallVec<[InsnOutput; 2]> = (0..ctx.num_outputs(insn))
//zz         .map(|i| InsnOutput { insn, output: i })
//zz         .collect();
//zz     let ty = if outputs.len() > 0 {
//zz         Some(ctx.output_ty(insn, 0))
//zz     } else {
//zz         None
//zz     };
//zz
//zz     match op {
//zz         Opcode::Iconst | Opcode::Bconst | Opcode::F32const | Opcode::F64const => {
//zz             let value = output_to_const(ctx, outputs[0]).unwrap();
//zz             let rd = output_to_reg(ctx, outputs[0]);
//zz             lower_constant(ctx, rd, value);
//zz         }
//zz         Opcode::Iadd => {
//zz             let rd = output_to_reg(ctx, outputs[0]);
//zz             let rn = input_to_reg(ctx, inputs[0]);
//zz             let rm = input_to_rse_imm12(ctx, inputs[1]);
//zz             let ty = ty.unwrap();
//zz             let alu_op = choose_32_64(ty, ALUOp::Add32, ALUOp::Add64);
//zz             ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
//zz         }
//zz         Opcode::Isub => {
//zz             let rd = output_to_reg(ctx, outputs[0]);
//zz             let rn = input_to_reg(ctx, inputs[0]);
//zz             let rm = input_to_rse_imm12(ctx, inputs[1]);
//zz             let ty = ty.unwrap();
//zz             let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
//zz             ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
//zz         }
//zz
//zz         Opcode::Load
//zz         | Opcode::Uload8
//zz         | Opcode::Sload8
//zz         | Opcode::Uload16
//zz         | Opcode::Sload16
//zz         | Opcode::Uload32
//zz         | Opcode::Sload32
//zz         | Opcode::LoadComplex
//zz         | Opcode::Uload8Complex
//zz         | Opcode::Sload8Complex
//zz         | Opcode::Uload16Complex
//zz         | Opcode::Sload16Complex
//zz         | Opcode::Uload32Complex
//zz         | Opcode::Sload32Complex => {
//zz             let off = ldst_offset(ctx.data(insn)).unwrap();
//zz             let elem_ty = match op {
//zz                 Opcode::Sload8 | Opcode::Uload8 | Opcode::Sload8Complex | Opcode::Uload8Complex => {
//zz                     I8
//zz                 }
//zz                 Opcode::Sload16
//zz                 | Opcode::Uload16
//zz                 | Opcode::Sload16Complex
//zz                 | Opcode::Uload16Complex => I16,
//zz                 Opcode::Sload32
//zz                 | Opcode::Uload32
//zz                 | Opcode::Sload32Complex
//zz                 | Opcode::Uload32Complex => I32,
//zz                 Opcode::Load | Opcode::LoadComplex => I64,
//zz                 _ => unreachable!(),
//zz             };
//zz
//zz             let mem = lower_address(ctx, elem_ty, &inputs[..], off);
//zz             let rd = output_to_reg(ctx, outputs[0]);
//zz
//zz             ctx.emit(match op {
//zz                 Opcode::Uload8 | Opcode::Uload8Complex => Inst::ULoad8 { rd, mem },
//zz                 Opcode::Sload8 | Opcode::Sload8Complex => Inst::SLoad8 { rd, mem },
//zz                 Opcode::Uload16 | Opcode::Uload16Complex => Inst::ULoad16 { rd, mem },
//zz                 Opcode::Sload16 | Opcode::Sload16Complex => Inst::SLoad16 { rd, mem },
//zz                 Opcode::Uload32 | Opcode::Uload32Complex => Inst::ULoad32 { rd, mem },
//zz                 Opcode::Sload32 | Opcode::Sload32Complex => Inst::SLoad32 { rd, mem },
//zz                 Opcode::Load | Opcode::LoadComplex => Inst::ULoad64 { rd, mem },
//zz                 _ => unreachable!(),
//zz             });
//zz         }
//zz
//zz         Opcode::Store
//zz         | Opcode::Istore8
//zz         | Opcode::Istore16
//zz         | Opcode::Istore32
//zz         | Opcode::StoreComplex
//zz         | Opcode::Istore8Complex
//zz         | Opcode::Istore16Complex
//zz         | Opcode::Istore32Complex => {
//zz             let off = ldst_offset(ctx.data(insn)).unwrap();
//zz             let elem_ty = match op {
//zz                 Opcode::Istore8 | Opcode::Istore8Complex => I8,
//zz                 Opcode::Istore16 | Opcode::Istore16Complex => I16,
//zz                 Opcode::Istore32 | Opcode::Istore32Complex => I32,
//zz                 Opcode::Store | Opcode::StoreComplex => I64,
//zz                 _ => unreachable!(),
//zz             };
//zz
//zz             let mem = lower_address(ctx, elem_ty, &inputs[1..], off);
//zz             let rd = input_to_reg(ctx, inputs[0]);
//zz
//zz             ctx.emit(match op {
//zz                 Opcode::Istore8 | Opcode::Istore8Complex => Inst::Store8 { rd, mem },
//zz                 Opcode::Istore16 | Opcode::Istore16Complex => Inst::Store16 { rd, mem },
//zz                 Opcode::Istore32 | Opcode::Istore32Complex => Inst::Store32 { rd, mem },
//zz                 Opcode::Store | Opcode::StoreComplex => Inst::Store64 { rd, mem },
//zz                 _ => unreachable!(),
//zz             });
//zz         }
//zz
//zz         Opcode::Return => {
//zz             // TODO: multiple return values.
//zz             assert!(inputs.len() <= 1);
//zz             if inputs.len() > 0 {
//zz                 let retval = input_to_reg(ctx, inputs[0]);
//zz                 let abi_reg = xreg(0);
//zz                 ctx.emit(Inst::gen_move(abi_reg, retval));
//zz             }
//zz             ctx.emit(Inst::Ret {});
//zz         }
//zz
//zz         // TODO: cmp
//zz         // TODO: more alu ops
//zz         _ => {
//zz             println!("Unimplemented opcode: {:?}", op);
//zz             unimplemented!()
//zz         }
//zz     }
//zz }
//zz
//zz fn choose_32_64(ty: Type, op32: ALUOp, op64: ALUOp) -> ALUOp {
//zz     if ty == I32 {
//zz         op32
//zz     } else if ty == I64 {
//zz         op64
//zz     } else {
//zz         panic!("type {} is not I32 or I64", ty);
//zz     }
//zz }
//zz
//zz fn branch_target(data: &InstructionData) -> Option<Ebb> {
//zz     match data {
//zz         &InstructionData::BranchIcmp { destination, .. }
//zz         | &InstructionData::Branch { destination, .. }
//zz         | &InstructionData::BranchInt { destination, .. }
//zz         | &InstructionData::Jump { destination, .. }
//zz         | &InstructionData::BranchTable { destination, .. }
//zz         | &InstructionData::BranchFloat { destination, .. } => Some(destination),
//zz         _ => {
//zz             assert!(!data.opcode().is_branch());
//zz             None
//zz         }
//zz     }
//zz }
//zz
//zz fn ldst_offset(data: &InstructionData) -> Option<i32> {
//zz     match data {
//zz         &InstructionData::Load { offset, .. }
//zz         | &InstructionData::StackLoad { offset, .. }
//zz         | &InstructionData::LoadComplex { offset, .. }
//zz         | &InstructionData::Store { offset, .. }
//zz         | &InstructionData::StackStore { offset, .. }
//zz         | &InstructionData::StoreComplex { offset, .. } => Some(offset.into()),
//zz         _ => None,
//zz     }
//zz }
//zz
//zz fn inst_condcode(data: &InstructionData) -> Option<IntCC> {
//zz     match data {
//zz         &InstructionData::IntCond { cond, .. }
//zz         | &InstructionData::BranchIcmp { cond, .. }
//zz         | &InstructionData::IntCompare { cond, .. }
//zz         | &InstructionData::IntCondTrap { cond, .. }
//zz         | &InstructionData::BranchInt { cond, .. }
//zz         | &InstructionData::IntSelect { cond, .. }
//zz         | &InstructionData::IntCompareImm { cond, .. } => Some(cond),
//zz         _ => None,
//zz     }
//zz }

impl LowerBackend for X64Backend {
    type MInst = Inst;

    fn lower_entry<C: LowerCtx<Inst>>(&self, _ctx: &mut C, _ebb: Ebb) {
        //zz         ctx.emit(Inst::LiveIns);
        //zz         // TODO: ABI support for more than 8 args.
        //zz         assert!(ctx.num_ebb_params(ebb) < 8);
        //zz         for i in 0..ctx.num_ebb_params(ebb) {
        //zz             let abi_reg = xreg(i as u8);
        //zz             let ebb_reg = ctx.ebb_param(ebb, i);
        //zz             ctx.emit(Inst::gen_move(ebb_reg, abi_reg));
        //zz         }
    }

    fn lower<C: LowerCtx<Inst>>(&self, _ctx: &mut C, _ir_inst: IRInst) {
        //zz         lower_insn_to_regs(ctx, ir_inst);
    }

    fn lower_branch_group<C: LowerCtx<Inst>>(
        &self,
        _ctx: &mut C,
        _branches: &[IRInst],
        _targets: &[BlockIndex],
        _fallthrough: Option<BlockIndex>,
    ) {
        //zz         // A block should end with at most two branches. The first may be a
        //zz         // conditional branch; a conditional branch can be followed only by an
        //zz         // unconditional branch or fallthrough. Otherwise, if only one branch,
        //zz         // it may be an unconditional branch, a fallthrough, a return, or a
        //zz         // trap. These conditions are verified by `is_ebb_basic()` during the
        //zz         // verifier pass.
        //zz         assert!(branches.len() <= 2);
        //zz
        //zz         if branches.len() == 2 {
        //zz             // Must be a conditional branch followed by an unconditional branch.
        //zz             let op1 = ctx.data(branches[0]).opcode();
        //zz             let op2 = ctx.data(branches[1]).opcode();
        //zz
        //zz             assert!(op2 == Opcode::Jump || op2 == Opcode::Fallthrough);
        //zz             let taken = BranchTarget::Block(targets[0]);
        //zz             let not_taken = match op2 {
        //zz                 Opcode::Jump => BranchTarget::Block(targets[1]),
        //zz                 Opcode::Fallthrough => BranchTarget::Block(fallthrough.unwrap()),
        //zz                 _ => unreachable!(), // assert above.
        //zz             };
        //zz             match op1 {
        //zz                 Opcode::Brz | Opcode::Brnz => {
        //zz                     let rt = input_to_reg(
        //zz                         ctx,
        //zz                         InsnInput {
        //zz                             insn: branches[0],
        //zz                             input: 0,
        //zz                         },
        //zz                     );
        //zz                     let kind = match op1 {
        //zz                         Opcode::Brz => CondBrKind::Zero(rt),
        //zz                         Opcode::Brnz => CondBrKind::NotZero(rt),
        //zz                         _ => unreachable!(),
        //zz                     };
        //zz                     ctx.emit(Inst::CondBr {
        //zz                         taken,
        //zz                         not_taken,
        //zz                         kind,
        //zz                     });
        //zz                 }
        //zz                 Opcode::BrIcmp => {
        //zz                     let rn = input_to_reg(
        //zz                         ctx,
        //zz                         InsnInput {
        //zz                             insn: branches[0],
        //zz                             input: 0,
        //zz                         },
        //zz                     );
        //zz                     let rm = input_to_reg(
        //zz                         ctx,
        //zz                         InsnInput {
        //zz                             insn: branches[0],
        //zz                             input: 1,
        //zz                         },
        //zz                     );
        //zz                     let ty = ctx.input_ty(branches[0], 0);
        //zz                     let alu_op = choose_32_64(ty, ALUOp::SubS32, ALUOp::SubS64);
        //zz                     let rd = zero_reg();
        //zz                     ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
        //zz                     let cond = lower_condcode(inst_condcode(ctx.data(branches[0])).unwrap());
        //zz                     ctx.emit(Inst::CondBr {
        //zz                         taken,
        //zz                         not_taken,
        //zz                         kind: CondBrKind::Cond(cond),
        //zz                     });
        //zz                 }
        //zz
        //zz                 // TODO: Brif/icmp, Brff/icmp, jump tables, call, ret
        //zz                 _ => unimplemented!(),
        //zz             }
        //zz         } else {
        //zz             assert!(branches.len() == 1);
        //zz
        //zz             // Must be an unconditional branch, fallthrough, return, or trap.
        //zz             let op = ctx.data(branches[0]).opcode();
        //zz             match op {
        //zz                 Opcode::Jump => {
        //zz                     ctx.emit(Inst::Jump {
        //zz                         dest: BranchTarget::Block(targets[0]),
        //zz                     });
        //zz                 }
        //zz                 Opcode::Fallthrough => {
        //zz                     ctx.emit(Inst::Jump {
        //zz                         dest: BranchTarget::Block(targets[0]),
        //zz                     });
        //zz                 }
        //zz
        //zz                 Opcode::FallthroughReturn => {
        //zz                     // What is this? The definition says it's a "special
        //zz                     // instruction" meant to allow falling through into an
        //zz                     // epilogue that will then return; that just sounds like a
        //zz                     // normal fallthrough. TODO: Do we need to handle this
        //zz                     // differently?
        //zz                     unimplemented!();
        //zz                 }
        //zz
        //zz                 Opcode::Return => {
        //zz                     ctx.emit(Inst::Ret {});
        //zz                 }
        //zz
        //zz                 Opcode::Trap => unimplemented!(),
        //zz
        //zz                 _ => panic!("Unknown branch type!"),
        //zz             }
        //zz         }
    }
}
