//! Lowering rules for ARM64.

#![allow(dead_code)]
#![allow(non_snake_case)]

//zz use crate::ir::condcodes::IntCC;
//zz use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{Block, InstructionData, Opcode, Type};

use crate::ir::types;
use crate::ir::types::*;

use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::x64::inst::*;
use crate::isa::x64::X64Backend;

use regalloc::{RealReg, Reg, RegClass, VirtualReg, Writable};

use smallvec::SmallVec;

//============================================================================
//

/// Context passed to all lowering functions.
type Ctx<'a> = &'a mut dyn LowerCtx<Inst>;

//=============================================================================
// Helpers for instruction lowering.

fn to_is64(ty: Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 => false,
        types::I64 => true,
        _ => panic!("type {} is none of I8, I16, I32 or I64", ty),
    }
}

//=============================================================================
// Top-level instruction lowering entry point, for one instruction.

/// Actually codegen an instruction's results into registers.
fn lower_insn_to_regs<'a>(ctx: Ctx<'a>, iri: IRInst) {
    let op = ctx.data(iri).opcode();
    let ty = if ctx.num_outputs(iri) == 1 {
        Some(ctx.output_ty(iri, 0))
    } else {
        None
    };

    match op {
        //zz         Opcode::Iconst | Opcode::Bconst | Opcode::F32const | Opcode::F64const => {
        //zz             let value = output_to_const(ctx, outputs[0]).unwrap();
        //zz             let rd = output_to_reg(ctx, outputs[0]);
        //zz             lower_constant(ctx, rd, value);
        //zz         }
        Opcode::Iadd => {
            let regD = ctx.get_output_writable_reg(iri, 0);
            let regL = ctx.get_input_reg(iri, 0);
            let regR = ctx.get_input_reg(iri, 1);
            let is64 = to_is64(ty.unwrap());
            // ** Earth to Move Coalescer: do you copy? **
            ctx.emit(i_Mov_R_R(true, regL, regD));
            ctx.emit(i_Alu_RMI_R(is64, RMI_R_Op::Add, ip_RMI_R(regR), regD));
        }
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
        Opcode::Return => {
            for i in 0..ctx.num_inputs(iri) {
                let src_reg = ctx.get_input_reg(iri, i);
                let retval_reg = ctx.retval(i);
                ctx.emit(i_Mov_R_R(/*is64=*/ true, src_reg, retval_reg));
            }
            // We don't generate the actual |ret| insn here (no way we could)
            // since it first requires a prologue to be generated.  That's a
            // job for the ABI machinery.
        }

        _ => {
            println!("Unimplemented opcode: {:?}", op);
            unimplemented!()
        }
    }
}

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for X64Backend {
    type MInst = Inst;

    fn lower<C: LowerCtx<Inst>>(&self, ctx: &mut C, ir_inst: IRInst) {
        lower_insn_to_regs(ctx, ir_inst);
    }

    fn lower_branch_group<C: LowerCtx<Inst>>(
        &self,
        _ctx: &mut C,
        _branches: &[IRInst],
        _targets: &[BlockIndex],
        _fallthrough: Option<BlockIndex>,
    ) {
        unimplemented!()
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
        //zz                 Opcode::Trap => unimplemented!(),
        //zz
        //zz                 _ => panic!("Unknown branch type!"),
        //zz             }
        //zz         }
    }
}
