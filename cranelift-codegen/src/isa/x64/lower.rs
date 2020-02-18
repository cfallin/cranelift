//! Lowering rules for X64.

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

fn iri_to_u64_immediate<'a>(ctx: Ctx<'a>, iri: IRInst) -> Option<u64> {
    let inst_data = ctx.data(iri);
    if inst_data.opcode() == Opcode::Null {
        Some(0)
    } else {
        match inst_data {
            &InstructionData::UnaryImm { opcode: _, imm } => {
                // Only has Into for i64; we use u64 elsewhere, so we cast.
                let imm: i64 = imm.into();
                Some(imm as u64)
            }
            _ => None
        }
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

    // This is all outstandingly feeble.  TODO: much better!

    match op {
        Opcode::Iconst => {
            if let Some(w64) = iri_to_u64_immediate(ctx, iri) {
                // Get exactly the bit pattern in 'w64' into the dest.  No
                // monkeying with sign extension etc.
                let dstIs64 = w64 > 0xFFFF_FFFF;
                let regD = ctx.output(iri, 0);
                ctx.emit(i_Imm_R(dstIs64, w64, regD));
            } else {
                unimplemented!()
            }
        }
        Opcode::Bconst | Opcode::F32const | Opcode::F64const | Opcode::Null => {
            //let value = output_to_const(ctx, outputs[0]).unwrap();
            //let rd = output_to_reg(ctx, outputs[0]);
            //lower_constant(ctx, rd, value);
            unimplemented!()
        }
        Opcode::Iadd => {
            let regD = ctx.output(iri, 0);
            let regL = ctx.input(iri, 0);
            let regR = ctx.input(iri, 1);
            let is64 = to_is64(ty.unwrap());
            // ** Earth to Move Coalescer: do you copy? **
            ctx.emit(i_Mov_R_R(true, regL, regD));
            ctx.emit(i_Alu_RMI_R(is64, RMI_R_Op::Add, ip_RMI_R(regR), regD));
        }
        Opcode::Isub => {
            //let rd = output_to_reg(ctx, outputs[0]);
            //let rn = input_to_reg(ctx, inputs[0]);
            //let rm = input_to_rse_imm12(ctx, inputs[1]);
            //let ty = ty.unwrap();
            //let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            //ctx.emit(alu_inst_imm12(alu_op, rd, rn, rm));
            unimplemented!()
        }

        Opcode::UaddSat | Opcode::SaddSat => {
            // TODO: open-code a sequence: adds, then branch-on-no-overflow
            // over a load of the saturated value.
            unimplemented!()
        }

        Opcode::UsubSat | Opcode::SsubSat => {
            // TODO
            unimplemented!()
        }

        Opcode::Ineg => {
            //let rd = output_to_reg(ctx, outputs[0]);
            //let rn = zero_reg();
            //let rm = input_to_reg(ctx, inputs[0]);
            //let ty = ty.unwrap();
            //let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            //ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
            unimplemented!()
        }

        Opcode::Imul => {
            //let rd = output_to_reg(ctx, outputs[0]);
            //let rn = input_to_reg(ctx, inputs[0]);
            //let rm = input_to_reg(ctx, inputs[1]);
            //let ty = ty.unwrap();
            //let alu_op = choose_32_64(ty, ALUOp::MAdd32, ALUOp::MAdd64);
            //ctx.emit(Inst::AluRRRR {
            //    alu_op,
            //    rd,
            //    rn,
            //    rm,
            //    ra: zero_reg(),
            //});
            unimplemented!()
        }

        Opcode::Umulhi | Opcode::Smulhi => {
            //let _ty = ty.unwrap();
            // TODO
            unimplemented!()
        }

        Opcode::Udiv | Opcode::Sdiv | Opcode::Urem | Opcode::Srem => {
            // TODO
            unimplemented!()
        }

        Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::Bnot
        | Opcode::BandNot
        | Opcode::BorNot
        | Opcode::BxorNot => {
            // TODO
            unimplemented!()
        }

        Opcode::Rotl | Opcode::Rotr => {
            // TODO
            unimplemented!()
        }

        Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => {
            // TODO
            unimplemented!()
        }

        Opcode::Bitrev => {
            // TODO
            unimplemented!()
        }

        Opcode::Clz | Opcode::Cls | Opcode::Ctz | Opcode::Popcnt => {
            // TODO
            unimplemented!()
        }

        Opcode::Load
        | Opcode::Uload8
        | Opcode::Sload8
        | Opcode::Uload16
        | Opcode::Sload16
        | Opcode::Uload32
        | Opcode::Sload32
        | Opcode::LoadComplex
        | Opcode::Uload8Complex
        | Opcode::Sload8Complex
        | Opcode::Uload16Complex
        | Opcode::Sload16Complex
        | Opcode::Uload32Complex
        | Opcode::Sload32Complex => {
            //let off = ldst_offset(ctx.data(insn)).unwrap();
            //let elem_ty = match op {
            //    Opcode::Sload8 | Opcode::Uload8 | Opcode::Sload8Complex | Opcode::Uload8Com//plex => {
            //        I8
            //    }
            //    Opcode::Sload16
            //    | Opcode::Uload16
            //    | Opcode::Sload16Complex
            //    | Opcode::Uload16Complex => I16,
            //    Opcode::Sload32
            //    | Opcode::Uload32
            //    | Opcode::Sload32Complex
            //    | Opcode::Uload32Complex => I32,
            //    Opcode::Load | Opcode::LoadComplex => I64,
            //    _ => unreachable!(),
            //};

            //let mem = lower_address(ctx, elem_ty, &inputs[..], off);
            //let rd = output_to_reg(ctx, outputs[0]);

            //ctx.emit(match op {
            //    Opcode::Uload8 | Opcode::Uload8Complex => Inst::ULoad8 { rd, mem },
            //    Opcode::Sload8 | Opcode::Sload8Complex => Inst::SLoad8 { rd, mem },
            //    Opcode::Uload16 | Opcode::Uload16Complex => Inst::ULoad16 { rd, mem },
            //    Opcode::Sload16 | Opcode::Sload16Complex => Inst::SLoad16 { rd, mem },
            //    Opcode::Uload32 | Opcode::Uload32Complex => Inst::ULoad32 { rd, mem },
            //    Opcode::Sload32 | Opcode::Sload32Complex => Inst::SLoad32 { rd, mem },
            //    Opcode::Load | Opcode::LoadComplex => Inst::ULoad64 { rd, mem },
            //    _ => unreachable!(),
            //});
            unimplemented!()
        }

        Opcode::Store
        | Opcode::Istore8
        | Opcode::Istore16
        | Opcode::Istore32
        | Opcode::StoreComplex
        | Opcode::Istore8Complex
        | Opcode::Istore16Complex
        | Opcode::Istore32Complex => {
            //let off = ldst_offset(ctx.data(insn)).unwrap();
            //let elem_ty = match op {
            //    Opcode::Istore8 | Opcode::Istore8Complex => I8,
            //    Opcode::Istore16 | Opcode::Istore16Complex => I16,
            //    Opcode::Istore32 | Opcode::Istore32Complex => I32,
            //    Opcode::Store | Opcode::StoreComplex => I64,
            //    _ => unreachable!(),
            //};

            //let mem = lower_address(ctx, elem_ty, &inputs[1..], off);
            //let rd = input_to_reg(ctx, inputs[0]);

            //ctx.emit(match op {
            //    Opcode::Istore8 | Opcode::Istore8Complex => Inst::Store8 { rd, mem },
            //    Opcode::Istore16 | Opcode::Istore16Complex => Inst::Store16 { rd, mem },
            //    Opcode::Istore32 | Opcode::Istore32Complex => Inst::Store32 { rd, mem },
            //    Opcode::Store | Opcode::StoreComplex => Inst::Store64 { rd, mem },
            //    _ => unreachable!(),
            //});
            unimplemented!()
        }

        Opcode::StackLoad => {
            // TODO
            unimplemented!()
        }

        Opcode::StackStore => {
            // TODO
            unimplemented!()
        }

        Opcode::StackAddr => {
            // TODO
            unimplemented!()
        }

        Opcode::GlobalValue => {
            // TODO
            unimplemented!()
        }

        Opcode::SymbolValue => {
            // TODO
            unimplemented!()
        }

        Opcode::HeapAddr => {
            // TODO
            unimplemented!()
        }

        Opcode::TableAddr => {
            // TODO
            unimplemented!()
        }

        Opcode::Nop => {
            // Nothing.
            unimplemented!()
        }

        Opcode::Select | Opcode::Selectif => {
            // TODO.
            unimplemented!()
        }

        Opcode::Bitselect => {
            // TODO.
            unimplemented!()
        }

        Opcode::IsNull | Opcode::IsInvalid | Opcode::Trueif | Opcode::Trueff => {
            // TODO.
            unimplemented!()
        }

        Opcode::Copy => {
            // TODO
            unimplemented!()
        }

        Opcode::Breduce | Opcode::Bextend | Opcode::Bint | Opcode::Bmask => {
            // TODO
            unimplemented!()
        }

        Opcode::Uextend | Opcode::Sextend => {
            // TODO: this is all extremely lame, all because Mov{ZX,SX}_M_R
            // don't accept a register source operand.  They should be changed
            // so as to have _RM_R form.
            // TODO2: if the source operand is a load, incorporate that.
            let isZX = op == Opcode::Uextend;
            let tyS = ctx.input_ty(iri, 0);
            let tyD = ctx.output_ty(iri, 0);
            let regS = ctx.input(iri, 0);
            let regD = ctx.output(iri, 0);
            ctx.emit(i_Mov_R_R(true, regS, regD));
            match (tyS, tyD, isZX) {
                (types::I8, types::I64, false) => {
                    ctx.emit(i_Shift_R(true, ShiftKind::Left, 56, regD));
                    ctx.emit(i_Shift_R(true, ShiftKind::RightS, 56, regD));
                }
                _ => unimplemented!()
            }
        }

        Opcode::Ireduce | Opcode::Isplit | Opcode::Iconcat => {
            // TODO
            unimplemented!()
        }

        Opcode::FallthroughReturn => {
            // What is this? The definition says it's a "special
            // instruction" meant to allow falling through into an
            // epilogue that will then return; that just sounds like a
            // normal fallthrough. TODO: Do we need to handle this
            // differently?
            unimplemented!();
        }

        Opcode::Return => {
            for i in 0..ctx.num_inputs(iri) {
                let src_reg = ctx.input(iri, i);
                let retval_reg = ctx.retval(i);
                ctx.emit(i_Mov_R_R(/*is64=*/ true, src_reg, retval_reg));
            }
            // We don't generate the actual |ret| insn here (no way we could)
            // since it first requires a prologue to be generated.  That's a
            // job for the ABI machinery.
        }

        Opcode::Icmp | Opcode::IcmpImm | Opcode::Ifcmp | Opcode::IfcmpImm => {
            // TODO
            unimplemented!()
        }

        Opcode::JumpTableEntry => {
            // TODO
            unimplemented!()
        }

        Opcode::JumpTableBase => {
            // TODO
            unimplemented!()
        }

        Opcode::Debugtrap => {
            unimplemented!()
        }

        Opcode::Trap => {
            unimplemented!()
        }

        Opcode::Trapz | Opcode::Trapnz | Opcode::Trapif | Opcode::Trapff => {
            unimplemented!()
        }

        Opcode::ResumableTrap => {
            unimplemented!()
        }

        Opcode::Safepoint => {
            unimplemented!()
        }

        Opcode::FuncAddr => {
            // TODO
            unimplemented!()
        }

        Opcode::Call | Opcode::CallIndirect => {
            //let (abi, inputs) = match op {
            //    Opcode::Call => {
            //        let extname = ctx.call_target(insn).unwrap();
            //        let sig = ctx.call_sig(insn).unwrap();
            //        assert!(inputs.len() == sig.params.len());
            //        assert!(outputs.len() == sig.returns.len());
            //        (ARM64ABICall::from_func(sig, extname), &inputs[..])
            //    }
            //    Opcode::CallIndirect => {
            //        let ptr = input_to_reg(ctx, inputs[0]);
            //        let sig = ctx.call_sig(insn).unwrap();
            //        assert!(inputs.len() - 1 == sig.params.len());
            //        assert!(outputs.len() == sig.returns.len());
            //        (ARM64ABICall::from_ptr(sig, ptr), &inputs[1..])
            //    }
            //    _ => unreachable!(),
            //};
            //for (i, input) in inputs.iter().enumerate() {
            //    let arg_reg = input_to_reg(ctx, *input);
            //    ctx.emit(abi.gen_copy_reg_to_arg(i, arg_reg));
            //}
            //ctx.emit(abi.gen_call());
            //for (i, output) in outputs.iter().enumerate() {
            //    let retval_reg = output_to_reg(ctx, *output);
            //    ctx.emit(abi.gen_copy_retval_to_reg(i, retval_reg));
            //}
            unimplemented!()
        }

        Opcode::GetPinnedReg
        | Opcode::SetPinnedReg
        | Opcode::Spill
        | Opcode::Fill
        | Opcode::FillNop
        | Opcode::Regmove
        | Opcode::CopySpecial
        | Opcode::CopyToSsa
        | Opcode::CopyNop
        | Opcode::AdjustSpDown
        | Opcode::AdjustSpUpImm
        | Opcode::AdjustSpDownImm
        | Opcode::IfcmpSp
        | Opcode::Regspill
        | Opcode::Regfill => {
            panic!("Unused opcode should not be encountered.");
        }

        // TODO: cmp
        // TODO: more alu ops
        Opcode::Jump
        | Opcode::Fallthrough
        | Opcode::Brz
        | Opcode::Brnz
        | Opcode::BrIcmp
        | Opcode::Brif
        | Opcode::Brff
        | Opcode::IndirectJumpTableBr
        | Opcode::BrTable => {
            panic!("Branch opcode reached non-branch lowering logic!");
        }

        Opcode::Vconst
        | Opcode::Shuffle
        | Opcode::Vsplit
        | Opcode::Vconcat
        | Opcode::Vselect
        | Opcode::VanyTrue
        | Opcode::VallTrue
        | Opcode::Splat
        | Opcode::Insertlane
        | Opcode::Extractlane
        | Opcode::Bitcast
        | Opcode::RawBitcast
        | Opcode::ScalarToVector => {
            // TODO
            panic!("Vector ops not implemented.");
        }

        Opcode::Fcmp
        | Opcode::Ffcmp
        | Opcode::Fadd
        | Opcode::Fsub
        | Opcode::Fmul
        | Opcode::Fdiv
        | Opcode::Sqrt
        | Opcode::Fma
        | Opcode::Fneg
        | Opcode::Fabs
        | Opcode::Fcopysign
        | Opcode::Fmin
        | Opcode::Fmax
        | Opcode::Ceil
        | Opcode::Floor
        | Opcode::Trunc
        | Opcode::Nearest
        | Opcode::Fpromote
        | Opcode::Fdemote
        | Opcode::FcvtToUint
        | Opcode::FcvtToUintSat
        | Opcode::FcvtToSint
        | Opcode::FcvtToSintSat
        | Opcode::FcvtFromUint
        | Opcode::FcvtFromSint => {
            panic!("Floating point ops not implemented.");
        }

        Opcode::IaddImm
        | Opcode::ImulImm
        | Opcode::UdivImm
        | Opcode::SdivImm
        | Opcode::UremImm
        | Opcode::SremImm
        | Opcode::IrsubImm
        | Opcode::IaddCin
        | Opcode::IaddIfcin
        | Opcode::IaddCout
        | Opcode::IaddIfcout
        | Opcode::IaddCarry
        | Opcode::IaddIfcarry
        | Opcode::IsubBin
        | Opcode::IsubIfbin
        | Opcode::IsubBout
        | Opcode::IsubIfbout
        | Opcode::IsubBorrow
        | Opcode::IsubIfborrow
        | Opcode::BandImm
        | Opcode::BorImm
        | Opcode::BxorImm
        | Opcode::RotlImm
        | Opcode::RotrImm
        | Opcode::IshlImm
        | Opcode::UshrImm
        | Opcode::SshrImm => {
            panic!("ALU+imm and ALU+carry ops should not appear here!");
        }

        Opcode::X86Udivmodx
        | Opcode::X86Sdivmodx
        | Opcode::X86Umulx
        | Opcode::X86Smulx
        | Opcode::X86Cvtt2si
        | Opcode::X86Fmin
        | Opcode::X86Fmax
        | Opcode::X86Push
        | Opcode::X86Pop
        | Opcode::X86Bsr
        | Opcode::X86Bsf
        | Opcode::X86Pshufd
        | Opcode::X86Pshufb
        | Opcode::X86Pextr
        | Opcode::X86Pinsr
        | Opcode::X86Insertps
        | Opcode::X86Movsd
        | Opcode::X86Movlhps
        | Opcode::X86Psll
        | Opcode::X86Psrl
        | Opcode::X86Psra
        | Opcode::X86Ptest
        | Opcode::X86Pmaxs
        | Opcode::X86Pmaxu
        | Opcode::X86Pmins
        | Opcode::X86Pminu => {
            panic!("x86-specific opcode in supposedly arch-neutral IR!");
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
