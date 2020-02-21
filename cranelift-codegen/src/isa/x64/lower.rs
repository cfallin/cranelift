//! Lowering rules for X64.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::ir::condcodes::IntCC;
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

fn is_int_ty(ty: Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        _ => false,
    }
}

fn int_ty_to_is64(ty: Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 => false,
        types::I64 => true,
        _ => panic!("type {} is none of I8, I16, I32 or I64", ty),
    }
}

fn int_ty_to_sizeB(ty: Type) -> u8 {
    match ty {
        types::I8 => 1,
        types::I16 => 2,
        types::I32 => 4,
        types::I64 => 8,
        _ => panic!("ity_to_sizeB"),
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
            _ => None,
        }
    }
}

// Clone of arm64 version.  TODO: de-clone, re-name?
fn inst_condcode(data: &InstructionData) -> IntCC {
    match data {
        &InstructionData::IntCond { cond, .. }
        | &InstructionData::BranchIcmp { cond, .. }
        | &InstructionData::IntCompare { cond, .. }
        | &InstructionData::IntCondTrap { cond, .. }
        | &InstructionData::BranchInt { cond, .. }
        | &InstructionData::IntSelect { cond, .. }
        | &InstructionData::IntCompareImm { cond, .. } => cond,
        _ => panic!("inst_condcode(x64): unhandled: {:?}", data),
    }
}

fn intCC_to_x64_CC(cc: IntCC) -> CC {
    match cc {
        IntCC::Equal => CC::Z,
        IntCC::NotEqual => CC::NZ,
        IntCC::SignedGreaterThanOrEqual => CC::NL,
        IntCC::SignedGreaterThan => CC::NLE,
        IntCC::SignedLessThanOrEqual => CC::LE,
        IntCC::SignedLessThan => CC::L,
        IntCC::UnsignedGreaterThanOrEqual => CC::NB,
        IntCC::UnsignedGreaterThan => CC::NBE,
        IntCC::UnsignedLessThanOrEqual => CC::BE,
        IntCC::UnsignedLessThan => CC::B,
        IntCC::Overflow => CC::O,
        IntCC::NotOverflow => CC::NO,
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

    let mut unimplemented = false;

    match op {
        Opcode::Iconst => {
            if let Some(w64) = iri_to_u64_immediate(ctx, iri) {
                // Get exactly the bit pattern in 'w64' into the dest.  No
                // monkeying with sign extension etc.
                let dstIs64 = w64 > 0xFFFF_FFFF;
                let regD = ctx.output(iri, 0);
                ctx.emit(i_Imm_R(dstIs64, w64, regD));
            } else {
                unimplemented = true;
            }
        }
        Opcode::Bconst | Opcode::F32const | Opcode::F64const | Opcode::Null => {
            //let value = output_to_const(ctx, outputs[0]).unwrap();
            //let rd = output_to_reg(ctx, outputs[0]);
            //lower_constant(ctx, rd, value);
            unimplemented = true;
        }
        Opcode::Iadd | Opcode::Isub => {
            let regD = ctx.output(iri, 0);
            let regL = ctx.input(iri, 0);
            let regR = ctx.input(iri, 1);
            let is64 = int_ty_to_is64(ty.unwrap());
            let how = if op == Opcode::Iadd {
                RMI_R_Op::Add
            } else {
                RMI_R_Op::Sub
            };
            ctx.emit(i_Mov_R_R(true, regL, regD));
            ctx.emit(i_Alu_RMI_R(is64, how, ip_RMI_R(regR), regD));
        }
        Opcode::Imax | Opcode::Imin | Opcode::Umin | Opcode::Umax => {
            // TODO
            unimplemented!()
        }

        Opcode::UaddSat | Opcode::SaddSat => {
            // TODO: open-code a sequence: adds, then branch-on-no-overflow
            // over a load of the saturated value.
            unimplemented = true;
        }

        Opcode::UsubSat | Opcode::SsubSat => {
            // TODO
            unimplemented = true;
        }

        Opcode::Ineg => {
            //let rd = output_to_reg(ctx, outputs[0]);
            //let rn = zero_reg();
            //let rm = input_to_reg(ctx, inputs[0]);
            //let ty = ty.unwrap();
            //let alu_op = choose_32_64(ty, ALUOp::Sub32, ALUOp::Sub64);
            //ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
            unimplemented = true;
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
            unimplemented = true;
        }

        Opcode::Umulhi | Opcode::Smulhi => {
            //let _ty = ty.unwrap();
            // TODO
            unimplemented = true;
        }

        Opcode::Udiv | Opcode::Sdiv | Opcode::Urem | Opcode::Srem => {
            // TODO
            unimplemented = true;
        }

        Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::Bnot
        | Opcode::BandNot
        | Opcode::BorNot
        | Opcode::BxorNot => {
            // TODO
            unimplemented = true;
        }

        Opcode::Rotl | Opcode::Rotr => {
            // TODO
            unimplemented = true;
        }

        Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => {
            // TODO: implement imm shift value into insn
            let tySL = ctx.input_ty(iri, 0);
            let tyD = ctx.output_ty(iri, 0); // should be the same as tySL
            let regSL = ctx.input(iri, 0);
            let regSR = ctx.input(iri, 1);
            let regD = ctx.output(iri, 0);
            if tyD == tySL && (tyD == types::I32 || tyD == types::I64) {
                let how = match op {
                    Opcode::Ishl => ShiftKind::Left,
                    Opcode::Ushr => ShiftKind::RightZ,
                    Opcode::Sshr => ShiftKind::RightS,
                    _ => unreachable!(),
                };
                let is64 = tyD == types::I64;
                let r_rcx = reg_RCX();
                let w_rcx = Writable::<Reg>::from_reg(r_rcx);
                ctx.emit(i_Mov_R_R(true, regSL, regD));
                ctx.emit(i_Mov_R_R(true, regSR, w_rcx));
                ctx.emit(i_Shift_R(is64, how, 0 /*%cl*/, regD));
            } else {
                unimplemented = true;
            }
        }

        Opcode::Bitrev => {
            // TODO
            unimplemented = true;
        }

        Opcode::Clz | Opcode::Cls | Opcode::Ctz | Opcode::Popcnt => {
            // TODO
            unimplemented = true;
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
            unimplemented = true;
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
            unimplemented = true;
        }

        Opcode::StackLoad => {
            // TODO
            unimplemented = true;
        }

        Opcode::StackStore => {
            // TODO
            unimplemented = true;
        }

        Opcode::StackAddr => {
            // TODO
            unimplemented = true;
        }

        Opcode::GlobalValue => {
            // TODO
            unimplemented = true;
        }

        Opcode::SymbolValue => {
            // TODO
            unimplemented = true;
        }

        Opcode::HeapAddr => {
            // TODO
            unimplemented = true;
        }

        Opcode::TableAddr => {
            // TODO
            unimplemented = true;
        }

        Opcode::Nop => {
            // Nothing.
            unimplemented = true;
        }

        Opcode::Select | Opcode::Selectif => {
            // TODO.
            unimplemented = true;
        }

        Opcode::Bitselect => {
            // TODO.
            unimplemented = true;
        }

        Opcode::IsNull | Opcode::IsInvalid | Opcode::Trueif | Opcode::Trueff => {
            // TODO.
            unimplemented = true;
        }

        Opcode::Copy => {
            // TODO
            unimplemented = true;
        }

        Opcode::Breduce | Opcode::Bextend | Opcode::Bint | Opcode::Bmask => {
            // TODO
            unimplemented = true;
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
                _ => {
                    unimplemented = true;
                }
            }
        }

        Opcode::Ireduce | Opcode::Isplit | Opcode::Iconcat => {
            // TODO
            unimplemented = true;
        }

        Opcode::FallthroughReturn => {
            // What is this? The definition says it's a "special
            // instruction" meant to allow falling through into an
            // epilogue that will then return; that just sounds like a
            // normal fallthrough. TODO: Do we need to handle this
            // differently?
            unimplemented = true;
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
            unimplemented = true;
        }

        Opcode::JumpTableEntry => {
            // TODO
            unimplemented = true;
        }

        Opcode::JumpTableBase => {
            // TODO
            unimplemented = true;
        }

        Opcode::Debugtrap => {
            unimplemented = true;
        }

        Opcode::Trap => {
            unimplemented = true;
        }

        Opcode::Trapz | Opcode::Trapnz | Opcode::Trapif | Opcode::Trapff => {
            unimplemented = true;
        }

        Opcode::ResumableTrap => {
            unimplemented = true;
        }

        Opcode::Safepoint => {
            unimplemented = true;
        }

        Opcode::FuncAddr => {
            // TODO
            unimplemented = true;
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
            unimplemented = true;
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

    if unimplemented {
        panic!("lower_insn_to_regs(x64): can't reduce: {:?}", ctx.data(iri));
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
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    ) {
        // A block should end with at most two branches. The first may be a
        // conditional branch; a conditional branch can be followed only by an
        // unconditional branch or fallthrough. Otherwise, if only one branch,
        // it may be an unconditional branch, a fallthrough, a return, or a
        // trap. These conditions are verified by `is_ebb_basic()` during the
        // verifier pass.
        assert!(branches.len() <= 2);

        let mut unimplemented = false;

        if branches.len() == 2 {
            // Must be a conditional branch followed by an unconditional branch.
            let op0 = ctx.data(branches[0]).opcode();
            let op1 = ctx.data(branches[1]).opcode();

            println!(
                "QQQQ lowering two-branch group: opcodes are {:?} and {:?}",
                op0, op1
            );

            assert!(op1 == Opcode::Jump || op1 == Opcode::Fallthrough);
            let taken = BranchTarget::Block(targets[0]);
            let not_taken = match op1 {
                Opcode::Jump => BranchTarget::Block(targets[1]),
                Opcode::Fallthrough => BranchTarget::Block(fallthrough.unwrap()),
                _ => unreachable!(), // assert above.
            };
            match op0 {
                Opcode::Brz | Opcode::Brnz => {
                    let tyS = ctx.input_ty(branches[0], 0);
                    if is_int_ty(tyS) {
                        let rS = ctx.input(branches[0], 0);
                        let cc = match op0 {
                            Opcode::Brz => CC::Z,
                            Opcode::Brnz => CC::NZ,
                            _ => unreachable!(),
                        };
                        let sizeB = int_ty_to_sizeB(tyS);
                        ctx.emit(i_Cmp_RMI_R(sizeB, ip_RMI_I(0), rS));
                        ctx.emit(i_JmpCondSymm(cc, taken, not_taken));
                    } else {
                        unimplemented = true;
                    }
                }
                Opcode::BrIcmp => {
                    let tyS = ctx.input_ty(branches[0], 0);
                    if is_int_ty(tyS) {
                        let rSL = ctx.input(branches[0], 0);
                        let rSR = ctx.input(branches[0], 1);
                        let cc = intCC_to_x64_CC(inst_condcode(ctx.data(branches[0])));
                        let sizeB = int_ty_to_sizeB(tyS);
                        // FIXME verify rSR vs rSL ordering
                        ctx.emit(i_Cmp_RMI_R(sizeB, ip_RMI_R(rSR), rSL));
                        ctx.emit(i_JmpCondSymm(cc, taken, not_taken));
                    } else {
                        unimplemented = true;
                    }
                }
                // TODO: Brif/icmp, Brff/icmp, jump tables
                _ => {
                    unimplemented = true;
                }
            }
        } else {
            assert!(branches.len() == 1);

            // Must be an unconditional branch or trap.
            let op = ctx.data(branches[0]).opcode();
            match op {
                Opcode::Jump => {
                    ctx.emit(i_JmpKnown(BranchTarget::Block(targets[0])));
                }
                Opcode::Fallthrough => {
                    ctx.emit(i_JmpKnown(BranchTarget::Block(targets[0])));
                }
                Opcode::Trap => {
                    unimplemented = true;
                }
                _ => panic!("Unknown branch type!"),
            }
        }

        if unimplemented {
            panic!("lower_branch_group(x64): can't handle: {:?}", branches);
        }
    }
}
