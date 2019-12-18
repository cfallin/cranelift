//! Lowering rules for ARM64.

use crate::ir::types::*;
use crate::ir::{InstructionData, Opcode, Type};
use crate::machinst::lower::*;
use crate::machinst::pattern::*;

use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::*;
use crate::lower_pattern;

/// Helper: in a lowering action, get a field of an instruction, or fail.
macro_rules! field {
    ($inst:expr, $fmt:ident, $field:ident) => {
        match $inst {
            InstructionData::$fmt { $field, .. } => $field,
            _ => {
                return false;
            }
        }
    };
}

/// Helper: in a lowering action, get a reference to a field of an instruction, or fail.
macro_rules! fieldref {
    ($inst:expr, $fmt:ident, $field:ident) => {
        match $inst {
            InstructionData::$fmt { ref $field, .. } => $field,
            _ => {
                return false;
            }
        }
    };
}

/// Create the lowering table for ARM64.
pub fn make_backend() -> LowerTable<Op, Arg> {
    let mut t = LowerTable::new();

    // -- branches / control flow (23)
    // Jump
    // Fallthrough
    // Brz
    // Brnz
    // BrIcmp
    // Brif
    // Brff
    // BrTable
    // JumpTableEntry
    // JumpTableBase
    // IndirectJumpTableBr
    // Debugtrap
    // Trap
    // Trapz
    // ResumableTrap
    // Trapnz
    // Trapif
    // Trapff
    // Return
    // FallthroughReturn
    // Call
    // CallIndirect
    // FuncAddr

    // -- memory (31)
    // Load
    // LoadComplex
    // Store
    // StoreComplex
    // Uload8
    // Uload8Complex
    // Sload8
    // Sload8Complex
    // Istore8
    // Istore8Complex
    // Uload16
    // Uload16Complex
    // Sload16
    // Sload16Complex
    // Istore16
    // Istore16Complex
    // Uload32
    // Uload32Complex
    // Sload32
    // Sload32Complex
    // Istore32
    // Istore32Complex
    // StackLoad
    // StackStore
    // StackAddr
    // GlobalValue
    // SymbolValue
    // HeapAddr
    // GetPinnedReg
    // SetPinnedReg
    // TableAddr

    // -- arith (112)
    // Vconst
    // Shuffle
    // Select
    // Selectif
    // Bitselect
    // Icmp
    // IcmpImm
    // Ifcmp
    // IfcmpImm
    //
    // IaddImm
    // ImulImm
    // UdivImm
    // SdivImm
    // UremImm
    // SremImm
    // IrsubImm
    // IaddCin
    // IaddIfcin
    // IaddCout
    // IaddIfcout
    // IaddCarry
    // IaddIfcarry
    // IsubBin
    // IsubIfbin
    // IsubBout
    // IsubIfbout
    // IsubBorrow
    // IsubIfborrow
    // UaddSat
    // SaddSat
    // UsubSat
    // SsubSat
    // Band
    // Bor
    // Bxor
    // Bnot
    // BandNot
    // BorNot
    // BxorNot
    // BandImm
    // BorImm
    // BxorImm
    // Rotl
    // Rotr
    // RotlImm
    // RotrImm
    // Ishl
    // Ushr
    // Sshr
    // IshlImm
    // UshrImm
    // SshrImm
    // Bitrev
    // Clz
    // Cls
    // Ctz
    // Popcnt
    // Fcmp
    // Ffcmp
    // Fadd
    // Fsub
    // Fmul
    // Fdiv
    // Sqrt
    // Fma
    // Fneg
    // Fabs
    // Fcopysign
    // Fmin
    // Fmax
    // Ceil
    // Floor
    // Trunc
    // Nearest
    // IsNull
    // Trueif
    // Trueff
    // Bitcast
    // RawBitcast
    // ScalarToVector
    // Breduce
    // Bextend
    // Bint
    // Bmask
    // Ireduce
    // Uextend
    // Sextend
    // Fpromote
    // Fdemote
    // FcvtToUint
    // FcvtToUintSat
    // FcvtToSint
    // FcvtToSintSat
    // FcvtFromUint
    // FcvtFromSint
    // Isplit
    // Iconcat
    //
    // -- misc (calling conv, regalloc, ...) (16)
    // Nop
    // Copy
    // Spill
    // Fill
    // FillNop
    // Regmove
    // CopySpecial
    // CopyToSsa
    // CopyNop
    // AdjustSpDown
    // AdjustSpUpImm
    // AdjustSpDownImm
    // IfcmpSp
    // Regspill
    // Regfill
    // Safepoint

    // -- vectors (8)
    // Vsplit
    // Vconcat
    // Vselect
    // VanyTrue
    // VallTrue
    // Splat
    // Insertlane
    // Extractlane

    lower_pattern!(t, Iconst, |ctx, inst| {
        let value = fieldref!(ctx.inst(inst), UnaryImm, imm).bits();
        load_imm(ctx, value as u64, ctx.output(inst, 0));
        true
    });

    lower_pattern!(t, F32const, |ctx, inst| {
        let value: u32 = field!(ctx.inst(inst), UnaryIeee32, imm).bits();
        load_imm(ctx, value as u64, ctx.output(inst, 0));
        true
    });

    lower_pattern!(t, F64const, |ctx, inst| {
        let value: u64 = field!(ctx.inst(inst), UnaryIeee64, imm).bits();
        load_imm(ctx, value, ctx.output(inst, 0));
        true
    });

    lower_pattern!(t, Bconst, |ctx, inst| {
        let ty = ctx.ty(inst);
        let value: u64 = match ty {
            B1 => 1,
            B8 => 0xff,
            B16 => 0xffff,
            B32 => 0xffff_ffff,
            B64 => 0xffff_ffff_ffff_ffff,
            _ => unimplemented!(),
        };
        load_imm(ctx, value, ctx.output(inst, 0));
        true
    });

    lower_pattern!(t, Null, |ctx, inst| {
        load_imm(ctx, 0, ctx.output(inst, 0));
        true
    });

    lower_pattern!(t, (Iadd Iconst _), |ctx, inst| {
        let iconst = ctx.input_inst(inst, 0);
        let value = fieldref!(ctx.inst(iconst), UnaryImm, imm).bits();
        with_imm12(ctx, value as u64, |ctx, imm| {
            ctx.emit(make_reg_reg_imm(Op::AddI,
                                      ctx.output(inst, 0),
                                      ctx.input(inst, 0),
                                      imm));
            ctx.unused(iconst);
        })
    });

    lower_pattern!(t, (Iadd _ Iconst), |ctx, inst| {
        let iconst = ctx.input_inst(inst, 1);
        let value = fieldref!(ctx.inst(iconst), UnaryImm, imm).bits();
        with_imm12(ctx, value as u64, |ctx, imm| {
            ctx.emit(make_reg_reg_imm(Op::AddI,
                                      ctx.output(inst, 0),
                                      ctx.input(inst, 1),
                                      imm));
            ctx.unused(iconst);
        })
    });

    lower_pattern!(t, Iadd, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::Add,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Isub, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::Sub,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Ineg, |ctx, inst| {
        ctx.emit(make_reg_reg(
            Op::Neg,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
        ));
        true
    });

    lower_pattern!(t, Imul, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::SMulL,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Umulhi, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::UMulH,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Smulhi, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::SMulH,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Udiv, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::UDiv,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Sdiv, |ctx, inst| {
        ctx.emit(make_reg_reg_reg(
            Op::SDiv,
            ctx.output(inst, 0),
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Urem, |ctx, inst| {
        let quotient = ctx.tmp_rc(GPR);
        ctx.emit(make_reg_reg_reg_reg(
            Op::UMSubL,
            ctx.output(inst, 0),
            quotient.clone(),
            ctx.input(inst, 1),
            ctx.input(inst, 0),
        ));
        ctx.emit(make_reg_reg_reg(
            Op::UDiv,
            quotient,
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    lower_pattern!(t, Srem, |ctx, inst| {
        let quotient = ctx.tmp_rc(GPR);
        ctx.emit(make_reg_reg_reg_reg(
            Op::SMSubL,
            ctx.output(inst, 0),
            quotient.clone(),
            ctx.input(inst, 1),
            ctx.input(inst, 0),
        ));
        ctx.emit(make_reg_reg_reg(
            Op::SDiv,
            quotient,
            ctx.input(inst, 0),
            ctx.input(inst, 1),
        ));
        true
    });

    t
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::condcodes::*;
    use crate::ir::types::*;
    use crate::ir::{Function, InstBuilder, ValueDef};
    use crate::machinst::lower::*;

    #[test]
    fn test_lower_simple_match() {
        let mut func = Function::new();
        let ebb0 = func.dfg.make_ebb();
        let arg0 = func.dfg.append_ebb_param(ebb0, I32);
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_ebb(ebb0);

        let v0 = pos.ins().iconst(I32, 1);
        let v1 = pos.ins().iadd(arg0, v0);
        let v2 = pos.ins().isub(arg0, v1);
        let v3 = pos.ins().iadd(v2, v2);

        let ins0 = func.dfg.value_def(v0).unwrap_inst();
        let ins1 = func.dfg.value_def(v1).unwrap_inst();
        let ins2 = func.dfg.value_def(v2).unwrap_inst();
        let ins3 = func.dfg.value_def(v3).unwrap_inst();

        let t = make_backend();

        let lowered = LowerResult::lower(&func, &t);

        println!("func: {:?}", func);
        println!("lowered: {:?}", lowered);

        assert!(lowered.insts().len() == 3);
    }
}
