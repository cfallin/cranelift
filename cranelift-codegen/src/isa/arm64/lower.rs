//! Lowering rules for ARM64.

use crate::ir::types::*;
use crate::ir::{InstructionData, Opcode, Type};

use crate::isa::arm64::inst::*;
use crate::isa::arm64::registers::*;

/*
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

    lower_pattern!(t, Iconst, |ctx| {
        let value = fieldref!(ctx.instdata(ctx.inst()), UnaryImm, imm).bits();
        let out = ctx.output(0);
        load_imm(ctx, value as u64, out);
        true
    });

    lower_pattern!(t, F32const, |ctx| {
        let value: u32 = field!(ctx.instdata(ctx.inst()), UnaryIeee32, imm).bits();
        let out = ctx.output(0);
        load_imm(ctx, value as u64, out);
        true
    });

    lower_pattern!(t, F64const, |ctx| {
        let value: u64 = field!(ctx.instdata(ctx.inst()), UnaryIeee64, imm).bits();
        let out = ctx.output(0);
        load_imm(ctx, value, out);
        true
    });

    lower_pattern!(t, Bconst, |ctx| {
        let ty = ctx.ty(ctx.inst());
        let value: u64 = match ty {
            B1 => 1,
            B8 => 0xff,
            B16 => 0xffff,
            B32 => 0xffff_ffff,
            B64 => 0xffff_ffff_ffff_ffff,
            _ => unimplemented!(),
        };
        let out = ctx.output(0);
        load_imm(ctx, value, out);
        true
    });

    lower_pattern!(t, Null, |ctx| {
        let out = ctx.output(0);
        load_imm(ctx, 0, out);
        true
    });

    lower_pattern!(t, (Iadd Iconst _), |ctx| {
        let iconst = ctx.input_inst(ctx.inst(), 0);
        let value = fieldref!(ctx.instdata(iconst), UnaryImm, imm).bits();
        let op = choose_32_64(ctx.ty(ctx.inst()), Op::AddI32, Op::AddI64);
        with_imm12(ctx, value as u64, |ctx, imm| {
            ctx.emit(make_reg_reg_imm(op,
                                      ctx.output(0),
                                      ctx.input(0),
                                      imm));
            ctx.mark_unused(iconst);
        })
    });

    lower_pattern!(t, (Iadd _ Iconst), |ctx| {
        let iconst = ctx.input_inst(ctx.inst(), 1);
        let value = fieldref!(ctx.instdata(iconst), UnaryImm, imm).bits();
        let op = choose_32_64(ctx.ty(ctx.inst()), Op::AddI32, Op::AddI64);
        with_imm12(ctx, value as u64, |ctx, imm| {
            ctx.emit(make_reg_reg_imm(op,
                                      ctx.output(0),
                                      ctx.input(1),
                                      imm));
            ctx.mark_unused(iconst);
        })
    });

    lower_pattern!(t, Iadd, |ctx| {
        ctx.emit(make_reg_reg_reg(
            choose_32_64(ctx.ty(ctx.inst()), Op::Add32, Op::Add64),
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Isub, |ctx| {
        ctx.emit(make_reg_reg_reg(
            choose_32_64(ctx.ty(ctx.inst()), Op::Sub32, Op::Sub64),
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Ineg, |ctx| {
        let op = choose_32_64(ctx.ty(ctx.inst()), Op::Neg32, Op::Neg64);
        ctx.emit(make_reg_reg(op, ctx.output(0), ctx.input(0)));
        true
    });

    lower_pattern!(t, Imul, |ctx| {
        ctx.emit(make_reg_reg_reg(
            Op::SMulL,
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Umulhi, |ctx| {
        ctx.emit(make_reg_reg_reg(
            Op::UMulH,
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Smulhi, |ctx| {
        ctx.emit(make_reg_reg_reg(
            Op::SMulH,
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Udiv, |ctx| {
        ctx.emit(make_reg_reg_reg(
            Op::UDiv,
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Sdiv, |ctx| {
        ctx.emit(make_reg_reg_reg(
            Op::SDiv,
            ctx.output(0),
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Urem, |ctx| {
        let quotient = ctx.rc_tmp(I64, GPR);
        ctx.emit(make_reg_reg_reg_reg(
            Op::UMSubL,
            ctx.output(0),
            quotient.clone(),
            ctx.input(1),
            ctx.input(0),
        ));
        ctx.emit(make_reg_reg_reg(
            Op::UDiv,
            quotient,
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    lower_pattern!(t, Srem, |ctx| {
        let quotient = ctx.rc_tmp(I64, GPR);
        ctx.emit(make_reg_reg_reg_reg(
            Op::SMSubL,
            ctx.output(0),
            quotient.clone(),
            ctx.input(1),
            ctx.input(0),
        ));
        ctx.emit(make_reg_reg_reg(
            Op::SDiv,
            quotient,
            ctx.input(0),
            ctx.input(1),
        ));
        true
    });

    t
}
*/
