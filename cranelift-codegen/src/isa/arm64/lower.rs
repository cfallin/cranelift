//! Lowering rules for ARM64.

use crate::ir::Opcode;
use crate::machinst::lower::*;
use crate::machinst::pattern::*;

use crate::isa::arm64::inst::*;
use crate::lower_pattern;

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
    // Iconst
    // F32const
    // F64const
    // Bconst
    // Vconst
    // Shuffle
    // Null
    // Select
    // Selectif
    // Bitselect
    // Icmp
    // IcmpImm
    // Ifcmp
    // IfcmpImm
    // Iadd
    // UaddSat
    // SaddSat
    // Isub
    // UsubSat
    // SsubSat
    // Ineg
    // Imul
    // Umulhi
    // Smulhi
    // Udiv
    // Sdiv
    // Urem
    // Srem
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
        with_imm12(ctx, inst, |ctx, imm| {
            let xzr = ctx.fixed(31);
            ctx.emit(make_reg_reg_imm(Op::AddI, ctx.output(inst, 0), xzr, imm));
        })
    });

    lower_pattern!(t, (Iadd Iconst _), |ctx, inst| {
        let iconst = ctx.input_inst(inst, 0);
        with_imm12(ctx, iconst, |ctx, imm| {
            ctx.emit(make_reg_reg_imm(Op::AddI,
                                      ctx.output(inst, 0),
                                      ctx.input(inst, 0),
                                      imm));
            ctx.unused(iconst);
        })
    });

    lower_pattern!(t, (Iadd _ Iconst), |ctx, inst| {
        let iconst = ctx.input_inst(inst, 1);
        with_imm12(ctx, iconst, |ctx, imm| {
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
