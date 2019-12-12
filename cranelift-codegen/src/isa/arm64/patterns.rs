//! This module defines patterns for instruction legalization and code generation.

/* TODO:
 * - AGen
 * - sign/zero-extend, bit-shift
 */

use crate::ir::Opcode;
use crate::isadef::*;
use smallvec::SmallVec;

type Arm64InstArgs = SmallVec<[Arm64Arg; 3]>;
type Arm64InstImms = SmallVec<[Arm64Imm; 2]>;

struct Arm64Inst {
    op: Arm64InstOp,
    args: Arm64InstArgs,
    cond: Option<Arm64InstCond>,
}

enum Arm64InstOp {
    Add,
    AddS,
    Sub,
    SubS,
    Cmp,
    Cmn,
    Neg,
    NegS,
    Mov,
    MovI,
    And,
    AndS,
    Orr,
    Orn,
    Eor,
    Eon,
    Bic,
    BicS,
    Tst,
    Asr,
    Lsl,
    Lsr,
    Ror,
    Asrv,
    Lslv,
    Lsrv,
    Rorv,
    Cls,
    Clz,
    Adc,
    AdcS,
    Csel,
}

enum Arm64Arg {
    Imm(Arm64ShiftedImm),
    Reg(MachReg),
    ShiftedReg(MachReg, Arm64ShiftOp, usize),
    ExtendedReg(MachReg, Arm64ExtendOp, usize),
    Mem(Arm64MemArg),
}

struct Arm64ShiftedImm {
    bits: usize,
    shift: usize,
}

enum Arm64ShiftOp {
    ASR,
    LSR,
    LSL,
    ROR,
}

enum Arm64ExtendOp {
    SXTB,
    SXTH,
    SXTW,
    XSTX,
    UXTB,
    UXTH,
    UXTW,
    UXTX,
}

struct Argm64MemArg {
    base: Option<MachReg>,
    offset: Option<MachReg>,
    shift: Option<usize>,
    inc: Arm64MemInc,
}

enum Arm64MemInc {
    None,
    Pre,
    Post,
}

// TODO: implement machine-instruction bits here. The MachInst infra is essentially:
//
// - A normalized interface to machine code (opaque insts with register slots) for regalloc and
//   binemit.
// - An API to the lowering patterns to allow principled emission of arm64 code.
impl MachInst for Arm64Inst {

}

// TODO: lowering by pattern-matching. Can we just handwrite this? No need for pattern-matching if
// we explicitly codegen with context (reg, addr-mode, different types of imm) in mind. Analogous
// to recursive-descent parser (fast!) vs. backtracking / pattern-matching (slow).


/*
fn build_backend() -> IsaDef {
    let mut b = IsaDef::new();

    let rb_GPR = b
        .reg_bank("GPR")
        .regs(32)
        .track_pressure(true)
        .name_prefix("X")
        .build();
    let rb_FPR = b
        .reg_bank("FPR")
        .regs(32)
        .track_pressure(true)
        .name_prefix("V")
        .build();
    let rb_FLAGS = b.reg_bank("FLAGS").regs(1).names(&["NZCV"]).build();
    let GPR = b.reg_class("GPR").bank(rb_GPR).build();
    let FPR = b.reg_class("FPR").bank(rb_FPR).build();
    let FLAGS = b.reg_class("FLAGS").bank(rb_FLAGS).build();

    // Legalize example: convert operand of any ld/st that is not an AGen into an AGen.
    // Legalize example: convert AGen with add/sub/shift args into improved AGen.

    // TODO: predicates on match. Allowed to examine captured subinsts too, and given cursor to
    // func.

    b.emit_pat(
        Opcode::Iadd,
        &[b.out_rc(0, GPR), b.in_rc(1, GPR), b.in_rc(2, GPR)],
        |ctx, args| {
            ctx.bits(11, 0b1000_1011_001);
            ctx.reg(&args[2]); // Rm
            ctx.bits(6, 0b000_000);
            ctx.reg(&args[1]); // Rn
            ctx.reg(&args[0]); // Rd
        },
    );

    b
}
*/
