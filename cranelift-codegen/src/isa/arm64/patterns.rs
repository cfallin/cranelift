//! This module defines patterns for instruction legalization and code generation.

/* TODO:
 * - AGen
 * - sign/zero-extend, bit-shift
 */

use crate::ir::Opcode;
use crate::isadef::*;

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
