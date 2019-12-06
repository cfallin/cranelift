//! This module defines patterns for instruction legalization and code generation.

use crate::isadef::{PatternAction, isa_patterns, isa_regclass};

// TODO: overall macro to create static RegInfo, and define separate toplevel bindings for GPR,
// FPR, X0..31, etc.

isa_regs!(REGINFO, {
    Bank(GPR,   from(0),  to(31), units(32), track_pressure(true), name_prefix(X));
    Bank(FPR,   from(32), to(63), units(32), track_pressure(true), name_prefix(V));
    Bank(Flags, from(64), to(64), units(1), track_pressure(false), names([NZCV]));
    Class(GPR,  
});

isa_patterns!(PATTERNS, {
    (Add, rc:GPR, rc:GPR, rc:GPR) => emit(curs, sink) {} size(curs) { 4 };
});
