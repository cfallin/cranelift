//! This module defines patterns for instruction legalization and code generation.

use crate::{isa_patterns, isa_regs};

isa_regs!(REGINFO,
          banks {
              GPR(  from(0),  to(31), units(32), track_pressure(true), name_prefix(X));
              FPR(  from(32), to(63), units(32), track_pressure(true), name_prefix(V));
              FLAGS(from(64), to(64), units(1), track_pressure(false), names([NZCV]));
          }
          classes {
              // `index` refers to banks above.
              // This descriptor DSL only allows top-level classes, not subclasses.
              // The `mask` arg is a set of three 32-bit bitmasks giving reg ranges.
              GPR(  index(0), bank(0), first(0),  mask([0, 31], [], []));
              FPR(  index(1), bank(1), first(32), mask([], [32, 63], []));
              FLAGS(index(2), bank(2), first(64), mask([], [], [64, 64]));
          });

/*
isa_patterns!(PATTERNS, {
    (Add, rc:GPR, rc:GPR, rc:GPR) => emit(curs, sink) {} size(curs) { 4 };
});
*/
