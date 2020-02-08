//! Pretty-printing for machine code (virtual-registerized or final).

use regalloc::{RealRegUniverse, Reg};

use std::string::{String, ToString};

// FIXME: Should this go into regalloc.rs instead?

/// A trait for printing instruction bits and pieces, with the the ability to
/// take a contextualising RealRegUniverse that is used to give proper names to
/// registers.
pub trait ShowWithRRU {
    /// Return a string that shows the implementing object in context of the
    /// given `RealRegUniverse`, if provided.
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String;
}

impl ShowWithRRU for Reg {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        if self.is_real() {
            if let Some(rru) = mb_rru {
                let reg_ix = self.get_index();
                if reg_ix < rru.regs.len() {
                    return rru.regs[reg_ix].1.to_string();
                } else {
                    // We have a real reg which isn't listed in the universe.  Per
                    // the regalloc.rs interface requirements, this is Totally Not
                    // Allowed.  Print it generically anyway, so we have something
                    // to debug.
                    return format!("!!{:?}!!", self);
                }
            }
        }
        // The reg is virtual, or we have no universe.  Be generic.
        format!("%{:?}", self)
    }
}
