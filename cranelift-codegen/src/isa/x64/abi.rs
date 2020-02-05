//! Implementation of the standard x64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::isa::x64::inst::*;
use crate::isa::x64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, Set};

pub struct X64ABIBody {}

impl X64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(_f: &ir::Function) -> X64ABIBody {
        unimplemented!()
    }
}

impl ABIBody<Inst> for X64ABIBody {
    fn liveins(&self) -> Set<RealReg> {
        unimplemented!()
    }

    fn liveouts(&self) -> Set<RealReg> {
        unimplemented!()
    }

    fn load_arg(&mut self, _idx: usize, _into_reg: Reg, _vcode: &mut VCodeBuilder<Inst>) {
        unimplemented!()
    }

    fn store_retval(&mut self, _idx: usize, _from_reg: Reg, _vcode: &mut VCodeBuilder<Inst>) {
        unimplemented!()
    }

    fn spillslots(&mut self, _slots: usize) {
        unimplemented!()
    }

    fn fixup(&mut self) {
        unimplemented!()
    }

    fn gen_prologue(&self) -> Vec<Inst> {
        unimplemented!()
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        unimplemented!()
    }
}
