//! Implementation of the standard x64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::StackSlot;
use crate::ir::Type;
use crate::isa::x64::inst::*;
use crate::isa::x64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, WritableReg};

pub struct X64ABIBody {}

impl X64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(_f: &ir::Function) -> X64ABIBody {
        unimplemented!()
    }
}

impl ABIBody<Inst> for X64ABIBody {
    fn num_args(&self) -> usize {
        unimplemented!()
    }

    fn num_retvals(&self) -> usize {
        unimplemented!()
    }

    fn num_stackslots(&self) -> usize {
        unimplemented!()
    }

    fn liveins(&self) -> Set<RealReg> {
        unimplemented!()
    }

    fn liveouts(&self) -> Set<RealReg> {
        unimplemented!()
    }

    fn load_arg(&self, _idx: usize, _into_reg: WritableReg<Reg>) -> Inst {
        unimplemented!()
    }

    fn store_retval(&self, _idx: usize, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn set_num_spillslots(&mut self, _slots: usize) {
        unimplemented!()
    }

    fn set_clobbered(&mut self, _clobbered: Set<WritableReg<RealReg>>) {
        unimplemented!()
    }

    fn load_stackslot(&self, _slot: StackSlot, _offset: usize, _ty: Type, _into_reg: WritableReg<Reg>) -> Inst {
        unimplemented!()
    }

    fn store_stackslot(&self, _slot: StackSlot, _offset: usize, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn load_spillslot(&self, _slot: SpillSlot, _ty: Type, _into_reg: WritableReg<Reg>) -> Inst {
        unimplemented!()
    }

    fn store_spillslot(&self, _slot: SpillSlot, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_prologue(&self) -> Vec<Inst> {
        unimplemented!()
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        unimplemented!()
    }

    fn get_spillslot_size(&self, rc: RegClass, ty: Type) -> u32 {
        // We allocate in terms of 8-byte slots.
        match (rc, ty) {
            (RegClass::I64, _) => 1,
            (RegClass::V128, F32) | (RegClass::V128, F64) => 1,
            (RegClass::V128, _) => 2,
            _ => panic!("Unexpected register class!"),
        }
    }

    fn gen_spill(&self, _to_slot: SpillSlot, _from_reg: RealReg, _ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_reload(&self, _to_reg: WritableReg<RealReg>, _from_slot: SpillSlot, _ty: Type) -> Inst {
        unimplemented!()
    }
}
