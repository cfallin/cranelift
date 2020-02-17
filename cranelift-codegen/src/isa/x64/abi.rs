//! Implementation of the standard x64 ABI.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::StackSlot;
use crate::ir::Type;
use crate::isa::x64::inst::*;
use crate::isa::x64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};

// Clone of arm64 version
#[derive(Clone, Debug)]
enum ABIArg {
    Reg(RealReg),
    Stack, // TODO
}

// Clone of arm64 version
#[derive(Clone, Debug)]
enum ABIRet {
    Reg(RealReg),
    Mem, // TODO
}

// Clone of arm64 version
pub struct X64ABIBody {
    args: Vec<ABIArg>,
    rets: Vec<ABIRet>,
    stackslots: Vec<usize>,            // offsets to each stackslot
    stackslots_size: usize,            // total stack size of all stackslots
    clobbered: Set<Writable<RealReg>>, // clobbered registers, from regalloc.
    spillslots: Option<usize>,         // total number of spillslots, from regalloc.
}

// Clone of arm64 version
fn in_int_reg(ty: types::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        types::B1 | types::B8 | types::B16 | types::B32 | types::B64 => true,
        _ => false,
    }
}

fn get_intreg_for_arg_ELF(idx: usize) -> Option<Reg> {
    match idx {
        0 => Some(reg_RDI()),
        1 => Some(reg_RSI()),
        2 => Some(reg_RDX()),
        3 => Some(reg_RCX()),
        4 => Some(reg_R8()),
        5 => Some(reg_R9()),
        _ => None,
    }
}

fn get_intreg_for_retval_ELF(idx: usize) -> Option<Reg> {
    match idx {
        0 => Some(reg_RAX()),
        1 => Some(reg_RDX()), // is that correct?
        _ => None,
    }
}

impl X64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function) -> X64ABIBody {
        println!("X64 ABI: func signature {:?}", f.signature);

        // Compute args and retvals from signature.
        let mut args = vec![];
        let mut next_int_arg = 0;
        for param in &f.signature.params {
            let mut ok = false;
            if &param.purpose == &ir::ArgumentPurpose::Normal && in_int_reg(param.value_type) {
                if let Some(reg) = get_intreg_for_arg_ELF(next_int_arg) {
                    args.push(ABIArg::Reg(reg.to_real_reg()));
                    ok = true;
                }
                next_int_arg += 1;
            }
            if !ok {
                panic!("Unsupported argument in signature: {:?}", f.signature);
            }
        }

        let mut rets = vec![];
        let mut next_int_retval = 0;
        for ret in &f.signature.returns {
            let mut ok = false;
            if &ret.purpose == &ir::ArgumentPurpose::Normal && in_int_reg(ret.value_type) {
                if let Some(reg) = get_intreg_for_retval_ELF(next_int_retval) {
                    rets.push(ABIRet::Reg(reg.to_real_reg()));
                    ok = true;
                }
                next_int_retval += 1;
            }
            if !ok {
                panic!("Unsupported return value in signature: {:?}", f.signature);
            }
        }

        // Compute stackslot locations and total stackslot size.
        let mut stack_offset: usize = 0;
        let mut stackslots = vec![];
        for (stackslot, data) in f.stack_slots.iter() {
            let off = stack_offset;
            stack_offset += data.size as usize;
            stack_offset = (stack_offset + 7) & !7usize;
            assert_eq!(stackslot.as_u32() as usize, stackslots.len());
            stackslots.push(off);
        }

        X64ABIBody {
            args,
            rets,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
        }
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

    // Clone of arm64 version
    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for arg in &self.args {
            if let &ABIArg::Reg(r) = arg {
                set.insert(r);
            }
        }
        println!("X64 ABI: liveins {:?}", set);
        set
    }

    // Clone of arm64 version
    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for ret in &self.rets {
            if let &ABIRet::Reg(r) = ret {
                set.insert(r);
            }
        }
        println!("X64 ABI: liveouts {:?}", set);
        set
    }

    fn gen_copy_arg_to_reg(&self, idx: usize, to_reg: Writable<Reg>) -> Inst {
        if let Some(from_reg) = get_intreg_for_arg_ELF(idx) {
            return i_Mov_R_R(/*is64=*/ true, from_reg, to_reg);
        }
        unimplemented!()
    }

    fn gen_copy_reg_to_retval(&self, idx: usize, from_reg: Reg) -> Inst {
        if let Some(to_reg) = get_intreg_for_retval_ELF(idx) {
            return i_Mov_R_R(
                /*is64=*/ true,
                from_reg,
                Writable::<Reg>::from_reg(to_reg),
            );
        }
        unimplemented!()
    }

    fn gen_ret(&self) -> Inst {
        unimplemented!()
    }

    fn set_num_spillslots(&mut self, _slots: usize) {
        unimplemented!()
    }

    fn set_clobbered(&mut self, _clobbered: Set<Writable<RealReg>>) {
        unimplemented!()
    }

    fn load_stackslot(
        &self,
        _slot: StackSlot,
        _offset: usize,
        _ty: Type,
        _into_reg: Writable<Reg>,
    ) -> Inst {
        unimplemented!()
    }

    fn store_stackslot(&self, _slot: StackSlot, _offset: usize, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn load_spillslot(&self, _slot: SpillSlot, _ty: Type, _into_reg: Writable<Reg>) -> Inst {
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

    fn gen_reload(&self, _to_reg: Writable<RealReg>, _from_slot: SpillSlot, _ty: Type) -> Inst {
        unimplemented!()
    }
}
