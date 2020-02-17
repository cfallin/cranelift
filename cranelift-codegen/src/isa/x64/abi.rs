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
    // Calculated while creating the prologue, and used when creating the
    // epilogue.  Amount by which RSP is adjusted downwards to allocate the
    // spill area.
    spill_area_sizeB: Option<usize>,
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

fn is_callee_save_ELF(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => match r.get_hw_encoding() as u8 {
            ENC_RBX | ENC_RBP | ENC_R12 | ENC_R13 | ENC_R14 | ENC_R15 => true,
            _ => false,
        },
        _ => unimplemented!(),
    }
}

// Clone of arm64 version
fn get_callee_saves(regs: Vec<Writable<RealReg>>) -> Vec<Writable<RealReg>> {
    regs.into_iter()
        .filter(|r| is_callee_save_ELF(r.to_reg()))
        .collect()
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
            spill_area_sizeB: None,
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
        i_Ret()
    }

    // Clone of arm64
    fn set_num_spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
    }

    // Clone of arm64
    fn set_clobbered(&mut self, clobbered: Set<Writable<RealReg>>) {
        self.clobbered = clobbered;
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

    fn gen_prologue(&mut self) -> Vec<Inst> {
        let mut insts = vec![];
        let total_stacksize = self.stackslots_size + 8 * self.spillslots.unwrap();
        let total_stacksize = (total_stacksize + 15) & !15; // 16-align the stack

        let r_rbp = reg_RBP();
        let r_rsp = reg_RSP();
        let w_rbp = Writable::<Reg>::from_reg(r_rbp);
        let w_rsp = Writable::<Reg>::from_reg(r_rsp);

        // The "traditional" pre-preamble
        // RSP before the call will be 0 % 16.  So here, it is 8 % 16.
        insts.push(i_Push64(ip_RMI_R(r_rbp)));
        // RSP is now 0 % 16
        insts.push(i_Mov_R_R(true, r_rsp, w_rbp));

        // Save callee saved registers that we trash.  Keep track of how much
        // space we've used, so as to know what we have to do to get the base
        // of the spill area 0 % 16.
        let mut callee_saved_used = 0;
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg in clobbered {
            let r_reg = reg.to_reg();
            match r_reg.get_class() {
                RegClass::I64 => {
                    insts.push(i_Push64(ip_RMI_R(r_reg.to_reg())));
                    callee_saved_used += 8;
                }
                _ => unimplemented!(),
            }
        }

        // Allocate the frame.  Now, be careful: RSP may now not be 0 % 16.
        // If it isn't, increase total_stacksize to compensate.  Because
        // total_stacksize is 0 % 16, this ensures that RSP after this
        // subtraction, is still 16 aligned.
        let spill_area_sizeB = total_stacksize + ((16 - callee_saved_used) % 16);
        if spill_area_sizeB >= 0 {
            // FIXME JRS 2020Feb16: what if spill_area_size >= 2G?
            insts.push(i_Alu_RMI_R(
                true,
                RMI_R_Op::Sub,
                ip_RMI_I(spill_area_sizeB as u32),
                w_rsp,
            ));
        }
        debug_assert!(self.spill_area_sizeB.is_none());
        // Stash this value.  We'll need it for the epilogue.
        self.spill_area_sizeB = Some(spill_area_sizeB);

        insts
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        let mut insts = vec![];
        let r_rbp = reg_RBP();
        let r_rsp = reg_RSP();
        let w_rbp = Writable::<Reg>::from_reg(r_rbp);
        let w_rsp = Writable::<Reg>::from_reg(r_rsp);

        // Undo what we did in the prologue.

        // Clear the spill area and the 16-alignment padding below it.
        debug_assert!(self.spill_area_sizeB.is_some());
        let spill_area_sizeB = self.spill_area_sizeB.unwrap();
        if spill_area_sizeB >= 0 {
            // FIXME JRS 2020Feb16: what if spill_area_size >= 2G?
            insts.push(i_Alu_RMI_R(
                true,
                RMI_R_Op::Add,
                ip_RMI_I(spill_area_sizeB as u32),
                w_rsp,
            ));
        }

        // Restore regs.
        let mut tmp_insts = vec![];
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for w_real_reg in clobbered {
            match w_real_reg.to_reg().get_class() {
                RegClass::I64 => {
                    // TODO: make these conversion sequences less cumbersome.
                    tmp_insts.push(i_Pop64(Writable::<Reg>::from_reg(
                        w_real_reg.to_reg().to_reg(),
                    )))
                }
                _ => unimplemented!(),
            }
        }
        tmp_insts.reverse();
        for i in tmp_insts {
            insts.push(i);
        }

        // Undo the "traditional" pre-preamble
        // RSP before the call will be 0 % 16.  So here, it is 8 % 16.
        // uhhhh .. insts.push(i_Mov_R_R(true, r_rbp, w_rsp));
        insts.push(i_Pop64(w_rbp));

        insts.push(i_Ret());
        insts
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
