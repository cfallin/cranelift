//! Implementation of the standard ARM64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, Set};

#[derive(Clone, Debug)]
enum ABIArg {
    Reg(RealReg),
    Stack, // TODO
}

#[derive(Clone, Debug)]
enum ABIRet {
    Reg(RealReg),
    Mem, // TODO
}

/// ARM64 ABI object for a function body.
pub struct ARM64ABIBody {
    args: Vec<ABIArg>,
    rets: Vec<ABIRet>,
    spillslots: Option<usize>,
}

fn in_int_reg(ty: types::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        types::B1 | types::B8 | types::B16 | types::B32 | types::B64 => true,
        _ => false,
    }
}

impl ARM64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function) -> ARM64ABIBody {
        println!("ARM64 ABI: func signature {:?}", f.signature);

        // Compute args and retvals from signature.
        let mut args = vec![];
        let mut next_xreg = 0;
        for param in &f.signature.params {
            if &param.purpose == &ir::ArgumentPurpose::Normal
                && in_int_reg(param.value_type)
                && next_xreg < 8
            {
                let x = next_xreg;
                next_xreg += 1;
                let reg = xreg(x).to_real_reg();
                args.push(ABIArg::Reg(reg));
            } else {
                panic!("Unsupported argument in signature: {:?}", f.signature);
            }
        }

        let mut rets = vec![];
        next_xreg = 0;
        for ret in &f.signature.returns {
            if &ret.purpose == &ir::ArgumentPurpose::Normal
                && in_int_reg(ret.value_type)
                && next_xreg < 8
            {
                let x = next_xreg;
                next_xreg += 1;
                let reg = xreg(x).to_real_reg();
                rets.push(ABIRet::Reg(reg));
            } else {
                panic!("Unsupported return value in signature: {:?}", f.signature);
            }
        }

        ARM64ABIBody {
            args,
            rets,
            spillslots: None,
        }
    }
}

impl ABIBody<Inst> for ARM64ABIBody {
    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for arg in &self.args {
            if let &ABIArg::Reg(r) = arg {
                set.insert(r);
            }
        }
        println!("ARM64 ABI: liveins {:?}", set);
        set
    }

    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for ret in &self.rets {
            if let &ABIRet::Reg(r) = ret {
                set.insert(r);
            }
        }
        println!("ARM64 ABI: liveouts {:?}", set);
        set
    }

    fn num_args(&self) -> usize {
        self.args.len()
    }

    fn num_retvals(&self) -> usize {
        self.rets.len()
    }

    fn load_arg(&mut self, idx: usize, into_reg: Reg, vcode: &mut VCodeBuilder<Inst>) {
        match &self.args[idx] {
            &ABIArg::Reg(r) => {
                vcode.push(Inst::gen_move(into_reg, r.to_reg()));
            }
            _ => unimplemented!(),
        }
    }

    fn store_retval(&mut self, idx: usize, from_reg: Reg, vcode: &mut VCodeBuilder<Inst>) {
        match &self.rets[idx] {
            &ABIRet::Reg(r) => {
                vcode.push(Inst::gen_move(r.to_reg(), from_reg));
            }
            _ => unimplemented!(),
        }
    }

    fn spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
        // TODO: compute some sort of stack-frame layout information
    }

    fn gen_prologue(&self) -> Vec<Inst> {
        vec![]
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        vec![Inst::Ret {}]
    }
}
