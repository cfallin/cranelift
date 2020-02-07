//! Implementation of the standard ARM64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::ir::StackSlot;
use crate::ir::Type;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, Set, SpillSlot};

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
    stackslots: Vec<usize>,    // offsets to each stackslot
    stackslots_size: usize,    // total stack size of all stackslots
    clobbered: Set<RealReg>,   // clobbered registers, from regalloc.
    spillslots: Option<usize>, // total number of spillslots, from regalloc.
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

        ARM64ABIBody {
            args,
            rets,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
        }
    }
}

// Get a sequence of instructions and a memory argument that together
// will compute the address of a location on the stack, relative to FP.
fn get_stack_addr(fp_offset: i64) -> (Vec<Inst>, MemArg) {
    if let Some(simm9) = SImm9::maybe_from_i64(fp_offset) {
        let mem = MemArg::BaseSImm9(fp_reg(), simm9);
        (vec![], mem)
    } else {
        let tmp = spilltmp_reg();
        let const_data = u64_constant(fp_offset as u64);
        let const_inst = Inst::ULoad64 {
            rd: tmp,
            mem: MemArg::label(MemLabel::ConstantData(const_data)),
        };
        let add_inst = Inst::AluRRR {
            alu_op: ALUOp::Add64,
            rd: tmp,
            rn: tmp,
            rm: fp_reg(),
        };
        (vec![const_inst, add_inst], MemArg::Base(tmp))
    }
}

fn load_stack(fp_offset: i64, into_reg: Reg, ty: Type) -> Vec<Inst> {
    let (mut insts, mem) = get_stack_addr(fp_offset);

    let inst = match ty {
        types::B1 | types::B8 | types::I8 => Inst::ULoad8 { rd: into_reg, mem },
        types::B16 | types::I16 => Inst::ULoad16 { rd: into_reg, mem },
        types::B32 | types::I32 => Inst::ULoad32 { rd: into_reg, mem },
        types::B64 | types::I64 => Inst::ULoad64 { rd: into_reg, mem },
        _ => unimplemented!(),
    };
    insts.push(inst);
    insts
}

fn store_stack(fp_offset: i64, from_reg: Reg, ty: Type) -> Vec<Inst> {
    let (mut insts, mem) = get_stack_addr(fp_offset);

    let inst = match ty {
        types::B1 | types::B8 | types::I8 => Inst::Store8 { rd: from_reg, mem },
        types::B16 | types::I16 => Inst::Store16 { rd: from_reg, mem },
        types::B32 | types::I32 => Inst::Store32 { rd: from_reg, mem },
        types::B64 | types::I64 => Inst::Store64 { rd: from_reg, mem },
        _ => unimplemented!(),
    };
    insts.push(inst);
    insts
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

    fn num_stackslots(&self) -> usize {
        self.stackslots.len()
    }

    fn load_arg(&self, idx: usize, into_reg: Reg) -> Vec<Inst> {
        match &self.args[idx] {
            &ABIArg::Reg(r) => {
                return vec![Inst::gen_move(into_reg, r.to_reg())];
            }
            _ => unimplemented!(),
        }
    }

    fn store_retval(&self, idx: usize, from_reg: Reg) -> Vec<Inst> {
        match &self.rets[idx] {
            &ABIRet::Reg(r) => {
                return vec![Inst::gen_move(r.to_reg(), from_reg)];
            }
            _ => unimplemented!(),
        }
    }

    fn set_num_spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
    }

    fn set_clobbered(&mut self, clobbered: Set<RealReg>) {
        self.clobbered = clobbered;
    }

    fn load_stackslot(&self, slot: StackSlot, offset: usize, ty: Type, into_reg: Reg) -> Vec<Inst> {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        load_stack(fp_off, into_reg, ty)
    }

    fn store_stackslot(
        &self,
        slot: StackSlot,
        offset: usize,
        ty: Type,
        from_reg: Reg,
    ) -> Vec<Inst> {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        store_stack(fp_off, from_reg, ty)
    }

    // Load from a spillslot.
    fn load_spillslot(&self, slot: SpillSlot, ty: Type, into_reg: Reg) -> Vec<Inst> {
        // Offset from beginning of spillslot area, which is FP - stackslot_size - 8*spillslots.
        let slot = slot.get() as i64;
        let fp_off: i64 =
            -(self.stackslots_size as i64) - (8 * self.spillslots.unwrap() as i64) + 8 * slot;
        load_stack(fp_off, into_reg, ty)
    }

    // Store to a spillslot.
    fn store_spillslot(&self, slot: SpillSlot, ty: Type, from_reg: Reg) -> Vec<Inst> {
        // Offset from beginning of spillslot area, which is FP - stackslot_size - 8*spillslots.
        let slot = slot.get() as i64;
        let fp_off: i64 =
            -(self.stackslots_size as i64) - (8 * self.spillslots.unwrap() as i64) + 8 * slot;
        store_stack(fp_off, from_reg, ty)
    }

    fn gen_prologue(&self) -> Vec<Inst> {
        let mut insts = vec![];
        let total_stacksize = self.stackslots_size + 8 * self.spillslots.unwrap();
        let total_stacksize = (total_stacksize + 15) & !15; // 16-align the stack.

        // stp fp (x29), lr (x30), [sp, #-16]!
        insts.push(Inst::StoreP64 {
            rt: fp_reg(),
            rt2: link_reg(),
            mem: PairMemArg::PreIndexed(stack_reg(), SImm7::maybe_from_i64(-16).unwrap()),
        });
        // mov fp (x29), sp
        insts.push(Inst::Mov {
            rd: fp_reg(),
            rm: stack_reg(),
        });

        // sub sp, #total_stacksize
        if let Some(imm12) = Imm12::maybe_from_u64(total_stacksize as u64) {
            let sub_inst = Inst::AluRRImm12 {
                alu_op: ALUOp::Sub64,
                rd: stack_reg(),
                rn: stack_reg(),
                imm12,
            };
            insts.push(sub_inst);
        } else {
            let const_data = u64_constant(total_stacksize as u64);
            let tmp = spilltmp_reg();
            let const_inst = Inst::ULoad64 {
                rd: tmp,
                mem: MemArg::label(MemLabel::ConstantData(const_data)),
            };
            let sub_inst = Inst::AluRRR {
                alu_op: ALUOp::Sub64,
                rd: stack_reg(),
                rn: stack_reg(),
                rm: tmp,
            };
            insts.push(const_inst);
            insts.push(sub_inst);
        }

        // TODO: save clobbered registers (from regalloc). Save these below spillslots, right above SP,
        // with "push" (STP) instructions.
        insts
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        let mut insts = vec![];

        // TODO: restore clobbered registers
        // The MOV (alias of ORR) interprets x31 as XZR, so use an ADD here.
        // MOV to SP is an alias of ADD.
        insts.push(Inst::AluRRImm12 {
            alu_op: ALUOp::Add64,
            rd: stack_reg(),
            rn: fp_reg(),
            imm12: Imm12 {
                bits: 0,
                shift12: false,
            },
        });
        insts.push(Inst::LoadP64 {
            rt: fp_reg(),
            rt2: link_reg(),
            mem: PairMemArg::PostIndexed(stack_reg(), SImm7::maybe_from_i64(16).unwrap()),
        });
        insts.push(Inst::Ret {});
        insts
    }
}
