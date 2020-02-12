//! Implementation of the standard ARM64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::StackSlot;
use crate::ir::Type;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, WritableReg};

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
    stackslots: Vec<usize>,               // offsets to each stackslot
    stackslots_size: usize,               // total stack size of all stackslots
    clobbered: Set<WritableReg<RealReg>>, // clobbered registers, from regalloc.
    spillslots: Option<usize>,            // total number of spillslots, from regalloc.
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
        //println!("ARM64 ABI: func signature {:?}", f.signature);

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
fn get_stack_addr(fp_offset: i64) -> MemArg {
    MemArg::StackOffset(fp_offset)
}

fn load_stack(fp_offset: i64, into_reg: WritableReg<Reg>, ty: Type) -> Inst {
    let mem = get_stack_addr(fp_offset);

    match ty {
        types::B1 | types::B8 | types::I8 => Inst::ULoad8 { rd: into_reg, mem },
        types::B16 | types::I16 => Inst::ULoad16 { rd: into_reg, mem },
        types::B32 | types::I32 => Inst::ULoad32 { rd: into_reg, mem },
        types::B64 | types::I64 => Inst::ULoad64 { rd: into_reg, mem },
        _ => unimplemented!(),
    }
}

fn store_stack(fp_offset: i64, from_reg: Reg, ty: Type) -> Inst {
    let mem = get_stack_addr(fp_offset);

    match ty {
        types::B1 | types::B8 | types::I8 => Inst::Store8 { rd: from_reg, mem },
        types::B16 | types::I16 => Inst::Store16 { rd: from_reg, mem },
        types::B32 | types::I32 => Inst::Store32 { rd: from_reg, mem },
        types::B64 | types::I64 => Inst::Store64 { rd: from_reg, mem },
        _ => unimplemented!(),
    }
}

fn is_callee_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => {
            // x19 - x28 inclusive are callee-saves.
            r.get_hw_encoding() >= 19 && r.get_hw_encoding() <= 28
        }
        RegClass::V128 => {
            // v8 - v15 inclusive are callee-saves.
            r.get_hw_encoding() >= 8 && r.get_hw_encoding() <= 15
        }
        _ => panic!("Unexpected RegClass"),
    }
}

fn is_caller_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => {
            // x0 - x17 inclusive are caller-saves.
            r.get_hw_encoding() <= 17
        }
        RegClass::V128 => {
            // v0 - v7 inclusive and v16 - v31 inclusive are caller-saves.
            r.get_hw_encoding() <= 7 || (r.get_hw_encoding() >= 16 && r.get_hw_encoding() <= 31)
        }
        _ => panic!("Unexpected RegClass"),
    }
}

fn get_callee_saves(regs: Vec<WritableReg<RealReg>>) -> Vec<WritableReg<RealReg>> {
    regs.into_iter()
        .filter(|r| is_callee_save(r.to_reg()))
        .collect()
}

fn get_caller_saves_set() -> Set<RealReg> {
    let mut set = Set::empty();
    for i in 0..32 {
        let x = xreg(i).to_real_reg();
        if is_caller_save(x) {
            set.insert(x);
        }
    }
    for i in 0..32 {
        let v = vreg(i).to_real_reg();
        if is_caller_save(v) {
            set.insert(v);
        }
    }
    set
}

impl ABIBody<Inst> for ARM64ABIBody {
    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for arg in &self.args {
            if let &ABIArg::Reg(r) = arg {
                set.insert(r);
            }
        }
        //println!("ARM64 ABI: liveins {:?}", set);
        set
    }

    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for ret in &self.rets {
            if let &ABIRet::Reg(r) = ret {
                set.insert(r);
            }
        }
        //println!("ARM64 ABI: liveouts {:?}", set);
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

    fn load_arg(&self, idx: usize, into_reg: WritableReg<Reg>) -> Inst {
        match &self.args[idx] {
            &ABIArg::Reg(r) => {
                return Inst::gen_move(into_reg, r.to_reg());
            }
            _ => unimplemented!(),
        }
    }

    fn store_retval(&self, idx: usize, from_reg: Reg) -> Inst {
        match &self.rets[idx] {
            &ABIRet::Reg(r) => {
                return Inst::gen_move(WritableReg::from_reg(r.to_reg()), from_reg);
            }
            _ => unimplemented!(),
        }
    }

    fn set_num_spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
    }

    fn set_clobbered(&mut self, clobbered: Set<WritableReg<RealReg>>) {
        self.clobbered = clobbered;
    }

    fn load_stackslot(
        &self,
        slot: StackSlot,
        offset: usize,
        ty: Type,
        into_reg: WritableReg<Reg>,
    ) -> Inst {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        load_stack(fp_off, into_reg, ty)
    }

    fn store_stackslot(&self, slot: StackSlot, offset: usize, ty: Type, from_reg: Reg) -> Inst {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        store_stack(fp_off, from_reg, ty)
    }

    // Load from a spillslot.
    fn load_spillslot(&self, slot: SpillSlot, ty: Type, into_reg: WritableReg<Reg>) -> Inst {
        // Note that when spills/fills are generated, we don't yet know how many
        // spillslots there will be, so we allocate *downward* from the beginning
        // of the stackslot area. Hence: FP - stackslot_size - 8*spillslot -
        // sizeof(ty).
        let slot = slot.get() as i64;
        let ty_size = self.get_spillslot_size(into_reg.to_reg().get_class(), ty) * 8;
        let fp_off: i64 = -(self.stackslots_size as i64) - (8 * slot) - ty_size as i64;
        load_stack(fp_off, into_reg, ty)
    }

    // Store to a spillslot.
    fn store_spillslot(&self, slot: SpillSlot, ty: Type, from_reg: Reg) -> Inst {
        let slot = slot.get() as i64;
        let ty_size = self.get_spillslot_size(from_reg.get_class(), ty) * 8;
        let fp_off: i64 = -(self.stackslots_size as i64) - (8 * slot) - ty_size as i64;
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
            mem: PairMemArg::PreIndexed(
                writable_stack_reg(),
                SImm7::maybe_from_i64(-16 / 8).unwrap(),
            ),
        });
        // mov fp (x29), sp. This uses the ADDI rd, rs, 0 form of `MOV` because
        // the usual encoding (`ORR`) does not work with SP.
        insts.push(Inst::AluRRImm12 {
            alu_op: ALUOp::Add64,
            rd: writable_fp_reg(),
            rn: stack_reg(),
            imm12: Imm12 {
                bits: 0,
                shift12: false,
            },
        });

        if total_stacksize > 0 {
            // sub sp, sp, #total_stacksize
            if let Some(imm12) = Imm12::maybe_from_u64(total_stacksize as u64) {
                let sub_inst = Inst::AluRRImm12 {
                    alu_op: ALUOp::Sub64,
                    rd: writable_stack_reg(),
                    rn: stack_reg(),
                    imm12,
                };
                insts.push(sub_inst);
            } else {
                let const_data = u64_constant(total_stacksize as u64);
                let tmp = writable_spilltmp_reg();
                let const_inst = Inst::ULoad64 {
                    rd: tmp,
                    mem: MemArg::label(MemLabel::ConstantData(const_data)),
                };
                let sub_inst = Inst::AluRRR {
                    alu_op: ALUOp::Sub64,
                    rd: writable_stack_reg(),
                    rn: stack_reg(),
                    rm: tmp.to_reg(),
                };
                insts.push(const_inst);
                insts.push(sub_inst);
            }
        }

        // Save clobbered registers.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg_pair in clobbered.chunks(2) {
            let (r1, r2) = if reg_pair.len() == 2 {
                // .to_reg().to_reg(): WritableReg<RealReg> --> RealReg --> Reg
                (reg_pair[0].to_reg().to_reg(), reg_pair[1].to_reg().to_reg())
            } else {
                (reg_pair[0].to_reg().to_reg(), zero_reg())
            };
            // stp r1, r2, [sp, #-16]!
            insts.push(Inst::StoreP64 {
                rt: r1,
                rt2: r2,
                mem: PairMemArg::PreIndexed(
                    writable_stack_reg(),
                    SImm7::maybe_from_i64(-16 / 8).unwrap(),
                ),
            });
        }

        insts
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        let mut insts = vec![];

        // Restore clobbered registers.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg_pair in clobbered.chunks(2).rev() {
            let (r1, r2) = if reg_pair.len() == 2 {
                (
                    reg_pair[0].map(|r| r.to_reg()),
                    reg_pair[1].map(|r| r.to_reg()),
                )
            } else {
                (reg_pair[0].map(|r| r.to_reg()), writable_zero_reg())
            };
            // ldp r1, r2, [sp], #16
            insts.push(Inst::LoadP64 {
                rt: r1,
                rt2: r2,
                mem: PairMemArg::PostIndexed(
                    writable_stack_reg(),
                    SImm7::maybe_from_i64(16 / 8).unwrap(),
                ),
            });
        }

        // The MOV (alias of ORR) interprets x31 as XZR, so use an ADD here.
        // MOV to SP is an alias of ADD.
        insts.push(Inst::AluRRImm12 {
            alu_op: ALUOp::Add64,
            rd: writable_stack_reg(),
            rn: fp_reg(),
            imm12: Imm12 {
                bits: 0,
                shift12: false,
            },
        });
        insts.push(Inst::LoadP64 {
            rt: writable_fp_reg(),
            rt2: writable_link_reg(),
            mem: PairMemArg::PostIndexed(
                writable_stack_reg(),
                SImm7::maybe_from_i64(16 / 8).unwrap(),
            ),
        });
        insts.push(Inst::Ret {});
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

    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, ty: Type) -> Inst {
        self.store_spillslot(to_slot, ty, from_reg.to_reg())
    }

    fn gen_reload(&self, to_reg: WritableReg<RealReg>, from_slot: SpillSlot, ty: Type) -> Inst {
        self.load_spillslot(from_slot, ty, to_reg.map(|r| r.to_reg()))
    }
}
