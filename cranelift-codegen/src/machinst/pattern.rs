//! This module allows a backend to construct a lookup table used by the instruction lowering
//! process.

use crate::ir::{Function, Inst, InstructionData, Opcode, Type, Value, ValueDef};
use crate::isa::registers::{RegClass, RegUnit};
use crate::machinst::lower::*;
use crate::machinst::*;
use crate::num_uses::NumUses;

use crate::HashMap;

use alloc::boxed::Box;
use smallvec::SmallVec;
use std::hash::Hash;

/// A pattern: when we find an IR instruction with matching opcode, type, and arguments, we can
/// call the associated codegen function.
#[derive(Clone)]
pub struct LowerPattern<Op: MachInstOp, Arg: MachInstArg> {
    op: Opcode,
    ty: Type,
    args: Vec<LowerPatternArg<Op, Arg>>,
    action: LowerAction<Op, Arg>,
}

/// An argument to a pattern. May match a specific register, a register class, or a nested pattern
/// (only if the nested pattern produces one value and we are its one use).
#[derive(Clone)]
pub enum LowerPatternArg<Op: MachInstOp, Arg: MachInstArg> {
    /// A register.
    Reg,
    /// A nested pattern. This matches an instruction producing one value, with one use (this
    /// argument), and with no side-effects.
    Nested(Box<LowerPattern<Op, Arg>>),
}

/// An action to perform when matching a tree of ops. Returns `true` if codegen was successful.
/// Otherwise, another pattern/action should be used instead.
pub type LowerAction<Op: MachInstOp, Arg: MachInstArg> = for<'a> fn(
    ctx: &mut MachInstLowerCtx<'a, Op, Arg>,
    insts: &[&'a InstructionData],
    regs: &[MachReg],
    results: &[MachReg],
) -> bool;

/// A table of lowering actions that can be queried by opcode/ty.
#[derive(Clone)]
pub struct LowerTable<Op: MachInstOp, Arg: MachInstArg> {
    actions: HashMap<(Opcode, Type), SmallVec<[LowerTableEntry<Op, Arg>; 4]>>,
}

#[derive(Clone)]
struct LowerTableEntry<Op: MachInstOp, Arg: MachInstArg> {
    args: Vec<LowerTableEntryArg<Op, Arg>>,
    action: LowerAction<Op, Arg>,
}

#[derive(Clone)]
enum LowerTableEntryArg<Op: MachInstOp, Arg: MachInstArg> {
    Reg,
    Nested(LowerTable<Op, Arg>),
}

/// A lookup result from a lowering table.
pub struct LowerTableLookup<'a, Op: MachInstOp, Arg: MachInstArg> {
    /// The action at the root.
    action: Option<LowerAction<Op, Arg>>,
    /// Instructions, read in pre-order through the matching tree.
    insts: SmallVec<[&'a InstructionData; 4]>,
    /// Inputs, read in-order across the tree leaves.
    input_values: SmallVec<[Value; 4]>,
    /// Results of the root instruction.
    result_values: SmallVec<[Value; 4]>,
}

impl<Op: MachInstOp, Arg: MachInstArg> LowerTable<Op, Arg> {
    /// New lowering table.
    pub fn new() -> LowerTable<Op, Arg> {
        LowerTable {
            actions: HashMap::new(),
        }
    }

    /// Add a pattern to the table.
    pub fn add(&mut self, pat: LowerPattern<Op, Arg>) {
        let key = (pat.op.clone(), pat.ty.clone());
        let args = pat
            .args
            .iter()
            .map(|arg| match arg {
                &LowerPatternArg::Reg => LowerTableEntryArg::Reg,
                &LowerPatternArg::Nested(ref nested) => {
                    let mut subtable = LowerTable::new();
                    subtable.add((&**nested).clone());
                    LowerTableEntryArg::Nested(subtable)
                }
            })
            .collect();
        let entry = LowerTableEntry {
            args,
            action: pat.action,
        };
        self.actions
            .entry(key)
            .or_insert_with(SmallVec::new)
            .push(entry);
    }

    /// Perform a lookup in the (nested) lookup tables, starting from a root instruction, returning
    /// captured match info (bound registers and instruction data) if successful.
    pub fn lookup<'a>(
        &self,
        func: &'a Function,
        root: Inst,
        num_uses: &NumUses,
    ) -> Option<LowerTableLookup<'a, Op, Arg>> {
        let mut result = LowerTableLookup {
            action: None,
            insts: SmallVec::new(),
            input_values: SmallVec::new(),
            result_values: SmallVec::new(),
        };
        if self.matches(func, root, num_uses, &mut result) {
            for r in func.dfg.inst_results(root) {
                let v = func.dfg.resolve_aliases(*r);
                result.result_values.push(v);
            }
            Some(result)
        } else {
            None
        }
    }

    fn matches<'a>(
        &self,
        func: &'a Function,
        root: Inst,
        num_uses: &NumUses,
        result: &mut LowerTableLookup<'a, Op, Arg>,
    ) -> bool {
        let key = (func.dfg[root].opcode(), func.dfg.ctrl_typevar(root));
        result.insts.push(&func.dfg[root]);
        if let Some(entries) = self.actions.get(&key) {
            for entry in entries {
                if entry.matches(func, root, num_uses, result) {
                    result.action = Some(entry.action);
                    return true;
                }
            }
        }
        result.insts.pop();
        false
    }
}

impl<Op: MachInstOp, Arg: MachInstArg> LowerTableEntry<Op, Arg> {
    fn arg_matches<'a>(
        &self,
        func: &'a Function,
        root: Inst,
        num_uses: &NumUses,
        arg: Value,
        matcharg: &LowerTableEntryArg<Op, Arg>,
        result: &mut LowerTableLookup<'a, Op, Arg>,
    ) -> bool {
        let def = func.dfg.value_def(arg);
        match matcharg {
            &LowerTableEntryArg::Reg => true,
            &LowerTableEntryArg::Nested(ref subtable) => {
                // Determine whether (i) the producing instruction is in the same EBB, (ii) the
                // producing instruction produces only one value, and (iii) we are the only use
                // of the producing instruction.
                if let ValueDef::Result(def_inst, _) = def {
                    if func.layout.inst_ebb(def_inst) == func.layout.inst_ebb(root)
                        && num_uses.use_count(def_inst) == 1
                        && func.dfg.inst_results(def_inst).len() == 1
                    {
                        // Attempt a match on the nested table.
                        subtable.matches(func, def_inst, num_uses, result)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    fn matches<'a>(
        &self,
        func: &'a Function,
        root: Inst,
        num_uses: &NumUses,
        result: &mut LowerTableLookup<'a, Op, Arg>,
    ) -> bool {
        let input_len = result.input_values.len();
        for (arg, matcharg) in func.dfg.inst_args(root).iter().zip(self.args.iter()) {
            let arg = func.dfg.resolve_aliases(*arg);
            result.input_values.push(arg);
            if !self.arg_matches(func, root, num_uses, arg, matcharg, result) {
                result.input_values.truncate(input_len);
                return false;
            }
        }
        true
    }
}
