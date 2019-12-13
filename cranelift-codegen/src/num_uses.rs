//! A pass that computes the number of uses of any given instruction.

use crate::cursor::{Cursor, FuncCursor};
use crate::dce::has_side_effect;
use crate::entity::SecondaryMap;
use crate::ir::dfg::ValueDef;
use crate::ir::instructions::InstructionData;
use crate::ir::Value;
use crate::ir::{DataFlowGraph, Function, Inst, Opcode};

/// Auxiliary data structure that counts the number of uses of any given instruction in a Function.
///
/// Uses of an instruction include other instructions that use its result, and an implicit use (by
/// "the universe") if the instruction has a side-effect, such as a memory write or a possible
/// trap.
pub struct NumUses {
    uses: SecondaryMap<Inst, u32>,
}

impl NumUses {
    fn new() -> NumUses {
        NumUses {
            uses: SecondaryMap::with_default(0),
        }
    }

    pub fn compute(func: &Function) -> NumUses {
        let mut uses = NumUses::new();
        for ebb in func.layout.ebbs() {
            for inst in func.layout.ebb_insts(ebb) {
                // A side-effecting instruction has an implicit use.
                if has_side_effect(func, inst) {
                    uses.add_inst(inst);
                }
                for arg in func.dfg.inst_args(inst) {
                    let v = func.dfg.resolve_aliases(*arg);
                    uses.add_value(&func.dfg, v);
                }
            }
        }
        uses
    }

    fn add_inst(&mut self, inst: Inst) {
        self.uses[inst] += 1;
    }

    fn add_value(&mut self, dfg: &DataFlowGraph, v: Value) {
        match dfg.value_def(v) {
            ValueDef::Result(inst, _) => {
                self.uses[inst] += 1;
            }
            _ => {}
        }
    }

    fn use_count(&self, i: Inst) -> usize {
        self.uses[i] as usize
    }

    fn is_used(&self, i: Inst) -> bool {
        self.use_count(i) > 0
    }
}
