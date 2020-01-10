//! Split the outgoing edges of conditional branches that pass parameters.
//!
//! One of the reason for splitting edges is to be able to insert `copy` and `regmove` instructions
//! between a conditional branch and the following terminator.
//!
//! This `branch_splitting` module is a copy/fork of the one in `crate::regalloc`; it has been
//! adjusted for the new VCode / MachInst backend framework.
#![cfg(feature = "basic-blocks")]

use alloc::vec::Vec;

use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{Ebb, Function, Inst, InstBuilder, InstructionData, Opcode, ValueList};
use crate::HashMap;
use smallvec::SmallVec;

/// Split critical edges in the given function.
pub fn split_critical_edges(func: &mut Function) {
    let preds = count_predecessors(func);
    let mut ctx = Context {
        cur: FuncCursor::new(func),
        preds,
    };
    ctx.run()
}

struct Context<'a> {
    /// Current instruction as well as reference to function.
    cur: FuncCursor<'a>,
    /// Predecessor count per block.
    preds: HashMap<Ebb, u32>,
}

impl<'a> Context<'a> {
    fn run(&mut self) {
        // Any ebb order will do.
        let ebbs: SmallVec<[Ebb; 32]> = self.cur.func.layout.ebbs().into_iter().collect();
        for ebb in ebbs.into_iter() {
            // Branches can only be at the last or second to last position in an extended basic
            // block.
            self.cur.goto_last_inst(ebb);
            let terminator_inst = self.cur.current_inst().expect("terminator");
            if let Some(inst) = self.cur.prev_inst() {
                let opcode = self.cur.func.dfg[inst].opcode();
                if opcode.is_branch() {
                    self.visit_conditional_branch(inst, opcode);
                    self.cur.goto_inst(terminator_inst);
                    self.visit_terminator_branch(terminator_inst);
                }
            }
        }
    }

    fn visit_conditional_branch(&mut self, branch: Inst, opcode: Opcode) {
        // TODO: target = dfg[branch].branch_destination().expect("conditional branch");
        let target = match self.cur.func.dfg[branch] {
            InstructionData::Branch { destination, .. }
            | InstructionData::BranchIcmp { destination, .. }
            | InstructionData::BranchInt { destination, .. }
            | InstructionData::BranchFloat { destination, .. } => destination,
            _ => panic!("Unexpected instruction in visit_conditional_branch"),
        };

        // If there are any parameters, split the edge.
        if self.should_split_edge(target) {
            // Create the block the branch will jump to.
            let new_ebb = self.cur.func.dfg.make_ebb();

            // Insert the new block before the destination, such that it can fallthrough in the
            // target block.
            assert_ne!(Some(target), self.cur.layout().entry_block());
            self.cur.layout_mut().insert_ebb(new_ebb, target);

            // Extract the arguments of the branch instruction, split the Ebb parameters and the
            // branch arguments
            let num_fixed = opcode.constraints().num_fixed_value_arguments();
            let dfg = &mut self.cur.func.dfg;
            let old_args: Vec<_> = {
                let args = dfg[branch].take_value_list().expect("ebb parameters");
                args.as_slice(&dfg.value_lists).iter().copied().collect()
            };
            let (branch_args, ebb_params) = old_args.split_at(num_fixed);

            // Replace the branch destination by the new Ebb created with no parameters, and restore
            // the branch arguments, without the original Ebb parameters.
            {
                let branch_args = ValueList::from_slice(branch_args, &mut dfg.value_lists);
                let data = &mut dfg[branch];
                *data.branch_destination_mut().expect("branch") = new_ebb;
                data.put_value_list(branch_args);
            }

            // Insert a jump to the original target with its arguments into the new block.
            self.cur.goto_first_insertion_point(new_ebb);
            self.cur.ins().jump(target, ebb_params);

            // Reset the cursor to point to the branch.
            self.cur.goto_inst(branch);
        }
    }

    fn visit_terminator_branch(&mut self, inst: Inst) {
        let inst_data = &self.cur.func.dfg[inst];
        let opcode = inst_data.opcode();
        if opcode != Opcode::Jump && opcode != Opcode::Fallthrough {
            // This opcode is ignored as it does not have any EBB parameters.
            if opcode != Opcode::IndirectJumpTableBr {
                debug_assert!(!opcode.is_branch())
            }
            return;
        }

        let target = match inst_data {
            InstructionData::Jump { destination, .. } => destination,
            _ => panic!(
                "Unexpected instruction {:?} in visit_terminator_branch",
                self.cur.func.dfg[inst]
            ),
        };
        debug_assert!(self.cur.func.dfg[inst].opcode().is_terminator());

        // If there are any parameters, split the edge.
        if self.should_split_edge(*target) {
            // Create the block the branch will jump to.
            let new_ebb = self.cur.func.dfg.make_ebb();

            // Split the current block before its terminator, and insert a new jump instruction to
            // jump to it.
            let jump = self.cur.ins().jump(new_ebb, &[]);
            self.cur.insert_ebb(new_ebb);

            // Reset the cursor to point to new terminator of the old ebb.
            self.cur.goto_inst(jump);
        }
    }

    /// Returns whether we should introduce a new branch.
    fn should_split_edge(&self, target: Ebb) -> bool {
        // We should split the edge if the target has any parameters or if the
        // target has more than one block reaching it.
        !self.cur.func.dfg.ebb_params(target).is_empty()
            || self.preds.get(&target).cloned().unwrap_or(0) > 1
    }
}

fn count_predecessors(f: &Function) -> HashMap<Ebb, u32> {
    let mut preds = HashMap::new();

    // Implicit entry edge from virtual "start" block to entry block.
    if let Some(entry) = f.layout.entry_block() {
        preds.insert(entry, 1);
    }

    for ebb in f.layout.ebbs() {
        for inst in f.layout.ebb_insts(ebb) {
            // If this is a branch, inc the pred count.
            if let Some(dest) = f.dfg[inst].branch_destination() {
                *preds.entry(dest).or_insert(0) += 1;
            }
        }
    }
    preds
}
