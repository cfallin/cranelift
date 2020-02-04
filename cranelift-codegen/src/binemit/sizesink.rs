//! Code sink that accumulates size of code only.

use super::{Addend, CodeOffset, CodeSink, Reloc};
use crate::ir::entities::Value;
use crate::ir::{ConstantOffset, ExternalName, Function, JumpTable, SourceLoc, TrapCode};
use crate::isa::TargetIsa;

/// A CodeSink implementation that only computes the size of the generated
/// code, without actually saving it in memory.
pub struct SizeCodeSink {
    offset: usize,
}

impl SizeCodeSink {
    /// Create a new SizeCodeSink.
    pub fn new() -> SizeCodeSink {
        SizeCodeSink { offset: 0 }
    }

    /// Return the size of emitted code so far.
    pub fn size(&self) -> usize {
        self.offset
    }
}

impl CodeSink for SizeCodeSink {
    fn offset(&self) -> CodeOffset {
        self.offset as CodeOffset
    }

    fn put1(&mut self, _x: u8) {
        self.offset += 1;
    }

    fn put2(&mut self, _x: u16) {
        self.offset += 2;
    }

    fn put4(&mut self, _x: u32) {
        self.offset += 4;
    }

    fn put8(&mut self, _x: u64) {
        self.offset += 8;
    }

    fn reloc_ebb(&mut self, _rel: Reloc, _ebb_offset: CodeOffset) {}

    fn reloc_external(&mut self, _rel: Reloc, _name: &ExternalName, _addend: Addend) {}

    fn reloc_constant(&mut self, _rel: Reloc, _constant_offset: ConstantOffset) {}

    fn reloc_jt(&mut self, _rel: Reloc, _jt: JumpTable) {}

    fn trap(&mut self, _code: TrapCode, _srcloc: SourceLoc) {}

    fn begin_jumptables(&mut self) {}

    fn begin_rodata(&mut self) {}

    fn end_codegen(&mut self) {}

    fn add_stackmap(&mut self, _val_list: &[Value], _func: &Function, _isa: &dyn TargetIsa) {}
}
