//! ARM 64-bit Instruction Set Architecture.

use crate::binemit::{CodeSink, MemoryCodeSink, NullRelocSink, NullStackmapSink, NullTrapSink};
use crate::ir::Function;
use crate::machinst::compile;
use crate::machinst::MachBackend;
use crate::result::CodegenResult;

use alloc::vec::Vec;

// New backend:
mod inst;
mod lower;

/// An ARM64 backend.
pub struct Arm64Backend {}

impl Arm64Backend {
    /// Create a new ARM64 backend.
    pub fn new() -> Arm64Backend {
        Arm64Backend {}
    }
}

impl MachBackend for Arm64Backend {
    fn compile_function_to_memory(&mut self, mut func: Function) -> CodegenResult<Vec<u8>> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let mut vcode = compile::compile::<Arm64Backend>(&mut func, self);

        let mut buf: Vec<u8> = vec![];
        buf.resize(vcode.code_size(), 0);

        let mut relocs = NullRelocSink {};
        let mut traps = NullTrapSink {};
        let mut stackmaps = NullStackmapSink {};

        let mut sink = unsafe {
            MemoryCodeSink::new(buf.as_mut_ptr(), &mut relocs, &mut traps, &mut stackmaps)
        };
        vcode.emit(&mut sink);

        Ok(buf)
    }
}
