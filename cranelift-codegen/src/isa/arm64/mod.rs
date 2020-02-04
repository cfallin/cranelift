//! ARM 64-bit Instruction Set Architecture.

#![allow(unused_imports)]

use crate::binemit::{CodeSink, MemoryCodeSink, RelocSink, StackmapSink, TrapSink};
use crate::ir::Function;
use crate::machinst::compile;
use crate::machinst::MachBackend;
use crate::result::CodegenResult;
use crate::settings;

use alloc::vec::Vec;

// New backend:
mod inst;
mod lower;

/// An ARM64 backend.
pub struct Arm64Backend {
    flags: settings::Flags,
}

impl Arm64Backend {
    /// Create a new ARM64 backend.
    pub fn new() -> Arm64Backend {
        Arm64Backend {
            flags: settings::Flags::new(settings::builder()),
        }
    }
}

impl MachBackend for Arm64Backend {
    fn compile_function_to_memory(
        &self,
        mut func: Function,
        relocs: &mut dyn RelocSink,
        traps: &mut dyn TrapSink,
        stackmaps: &mut dyn StackmapSink,
    ) -> CodegenResult<Vec<u8>> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let /*mut*/ vcode = compile::compile::<Arm64Backend>(&mut func, self);

        let mut buf: Vec<u8> = vec![];
        buf.resize(vcode.code_size(), 0);

        let mut sink = unsafe { MemoryCodeSink::new(buf.as_mut_ptr(), relocs, traps, stackmaps) };
        vcode.emit(&mut sink);

        Ok(buf)
    }

    fn flags(&self) -> &settings::Flags {
        &self.flags
    }
}
