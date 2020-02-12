//! X86_64-bit Instruction Set Architecture.

#![allow(unused_imports)]

use crate::binemit::{CodeSink, MemoryCodeSink, RelocSink, StackmapSink, TrapSink};
use crate::ir::Function;
use crate::machinst::compile;
use crate::machinst::MachBackend;
use crate::result::CodegenResult;
use crate::settings;

use alloc::boxed::Box;
use alloc::vec::Vec;

use regalloc::RealRegUniverse;

// New backend:
mod abi;
mod inst;
mod lower;

use inst::create_reg_universe;

/// An X64 backend.
pub struct X64Backend {
    flags: settings::Flags,
}

impl X64Backend {
    /// Create a new X64 backend.
    pub fn new() -> X64Backend {
        X64Backend {
            flags: settings::Flags::new(settings::builder()),
        }
    }
}

impl MachBackend for X64Backend {
    fn compile_function_to_memory(
        &self,
        mut func: Function,
        relocs: &mut dyn RelocSink,
        traps: &mut dyn TrapSink,
        stackmaps: &mut dyn StackmapSink,
    ) -> CodegenResult<Vec<u8>> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let abi = Box::new(abi::X64ABIBody::new(&func));
        let vcode = compile::compile::<X64Backend>(&mut func, self, abi);

        let mut buf: Vec<u8> = vec![0; vcode.emitted_size()];

        let mut sink = unsafe { MemoryCodeSink::new(buf.as_mut_ptr(), relocs, traps, stackmaps) };
        vcode.emit(&mut sink);

        Ok(buf)
    }

    fn flags(&self) -> &settings::Flags {
        &self.flags
    }

    fn reg_universe(&self) -> RealRegUniverse {
        create_reg_universe()
    }
}
