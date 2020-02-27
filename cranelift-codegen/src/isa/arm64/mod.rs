//! ARM 64-bit Instruction Set Architecture.

#![allow(unused_imports)]

use crate::binemit::{CodeSink, MemoryCodeSink, RelocSink, StackmapSink, TrapSink};
use crate::ir::Function;
use crate::machinst::{compile, MachBackend, MachCompileResult, ShowWithRRU, VCode};
use crate::machinst::{ABIBody, ABICall};
use crate::result::CodegenResult;
use crate::settings;

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::str::FromStr;
use std::string::String;

use regalloc::RealRegUniverse;
use target_lexicon::Triple;

// New backend:
mod abi;
mod inst;
mod lower;

use inst::create_reg_universe;

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

    fn compile_vcode(&self, mut func: Function) -> VCode<inst::Inst> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let abi = Box::new(abi::ARM64ABIBody::new(&func));
        compile::compile::<Arm64Backend>(&mut func, self, abi)
    }
}

impl MachBackend for Arm64Backend {
    fn compile_function(
        &self,
        func: Function,
        want_disasm: bool,
    ) -> CodegenResult<MachCompileResult> {
        let vcode = self.compile_vcode(func);
        let sections = vcode.emit();
        let frame_size = vcode.frame_size();

        let disasm = if want_disasm {
            Some(vcode.show_rru(Some(&create_reg_universe())))
        } else {
            None
        };

        Ok(MachCompileResult {
            sections,
            frame_size,
            disasm,
        })
    }

    fn name(&self) -> &'static str {
        "arm64"
    }

    fn triple(&self) -> Triple {
        FromStr::from_str("arm64").unwrap()
    }

    fn flags(&self) -> &settings::Flags {
        &self.flags
    }

    fn reg_universe(&self) -> RealRegUniverse {
        create_reg_universe()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::binemit::{NullRelocSink, NullStackmapSink, NullTrapSink};
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::types::*;
    use crate::ir::{AbiParam, ExternalName, Function, InstBuilder, Signature};
    use crate::isa::CallConv;

    #[test]
    fn test_compile_function() {
        let name = ExternalName::testcase("test0");
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(I32));
        sig.returns.push(AbiParam::new(I32));
        let mut func = Function::with_name_signature(name, sig);

        let bb0 = func.dfg.make_block();
        let arg0 = func.dfg.append_block_param(bb0, I32);

        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(bb0);
        let v0 = pos.ins().iconst(I32, 0x12345678);
        let v1 = pos.ins().iadd(arg0, v0);
        pos.ins().return_(&[v1]);

        let backend = Arm64Backend::new();
        let sections = backend.compile_function(func, false).unwrap().sections;
        let code = &sections.sections[0].data;

        // stp x29, x30, [sp, #-16]!
        // mov x29, sp
        // ldr x1, 0x20
        // add w0, w0, w1
        // mov sp, x29
        // ldp x29, x30, [sp], #16
        // ret
        let golden = vec![
            0xfd, 0x7b, 0xbf, 0xa9, 0xfd, 0x03, 0x00, 0x91, 0xc1, 0x00, 0x00, 0x58, 0x00, 0x00,
            0x01, 0x0b, 0xbf, 0x03, 0x00, 0x91, 0xfd, 0x7b, 0xc1, 0xa8, 0xc0, 0x03, 0x5f, 0xd6,
        ];

        assert_eq!(code, &golden);
    }
}
