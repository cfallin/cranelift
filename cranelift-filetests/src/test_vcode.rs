use crate::subtest::{run_filecheck, Context, SubTest, SubtestResult};
use cranelift_codegen::ir::Function;
use cranelift_codegen::isa::lookup_mach_backend;
use cranelift_reader::{TestCommand, TestOption};
use target_lexicon::Triple;

use log::info;
use std::borrow::Cow;
use std::str::FromStr;
use std::string::String;

struct TestVCode {
    arch: String,
}

pub fn subtest(parsed: &TestCommand) -> SubtestResult<Box<dyn SubTest>> {
    assert_eq!(parsed.command, "vcode");

    let mut arch = "arm64".to_string();
    for option in &parsed.options {
        match option {
            TestOption::Value(k, v) if k == &"arch" => {
                arch = v.to_string();
            }
            _ => {}
        }
    }

    Ok(Box::new(TestVCode { arch }))
}

impl SubTest for TestVCode {
    fn name(&self) -> &'static str {
        "vcode"
    }

    fn is_mutating(&self) -> bool {
        true
    }

    fn needs_isa(&self) -> bool {
        false
    }

    fn run(&self, func: Cow<Function>, context: &Context) -> SubtestResult<()> {
        let func = func.into_owned();

        let triple =
            Triple::from_str(&self.arch).map_err(|_| format!("Unknown arch: '{}'", self.arch))?;

        let backend = lookup_mach_backend(triple)
            .map_err(|_| format!("Could not look up backend for arch '{}'", self.arch))?;

        let text = backend
            .compile_function(func, /* want_disasm = */ true)
            .map_err(|e| format!("Error from backend compilation: {:?}", e))?
            .disasm
            .unwrap();

        info!("text input to filecheck is:\n{}\n", text);

        run_filecheck(&text, context)
    }
}
