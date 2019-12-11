//! This module exposes a set of tools, such as a pattern-matcher and a simple builder API for
//! those patterns, that allow an ISA definition to define legalization and code-generation rules.

use crate::ir::{Opcode, Type};
use alloc::vec::Vec;
use alloc::boxed::Box;

mod pattern;
pub use pattern::*;

/// ISA definition. Can be used to code-generate a backend.
pub struct IsaDef {
    reg_banks: Vec<IsaRegBank>,
    reg_classes: Vec<IsaRegClass>,
    legalize_rules: Vec<IsaRuleLegalize>,
    emit_rules: Vec<IsaRuleEmit>,
}

/// A register bank definition.
struct IsaRegBank {
    name: &'static str,
    regs: usize,
    start_reg: usize,
    track_pressure: bool,
    name_prefix: Option<&'static str>,
    names: Option<&'static [&'static str]>,
}

/// A builder for a register bank definition.
pub struct IsaRegBankBuilder<'a> {
    bank: &'a mut IsaRegBank,
    index: IsaRegBankIndex,
}

/// A reference to a register bank.
pub type IsaRegBankIndex = usize;

/// A register class definition.
struct IsaRegClass {
    name: &'static str,
    bank: IsaRegBankIndex,
}

/// A builder for a register class definition.
pub struct IsaRegClassBuilder<'a> {
    class: &'a mut IsaRegClass,
    index: IsaRegClassIndex,
}

/// A reference to a register class.
pub type IsaRegClassIndex = usize;

/// A pattern.
pub struct IsaPattern {
    op: ir::Opcode,
    tys: Option<Vec<Type>>,
    args: Vec<IsaPatternArg>,
}

/// An argument to a pattern.
pub struct IsaPatternArg {
    dir: IsaPatternArgDir,
    kind: IsaPatternArgKind,
    binding: IsaPatternArgBinding,
}

enum IsaPatternArgDir {
    Out,
    In,
}

enum IsaPatternArgKind {
    Reg(IsaRegBankIndex, usize),
    RegClass(IsaRegClassIndex),
    AnyReg,  // mostly useful with IsaPatternArgBinding::Match
    Insn(Box<IsaPattern>),  // nested insn. Cannot be captured.
}

enum IsaPatternArgBinding {
    None,
    Capture(usize),  // only valid on register pattern args.
    Match(usize),    // only valid on register pattern args.
}

/// A binding from a successful pattern match.
pub enum IsaPatternMatchBinding {
    /// A bound register.
    Reg(IsaRegBankIndex, usize),
}

/// The context within an emit action.
pub struct IsaPatternEmitCtx<'a> {
    isa: &'a mut IsaDef,
    // TODO: in-progress bit patterns.
}

/// The context within a legalize action.
pub struct IsaPatternLegalizeCtx<'a> {
    isa: &'a mut IsaDef,
    // TODO: in-progress legalize output.
}

impl IsaDef {
    /// Create a new ISA definition.
    pub fn new() -> IsaDef {
        IsaDef {
            reg_banks: vec![],
            reg_classes: vec![],
            patterns: vec![],
        }
    }

    /// Add a register bank.
    pub fn reg_bank<'a>(&'a mut self, name: &'static str) -> IsaRegBankBuilder<'a> {
        let index = self.reg_banks.len();
        self.reg_banks.push(IsaRegBank {
            name,
            regs: 0,
            start_reg: 0,
            track_pressure: false,
            name_prefix: None,
            names: None,
        });
        IsaRegBankBuilder {
            bank: &mut self.reg_banks[index],
            index,
        }
    }

    /// Add a register class.
    pub fn reg_class<'a>(&'a mut self, name: &'static str) -> IsaRegClassBuilder<'a> {
        let index = self.reg_classes.len();
        self.reg_classes.push(IsaRegClass { name, bank: 0 });
        IsaRegClassBuilder {
            class: &mut self.reg_classes[index],
            index,
        }
    }

    /// Add an emission pattern.
    pub fn emit_pat<F>(&mut self, op: Opcode, args: &[IsaPatternArg], f: F)
    where
        F: for<'a> Fn(&mut IsaPatternEmitCtx<'a>, &[IsaPatternMatchBinding]),
    {
    }

    /// Get a pattern arg for an output register belonging to a class.
    pub fn out_rc(&self, rc: IsaRegClassIndex) -> IsaPatternArg {
        IsaPatternArg {
            dir: IsaPatternArgDir::Out,
            kind: IsaPatternArgKind::RegClass(rc),
        }
    }

    /// Get a pattern arg for an input register belonging to a class.
    pub fn in_rc(&self, rc: IsaRegClassIndex) -> IsaPatternArg {
        IsaPatternArg {
            dir: IsaPatternArgDir::In,
            kind: IsaPatternArgKind::RegClass(rc),
        }
    }
}

impl<'a> IsaRegBankBuilder<'a> {
    /// Set the number of registers.
    pub fn regs(&mut self, value: usize) -> &mut IsaRegBankBuilder<'a> {
        self.bank.regs = value;
        self
    }
    /// Set the `track_pressure` flag.
    pub fn track_pressure(&mut self, value: bool) -> &mut IsaRegBankBuilder<'a> {
        self.bank.track_pressure = value;
        self
    }
    /// Set the prefix for register names.
    pub fn name_prefix(&mut self, value: &'static str) -> &mut IsaRegBankBuilder<'a> {
        self.bank.name_prefix = Some(value);
        self
    }
    /// Set the explicit register names.
    pub fn names(&mut self, value: &'static [&'static str]) -> &mut IsaRegBankBuilder<'a> {
        self.bank.names = Some(value);
        self
    }
    /// Build the Regbank.
    pub fn build(&self) -> IsaRegBankIndex {
        self.index
    }
}

impl<'a> IsaRegClassBuilder<'a> {
    /// Set the bank from which this RegClass draws.
    pub fn bank(&mut self, value: IsaRegBankIndex) -> &mut IsaRegClassBuilder<'a> {
        self.class.bank = value;
        self
    }
    /// Build the RegClass.
    pub fn build(&self) -> IsaRegBankIndex {
        self.index
    }
}

impl<'a> IsaPatternEmitCtx<'a> {
    /// Emit some bits.
    pub fn bits(&mut self, nbits: usize, pattern: u64) {}
    /// Emit a register number.
    pub fn reg(&mut self, reg: &IsaPatternMatchBinding) {}
}
