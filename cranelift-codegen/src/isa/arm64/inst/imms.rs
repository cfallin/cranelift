//! ARM64 ISA definitions: immediate constants.

#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::ir::types::*;
use crate::ir::Type;
use crate::machinst::*;

use regalloc::RealRegUniverse;

use std::string::String;

/// A signed, scaled 7-bit offset.
#[derive(Clone, Copy, Debug)]
pub struct SImm7Scaled {
    /// The value.
    pub value: i16,
    /// multiplied by the size of this type
    pub scale_ty: Type,
}

impl SImm7Scaled {
    /// Create a SImm7Scaled from a raw offset and the known scale type, if
    /// possible.
    pub fn maybe_from_i64(value: i64, scale_ty: Type) -> Option<SImm7Scaled> {
        assert!(scale_ty == I64 || scale_ty == I32);
        let scale = scale_ty.bytes();
        assert!(scale.is_power_of_two());
        let scale = scale as i64;
        let upper_limit = 63 * scale;
        let lower_limit = -(64 * scale);
        if value >= lower_limit && value <= upper_limit && (value & (scale - 1)) == 0 {
            Some(SImm7Scaled {
                value: value as i16,
                scale_ty,
            })
        } else {
            None
        }
    }

    /// Create a zero immediate of this format.
    pub fn zero(scale_ty: Type) -> SImm7Scaled {
        SImm7Scaled { value: 0, scale_ty }
    }

    /// Bits for encoding.
    pub fn bits(&self) -> u32 {
        ((self.value / self.scale_ty.bytes() as i16) as u32) & 0x7f
    }
}

/// a 9-bit signed offset.
#[derive(Clone, Copy, Debug)]
pub struct SImm9 {
    /// The value.
    pub value: i16,
}

impl SImm9 {
    /// Create a signed 9-bit offset from a full-range value, if possible.
    pub fn maybe_from_i64(value: i64) -> Option<SImm9> {
        if value >= -256 && value <= 255 {
            Some(SImm9 {
                value: value as i16,
            })
        } else {
            None
        }
    }

    /// Create a zero immediate of this format.
    pub fn zero() -> SImm9 {
        SImm9 { value: 0 }
    }

    /// Bits for encoding.
    pub fn bits(&self) -> u32 {
        (self.value as u32) & 0x1ff
    }
}

/// An unsigned, scaled 12-bit offset.
#[derive(Clone, Copy, Debug)]
pub struct UImm12Scaled {
    /// The value.
    pub value: u16,
    /// multiplied by the size of this type
    pub scale_ty: Type,
}

impl UImm12Scaled {
    /// Create a UImm12Scaled from a raw offset and the known scale type, if
    /// possible.
    pub fn maybe_from_i64(value: i64, scale_ty: Type) -> Option<UImm12Scaled> {
        let scale = scale_ty.bytes();
        assert!(scale.is_power_of_two());
        let scale = scale as i64;
        let limit = 4095 * scale;
        if value >= 0 && value <= limit && (value & (scale - 1)) == 0 {
            Some(UImm12Scaled {
                value: value as u16,
                scale_ty,
            })
        } else {
            None
        }
    }

    /// Create a zero immediate of this format.
    pub fn zero(scale_ty: Type) -> UImm12Scaled {
        UImm12Scaled { value: 0, scale_ty }
    }

    /// Encoded bits.
    pub fn bits(&self) -> u32 {
        (self.value as u32 / self.scale_ty.bytes()) & 0xfff
    }
}

/// A shifted immediate value in 'imm12' format: supports 12 bits, shifted
/// left by 0 or 12 places.
#[derive(Clone, Debug)]
pub struct Imm12 {
    /// The immediate bits.
    pub bits: usize,
    /// Whether the immediate bits are shifted left by 12 or not.
    pub shift12: bool,
}

impl Imm12 {
    /// Compute a Imm12 from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<Imm12> {
        if val == 0 {
            Some(Imm12 {
                bits: 0,
                shift12: false,
            })
        } else if val < 0xfff {
            Some(Imm12 {
                bits: val as usize,
                shift12: false,
            })
        } else if val < 0xfff_000 && (val & 0xfff == 0) {
            Some(Imm12 {
                bits: (val as usize) >> 12,
                shift12: true,
            })
        } else {
            None
        }
    }

    /// Bits for 2-bit "shift" field in e.g. AddI.
    pub fn shift_bits(&self) -> u8 {
        if self.shift12 {
            0b01
        } else {
            0b00
        }
    }

    /// Bits for 12-bit "imm" field in e.g. AddI.
    pub fn imm_bits(&self) -> u16 {
        self.bits as u16
    }
}

/// An immediate for logical instructions.
#[derive(Clone, Debug)]
pub struct ImmLogic {
    /// `N` flag.
    pub N: bool,
    /// `S` field: element size and element bits.
    pub R: u8,
    /// `R` field: rotate amount.
    pub S: u8,
}

impl ImmLogic {
    /// Compute an ImmLogic from raw bits, if possible.
    pub fn maybe_from_u64(_val: u64) -> Option<ImmLogic> {
        // TODO: implement.
        None
    }

    /// Returns bits ready for encoding: (N:1, R:6, S:6)
    pub fn enc_bits(&self) -> u16 {
        ((self.N as u16) << 12) | ((self.R as u16) << 6) | (self.S as u16)
    }

    /// Returns the value that this immediate represents.
    pub fn value(&self) -> u64 {
        unimplemented!()
    }

    /// Return an immediate for the bitwise-inverted value.
    /// TODO: is this always possible? If not, isel for AndNot/OrNot/XorNot
    /// will have to compensate.
    pub fn invert(&self) -> ImmLogic {
        unimplemented!()
    }
}

/// An immediate for shift instructions.
#[derive(Clone, Debug)]
pub struct ImmShift {
    /// 6-bit shift amount.
    pub imm: u8,
}

impl ImmShift {
    /// Create an ImmShift from raw bits, if possible.
    pub fn maybe_from_u64(val: u64) -> Option<ImmShift> {
        if val < 64 {
            Some(ImmShift { imm: val as u8 })
        } else {
            None
        }
    }

    /// Get the immediate value.
    pub fn value(&self) -> u8 {
        self.imm
    }
}

/// A 16-bit immediate for a MOVZ instruction, with a {0,16,32,48}-bit shift.
#[derive(Clone, Copy, Debug)]
pub struct MoveWideConst {
    /// The value.
    pub bits: u16,
    /// shifted 16*shift bits to the left.
    pub shift: u8,
}

impl MoveWideConst {
    /// Construct a MoveWideConst from an arbitrary 64-bit constant if possible.
    pub fn maybe_from_u64(value: u64) -> Option<MoveWideConst> {
        let mask0 = 0x0000_0000_0000_ffffu64;
        let mask1 = 0x0000_0000_ffff_0000u64;
        let mask2 = 0x0000_ffff_0000_0000u64;
        let mask3 = 0xffff_0000_0000_0000u64;

        if value == (value & mask0) {
            return Some(MoveWideConst {
                bits: (value & mask0) as u16,
                shift: 0,
            });
        }
        if value == (value & mask1) {
            return Some(MoveWideConst {
                bits: ((value >> 16) & mask0) as u16,
                shift: 1,
            });
        }
        if value == (value & mask2) {
            return Some(MoveWideConst {
                bits: ((value >> 32) & mask0) as u16,
                shift: 2,
            });
        }
        if value == (value & mask3) {
            return Some(MoveWideConst {
                bits: ((value >> 48) & mask0) as u16,
                shift: 3,
            });
        }
        None
    }

    /// Returns the value that this constant represents.
    pub fn value(&self) -> u64 {
        (self.bits as u64) << (16 * self.shift)
    }
}

impl ShowWithRRU for Imm12 {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        let shift = if self.shift12 { 12 } else { 0 };
        let value = self.bits << shift;
        format!("#{}", value)
    }
}

impl ShowWithRRU for SImm7Scaled {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value)
    }
}

impl ShowWithRRU for SImm9 {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value)
    }
}

impl ShowWithRRU for UImm12Scaled {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value)
    }
}

impl ShowWithRRU for ImmLogic {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value())
    }
}

impl ShowWithRRU for ImmShift {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.imm)
    }
}

impl ShowWithRRU for MoveWideConst {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("#{}", self.value())
    }
}
