use crate::rng::{Rng32, Rng64};
use std::io::{self, Read};

/// Adapts any [`Rng32`] into a [`Read`] byte stream.
///
/// Bytes are emitted in little-endian order from each `u32` output.
/// The stream is infinite; it never returns `Ok(0)`.
pub struct ByteStream32<R: Rng32> {
    rng: R,
    buf: [u8; 4],
    pos: usize,
}

impl<R: Rng32> ByteStream32<R> {
    pub fn new(rng: R) -> Self {
        Self { rng, buf: [0; 4], pos: 4 }
    }
}

impl<R: Rng32> Read for ByteStream32<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        for (i, byte) in out.iter_mut().enumerate() {
            if self.pos >= 4 {
                self.buf = self.rng.nextu().to_le_bytes();
                self.pos = 0;
            }
            *byte = self.buf[self.pos];
            self.pos += 1;
            // Return early in large chunks to avoid long blocking loops.
            // Callers re-invoke read() for the rest.
            if i == 65535 {
                return Ok(i + 1);
            }
        }
        Ok(out.len())
    }
}

/// Adapts any [`Rng64`] into a [`Read`] byte stream.
///
/// Bytes are emitted in little-endian order from each `u64` output.
/// The stream is infinite; it never returns `Ok(0)`.
pub struct ByteStream64<R: Rng64> {
    rng: R,
    buf: [u8; 8],
    pos: usize,
}

impl<R: Rng64> ByteStream64<R> {
    pub fn new(rng: R) -> Self {
        Self { rng, buf: [0; 8], pos: 8 }
    }
}

impl<R: Rng64> Read for ByteStream64<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        for (i, byte) in out.iter_mut().enumerate() {
            if self.pos >= 8 {
                self.buf = self.rng.nextu().to_le_bytes();
                self.pos = 0;
            }
            *byte = self.buf[self.pos];
            self.pos += 1;
            if i == 65535 {
                return Ok(i + 1);
            }
        }
        Ok(out.len())
    }
}
