//! Mockups for the question "how does an operation name its output layout type?"
//! (ndarray-design `roadmap.md` §3.1). See `README.md` in this directory.
//!
//! Everything is done at the *shape* level: strides and offsets add nothing to the
//! output-*type* question, since a strided layout re-ranks exactly like its shape.

pub mod rank;
pub mod shape;

pub mod a_assoc;
pub mod b_gat;

#[cfg(test)]
mod tests;
