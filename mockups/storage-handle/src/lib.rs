//! Mockup for roadmap §1c: sparse is a *storage*, reached through a handle +
//! accessor (mdspan's `data_handle_type` + `AccessorPolicy`), while `Layout`
//! stays a total index → offset map. See `README.md` in this directory.
//!
//! The pieces, bottom-up:
//! - [`layout`]: `Layout` (total `offset`, inverse `index_of`, `is_unique`) and a
//!   strided impl, plus the structural ops (`AxisOps`) that only touch the layout.
//! - [`accessor`]: `Accessor` = element type + `Handle` + `get`/`get_mut`/`stored`.
//!   `Dense<A>` (handle is a pointer) and `Sparse<A>` (handle points at a
//!   heap-stable buffer of sorted `(offset, value)` pairs).
//! - [`storage`]: ownership kinds, each naming its accessor family.
//! - [`array`]: `ArrayBase<S, L>`, the `repr(transparent)` cast to
//!   `ArrayRef<S::Acc, L>`, and the methods that are generic over the accessor.

pub mod accessor;
pub mod array;
pub mod layout;
pub mod storage;

#[cfg(test)]
mod tests;

use array::{ArrayRef, DenseRef};
use layout::Strided;

/// For inspecting codegen: the dense unchecked access path should be
/// `ptr + i*s0 + j*s1`, with no branch introduced by the accessor indirection.
#[inline(never)]
pub fn dense_uget(a: &DenseRef<f64, Strided<2>>, i: usize, j: usize) -> f64
{
    unsafe { *a.uget([i, j]) }
}

/// Same, written against *any* accessor: one generic story for dense and sparse.
#[inline(never)]
pub fn generic_get<Acc: accessor::Accessor<Elem = f64>>(a: &ArrayRef<Acc, Strided<2>>, i: usize, j: usize) -> f64
{
    *a.get([i, j]).unwrap()
}
