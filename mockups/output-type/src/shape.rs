//! Shared minimal `Shape` (as on the `shape-trait` branch) and three concrete shapes
//! used by every mockup:
//!
//! - `NShape<N>`: const rank, runtime lengths (today's `Ix<N>`).
//! - `DShape`: dynamic rank (today's `IxDyn`).
//! - `ConstShape2<A>`: rank 2, axis 0 has the *const* length `A`, axis 1 is runtime.
//!   This is the "partially const" case used to see what erasure looks like.

use core::fmt::Debug;

use crate::rank::{ConstRank, DynRank, Rank, Ranked};

pub trait Shape: Ranked + Clone + Debug + Eq
{
    fn axis_len(&self, axis: usize) -> usize;
    fn iter(&self) -> impl Iterator<Item = usize> + '_;
    fn size(&self) -> usize
    {
        self.iter().product()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NShape<const N: usize>(pub [usize; N]);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DShape(pub Vec<usize>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstShape2<const A: usize>(pub usize);

impl<const N: usize> Ranked for NShape<N>
where ConstRank<N>: Rank
{
    type NDim = ConstRank<N>;
    fn ndim(&self) -> usize
    {
        N
    }
}

impl<const N: usize> Shape for NShape<N>
where ConstRank<N>: Rank
{
    fn axis_len(&self, axis: usize) -> usize
    {
        self.0[axis]
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_
    {
        self.0.iter().copied()
    }
}

impl Ranked for DShape
{
    type NDim = DynRank;
    fn ndim(&self) -> usize
    {
        self.0.len()
    }
}

impl Shape for DShape
{
    fn axis_len(&self, axis: usize) -> usize
    {
        self.0[axis]
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_
    {
        self.0.iter().copied()
    }
}

impl<const A: usize> Ranked for ConstShape2<A>
{
    type NDim = ConstRank<2>;
    fn ndim(&self) -> usize
    {
        2
    }
}

impl<const A: usize> Shape for ConstShape2<A>
{
    fn axis_len(&self, axis: usize) -> usize
    {
        [A, self.0][axis]
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_
    {
        [A, self.0].into_iter()
    }
}

/// Helpers shared by the mockups: build the lengths of the result as a `Vec`, then let
/// each mockup turn that into its output type. (Allocation is irrelevant here; only the
/// *types* are under study.)
pub fn removed(s: &impl Shape, axis: usize) -> Vec<usize>
{
    let mut v: Vec<usize> = s.iter().collect();
    v.remove(axis);
    v
}

pub fn inserted(s: &impl Shape, axis: usize, len: usize) -> Vec<usize>
{
    let mut v: Vec<usize> = s.iter().collect();
    v.insert(axis, len);
    v
}

pub fn sliced(s: &impl Shape, axis: usize, len: usize) -> Vec<usize>
{
    let mut v: Vec<usize> = s.iter().collect();
    v[axis] = len;
    v
}

pub fn broadcast_lens(a: &impl Shape, b: &impl Shape) -> Option<Vec<usize>>
{
    let (a, b): (Vec<_>, Vec<_>) = (a.iter().collect(), b.iter().collect());
    let (long, short) = if a.len() >= b.len() { (&a, &b) } else { (&b, &a) };
    let pad = long.len() - short.len();
    let mut out = long.clone();
    for (i, &s) in short.iter().enumerate() {
        let l = long[pad + i];
        out[pad + i] = match (l, s) {
            (l, s) if l == s => l,
            (1, s) => s,
            (l, 1) => l,
            _ => return None,
        };
    }
    Some(out)
}

/// Convert a `Vec` of lengths into a fixed-rank array; panics on rank mismatch
/// (a real implementation would carry the rank proof through the operation).
pub fn to_array<const N: usize>(v: Vec<usize>) -> [usize; N]
{
    v.try_into().unwrap_or_else(|v: Vec<usize>| panic!("rank {} != {N}", v.len()))
}
