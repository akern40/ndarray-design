//! Mockup B: **one GAT, `WithRank<R>`**, that re-ranks a representation. All rank-changing
//! operations, `into_dyn`, erasure and broadcasting are expressed through it, so the trait
//! has one associated type instead of four plus a pairwise table.
//!
//! The catch: `WithRank<R>` has to be a single type expression in `R`, and on stable Rust
//! the only way to pick `[usize; N]` for `ConstRank<N>` and `Vec<usize>` for `DynRank` is a
//! trait implemented **on the rank types**. Here that is `Storage` (a rank→container map,
//! the least-knowledge version of "`Rank` as a type map"). It must hold for *every* rank
//! that can appear in a signature (`Smaller`, `Larger`, `RMax::Output`), which in practice
//! means it is a supertrait of `Rank` — see the comment on `Storage`.
//!
//! The pay-off is that the whole "standard" family is one struct, `Std<R>`, with one impl.

use core::fmt::Debug;

use crate::rank::{ConstRank, DynRank, RMax, Rank, Ranked};
use crate::shape::*;

/// Rank → fixed-length container. Implemented for every rank type.
///
/// If this were a *separate* trait not required by `Rank`, every generic signature using
/// `WithRank<X>` would need `where X: Storage` for each rank expression `X` it mentions
/// (`<S::NDim as Rank>::Smaller: Storage`, ...). The mockup avoids that by making it a
/// blanket requirement through the `Reranked` bound instead (see `Reranked::WithRank`).
pub trait Storage: Rank
{
    type Array<T: Copy + Debug + Eq>: AsRef<[T]> + Clone + Debug + Eq + From<Vec<T>>;
}

impl<const N: usize> Storage for ConstRank<N>
where ConstRank<N>: Rank
{
    type Array<T: Copy + Debug + Eq> = Arr<T, N>;
}

impl Storage for DynRank
{
    type Array<T: Copy + Debug + Eq> = Vec<T>;
}

/// `[T; N]` newtype, only because `[T; N]: From<Vec<T>>` doesn't exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arr<T, const N: usize>(pub [T; N]);
impl<T, const N: usize> AsRef<[T]> for Arr<T, N>
{
    fn as_ref(&self) -> &[T]
    {
        &self.0
    }
}
impl<T: Debug, const N: usize> From<Vec<T>> for Arr<T, N>
{
    fn from(v: Vec<T>) -> Self
    {
        Arr(v.try_into().unwrap_or_else(|v: Vec<T>| panic!("rank {} != {N}", v.len())))
    }
}

/// The standard shape family: `Std<ConstRank<2>>` is today's `Ix2`, `Std<DynRank>` is `IxDyn`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Std<R: Storage>(pub R::Array<usize>);

impl<R: Storage> Ranked for Std<R>
{
    type NDim = R;
    fn ndim(&self) -> usize
    {
        self.0.as_ref().len()
    }
}

impl<R: Storage> Shape for Std<R>
{
    fn axis_len(&self, axis: usize) -> usize
    {
        self.0.as_ref()[axis]
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_
    {
        self.0.as_ref().iter().copied()
    }
}

pub trait Reranked: Shape
{
    /// This representation family at rank `R`.
    ///
    /// `R: Storage` rather than `R: Rank` is where the rank→type map leaks into the trait.
    type WithRank<R: Storage>: Reranked<NDim = R>;

    fn remove_axis(&self, axis: usize) -> Self::WithRank<<Self::NDim as Rank>::Smaller>
    where <Self::NDim as Rank>::Smaller: Storage;
    fn insert_axis(&self, axis: usize, len: usize) -> Self::WithRank<<Self::NDim as Rank>::Larger>
    where <Self::NDim as Rank>::Larger: Storage;
    fn into_dyn(&self) -> Self::WithRank<DynRank>;
    /// Same rank, compile-time lengths forgotten.
    fn slice_axis(&self, axis: usize, len: usize) -> Self::WithRank<Self::NDim>
    where Self::NDim: Storage;
    /// Build a member of this family at rank `R2` from runtime lengths. With this,
    /// binary operations (broadcasting, concatenation) are *free functions* with
    /// rank-level bounds only; no pairwise trait table.
    fn build<R2: Storage>(lens: Vec<usize>) -> Self::WithRank<R2>;
}

/// Broadcasting without a pairwise table: the output rank comes from `RMax`, the output
/// representation from `S1`'s family. (It would need a trait method if `RMax` were a
/// where-clause on a trait method: rustc rejects `Self::NDim: RMax<..>` on impls as a
/// "stricter requirement" once `Self::NDim` normalizes — a real friction with this design.)
pub fn broadcast<S1: Reranked, S2: Shape>(a: &S1, b: &S2) -> Option<S1::WithRank<<S1::NDim as RMax<S2::NDim>>::Output>>
where
    S1::NDim: RMax<S2::NDim>,
    <S1::NDim as RMax<S2::NDim>>::Output: Storage,
{
    broadcast_lens(a, b).map(S1::build)
}

impl<R: Storage> Reranked for Std<R>
{
    type WithRank<R2: Storage> = Std<R2>;

    fn remove_axis(&self, axis: usize) -> Std<R::Smaller>
    where R::Smaller: Storage
    {
        Std(removed(self, axis).into())
    }
    fn insert_axis(&self, axis: usize, len: usize) -> Std<R::Larger>
    where R::Larger: Storage
    {
        Std(inserted(self, axis, len).into())
    }
    fn into_dyn(&self) -> Std<DynRank>
    {
        Std(self.iter().collect())
    }
    fn slice_axis(&self, axis: usize, len: usize) -> Std<R>
    {
        Std(sliced(self, axis, len).into())
    }
    fn build<R2: Storage>(lens: Vec<usize>) -> Std<R2>
    {
        Std(lens.into())
    }
}

/// The partially-const shape joins the standard family by erasing into it.
impl<const A: usize> Reranked for ConstShape2<A>
{
    type WithRank<R2: Storage> = Std<R2>;

    fn remove_axis(&self, axis: usize) -> Std<ConstRank<1>>
    {
        Std(removed(self, axis).into())
    }
    fn insert_axis(&self, axis: usize, len: usize) -> Std<ConstRank<3>>
    {
        Std(inserted(self, axis, len).into())
    }
    fn into_dyn(&self) -> Std<DynRank>
    {
        Std(self.iter().collect())
    }
    fn slice_axis(&self, axis: usize, len: usize) -> Std<ConstRank<2>>
    {
        Std(sliced(self, axis, len).into())
    }
    fn build<R2: Storage>(lens: Vec<usize>) -> Std<R2>
    {
        Std(lens.into())
    }
}
