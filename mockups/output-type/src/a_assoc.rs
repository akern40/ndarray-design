//! Mockup A: **per-operation associated types** — today's `Dimension::{Smaller, Larger}`
//! pattern, extended with what const axes need (`Erased`) and with broadcasting as a
//! pairwise trait (today's `DimMax`).
//!
//! No rank→type map is needed: every concrete shape names its own neighbours.
//! Cost: (#rank-changing ops) impls per type, plus (#types)² impls for every binary op.

use crate::rank::{DynRank, RMax, Rank};
use crate::shape::*;

pub trait RankOps: Shape
{
    /// Result of removing any one axis.
    type Smaller: RankOps<NDim = <Self::NDim as Rank>::Smaller>;
    /// Result of inserting an axis anywhere.
    type Larger: RankOps<NDim = <Self::NDim as Rank>::Larger>;
    /// Result of forgetting the rank.
    type Dyn: RankOps<NDim = DynRank>;
    /// Same rank, every compile-time length forgotten. Needed as soon as one axis
    /// can be `const`: slicing takes the axis at runtime, so the output type cannot
    /// depend on whether the const axis was the one sliced.
    type Erased: RankOps<NDim = Self::NDim>;

    fn remove_axis(&self, axis: usize) -> Self::Smaller;
    fn insert_axis(&self, axis: usize, len: usize) -> Self::Larger;
    fn into_dyn(&self) -> Self::Dyn;
    fn slice_axis(&self, axis: usize, len: usize) -> Self::Erased;
}

/// Today's `DimMax`: one impl per ordered pair of shape types.
pub trait BroadcastWith<Other: RankOps>: RankOps
where Self::NDim: RMax<Other::NDim>
{
    type Output: RankOps<NDim = <Self::NDim as RMax<Other::NDim>>::Output>;
    fn broadcast(&self, other: &Other) -> Option<Self::Output>;
}

macro_rules! impl_nshape {
    ($n:literal, smaller: $sm:ty, larger: $lg:ty) => {
        impl RankOps for NShape<$n>
        {
            type Smaller = $sm;
            type Larger = $lg;
            type Dyn = DShape;
            type Erased = Self;
            fn remove_axis(&self, axis: usize) -> Self::Smaller
            {
                removed(self, axis).into()
            }
            fn insert_axis(&self, axis: usize, len: usize) -> Self::Larger
            {
                inserted(self, axis, len).into()
            }
            fn into_dyn(&self) -> DShape
            {
                DShape(self.iter().collect())
            }
            fn slice_axis(&self, axis: usize, len: usize) -> Self
            {
                sliced(self, axis, len).into()
            }
        }
    };
}

impl<const N: usize> From<Vec<usize>> for NShape<N>
{
    fn from(v: Vec<usize>) -> Self
    {
        NShape(to_array(v))
    }
}
impl From<Vec<usize>> for DShape
{
    fn from(v: Vec<usize>) -> Self
    {
        DShape(v)
    }
}

impl_nshape!(0, smaller: NShape<0>, larger: NShape<1>);
impl_nshape!(1, smaller: NShape<0>, larger: NShape<2>);
impl_nshape!(2, smaller: NShape<1>, larger: NShape<3>);
impl_nshape!(3, smaller: NShape<2>, larger: NShape<4>);
impl_nshape!(4, smaller: NShape<3>, larger: DShape);

impl RankOps for DShape
{
    type Smaller = DShape;
    type Larger = DShape;
    type Dyn = DShape;
    type Erased = DShape;
    fn remove_axis(&self, axis: usize) -> DShape
    {
        DShape(removed(self, axis))
    }
    fn insert_axis(&self, axis: usize, len: usize) -> DShape
    {
        DShape(inserted(self, axis, len))
    }
    fn into_dyn(&self) -> DShape
    {
        self.clone()
    }
    fn slice_axis(&self, axis: usize, len: usize) -> DShape
    {
        DShape(sliced(self, axis, len))
    }
}

/// The partially-const shape: every runtime-axis operation erases the const length.
impl<const A: usize> RankOps for ConstShape2<A>
{
    type Smaller = NShape<1>;
    type Larger = NShape<3>;
    type Dyn = DShape;
    type Erased = NShape<2>;
    fn remove_axis(&self, axis: usize) -> NShape<1>
    {
        removed(self, axis).into()
    }
    fn insert_axis(&self, axis: usize, len: usize) -> NShape<3>
    {
        inserted(self, axis, len).into()
    }
    fn into_dyn(&self) -> DShape
    {
        DShape(self.iter().collect())
    }
    fn slice_axis(&self, axis: usize, len: usize) -> NShape<2>
    {
        sliced(self, axis, len).into()
    }
}

// ---- Broadcasting: the pairwise table. ----

macro_rules! impl_broadcast_nn {
    ($(($l:literal, $r:literal, $max:literal)),* $(,)?) => {
        $(
            impl BroadcastWith<NShape<$r>> for NShape<$l>
            {
                type Output = NShape<$max>;
                fn broadcast(&self, other: &NShape<$r>) -> Option<NShape<$max>>
                {
                    broadcast_lens(self, other).map(Into::into)
                }
            }
        )*
    };
}

impl_broadcast_nn!(
    (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4),
    (1, 0, 1), (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4),
    (2, 0, 2), (2, 1, 2), (2, 2, 2), (2, 3, 3), (2, 4, 4),
    (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 3, 3), (3, 4, 4),
    (4, 0, 4), (4, 1, 4), (4, 2, 4), (4, 3, 4), (4, 4, 4),
);

impl<const N: usize> BroadcastWith<DShape> for NShape<N>
where NShape<N>: RankOps<NDim = crate::rank::ConstRank<N>>
{
    type Output = DShape;
    fn broadcast(&self, other: &DShape) -> Option<DShape>
    {
        broadcast_lens(self, other).map(DShape)
    }
}
impl<const N: usize> BroadcastWith<NShape<N>> for DShape
where NShape<N>: RankOps<NDim = crate::rank::ConstRank<N>>
{
    type Output = DShape;
    fn broadcast(&self, other: &NShape<N>) -> Option<DShape>
    {
        broadcast_lens(self, other).map(DShape)
    }
}
impl BroadcastWith<DShape> for DShape
{
    type Output = DShape;
    fn broadcast(&self, other: &DShape) -> Option<DShape>
    {
        broadcast_lens(self, other).map(DShape)
    }
}
// ... and ConstShape2 needs its own row and column (only the cases the tests use):
impl<const A: usize> BroadcastWith<NShape<2>> for ConstShape2<A>
{
    type Output = NShape<2>;
    fn broadcast(&self, other: &NShape<2>) -> Option<NShape<2>>
    {
        broadcast_lens(self, other).map(Into::into)
    }
}
impl<const A: usize, const B: usize> BroadcastWith<ConstShape2<B>> for ConstShape2<A>
{
    type Output = NShape<2>;
    fn broadcast(&self, other: &ConstShape2<B>) -> Option<NShape<2>>
    {
        broadcast_lens(self, other).map(Into::into)
    }
}
