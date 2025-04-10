mod dimensionality;
mod dyn_repr;
mod n_repr;
mod shape;
mod strides;

pub mod dimensionality;
mod dyn_repr;
mod n_repr;
pub mod shape;
pub mod strides;

pub use dyn_repr::{DShape, DStrides};
pub use n_repr::{NShape, NStrides};

use alloc::borrow::Cow;
use core::{any::type_name, fmt::Display, marker::PhantomData};
use dimensionality::{Dimensionality, NDim};
use shape::Shape;
use strides::Strides;

/// The error type for dealing with shapes and strides
#[derive(Debug, Clone, Copy)]
pub enum ShapeStrideError<S> {
    /// Out of bounds; specifically, using an index that is larger than the dimensionality of the shape or strides `S`.
    OutOfBounds(PhantomData<S>, usize),
    /// The error when trying to mutate a shape or strides whose element is a hard-coded constant.
    FixedIndex(PhantomData<S>, usize),
    /// The error when trying to construct or mutate a shape or strides with the wrong dimensionality value.
    BadDimality(PhantomData<S>, usize),
}

impl<S: Strides> Display for ShapeStrideError<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShapeStrideError::FixedIndex(_, idx) => write!(f, "Cannot index {} at {idx}", type_name::<S>()),
            ShapeStrideError::OutOfBounds(_, idx) =>
                write!(f, "Index {idx} is larger than the dimensionality of {}", type_name::<S>()),
            ShapeStrideError::BadDimality(_, dimality) => write!(f, "{} has a dimensionality of {}, which is incompatible with requested dimensionality of {dimality}", type_name::<S>(), type_name::<S::Dimality>()),
        }
    }
}

/// Trait that associates a dimensionality to a type
pub trait Dimensioned {
    type Dimality: Dimensionality;
}

impl<T, const N: usize> Dimensioned for [T; N]
where
    NDim<N>: Dimensionality,
{
    type Dimality = NDim<N>;
}

/// A trait capturing how an array is laid out in memory.
pub trait Layout: Dimensioned {
    /// The type of shape that the array uses.
    ///
    /// Must implement [`Shape`] and have the same dimensionality.
    type Shape: Shape<Dimality = Self::Dimality>;

    /// The index type that this layout uses; e.g., `[usize; N]`.
    ///
    /// Must have the same dimensionality.
    type Index: Dimensioned<Dimality = Self::Dimality>;

    /// Get the shape of the layout.
    ///
    /// If the implementing type does not carry a shape directly,
    /// one should be constructed and passed as [`Cow::Owned`].
    fn shape(&self) -> Cow<'_, Self::Shape>;

    /// Index into this layout in "linear" fashion by moving across axes from left to right.
    fn index_linear_left(&self, idx: usize) -> isize;

    /// Index into this layout in "linear" fashion by moving across axes from right to left.
    fn index_linear_right(&self, idx: usize) -> isize;

    /// Index into this layout by traversing it in a memory order that is as efficient as possible.
    fn index_memory_order(&self, idx: usize) -> isize;

    /// Index into this layout with a multidimensional index.
    fn index(&self, idx: Self::Index) -> isize;

    fn first_index(&self) -> Option<Self::Index>;

    fn next_for(&self, index: Self::Index) -> Option<Self::Index>;

    // Shortcut methods, we could add more of these
    fn ndim(&self) -> usize {
        self.shape().ndim()
    }

    fn size(&self) -> usize {
        self.shape().size()
    }

    fn size_checked(&self) -> Option<usize> {
        self.shape().size_checked()
    }
}

pub trait Strided: Layout {
    type Strides: Strides<Dimality = Self::Dimality>;

    fn strides(&self) -> Cow<'_, Self::Strides>;
}

pub struct NLayout<const N: usize> {
    shape: NShape<N>,
    strides: NStrides<N>,
}

impl<const N: usize> Dimensioned for NLayout<N>
where
    NDim<N>: Dimensionality,
{
    type Dimality = NDim<N>;
}

impl<const N: usize> Layout for NLayout<N>
where
    NDim<N>: Dimensionality,
{
    type Shape = NShape<N>;

    type Index = [usize; N];

    fn shape(&self) -> Cow<'_, Self::Shape> {
        Cow::Borrowed(&self.shape)
    }

    fn index_linear_left(&self, idx: usize) -> isize {
        todo!()
    }

    fn index_linear_right(&self, idx: usize) -> isize {
        todo!()
    }

    fn index_memory_order(&self, idx: usize) -> isize {
        todo!()
    }

    fn index(&self, index: Self::Index) -> isize {
        let mut offset = 0isize;
        for idx in 0..N {
            offset += (index[idx] as isize) * self.strides[idx];
        }
        offset
    }

    fn first_index(&self) -> Option<Self::Index> {
        todo!()
    }

    fn next_for(&self, index: Self::Index) -> Option<Self::Index> {
        todo!()
    }
}
