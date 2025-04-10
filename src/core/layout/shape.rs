use alloc::borrow::Cow;
use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{DShape, Dimensioned, NShape};

use super::dimensionality::{DMax, Dimensionality, NDim};
use super::ShapeStrideError;

/// The trait for types that contain the shape of a multidimensional array.
///
/// To be specific, "shape" in this context for an `N`-dimensional array is a list of
/// `N` `usize` values, where the `i`th value describes the length of the `i`th dimension
/// of the `N`-dimensional array.
///
/// Types that implement `Shape` should not carry information about the strides
/// of an array, i.e., how the array's elements are laid out in memory. They only
/// inform the user about the lengths of each dimension.
///
/// # Constant Shapes
/// Having one or more axes of a shape be `const`-valued can be critical for array performance.
/// The `Shape` trait is agnostic as to whether any or all of the axes are `const` - as a result,
/// `Shape` is a read-only construct. See [`AxisMut`] for shapes that have a  non-`const` axis length.
/// See [`ShapeMut`] for shapes that don't have any `const` axis lengths.
pub trait Shape:
    Dimensioned + Index<usize, Output = usize> + Eq + Clone + Send + Sync + Debug
{
    /// Get the shape as a (possibly-borrowed) slice of `usize`.
    fn as_slice(&self) -> Cow<'_, [usize]>;

    /// Try to index the shape mutably, if `index` is a non-`const` axis.
    ///
    /// Types that implement `Shape` do not guarantee any sort of mutability; this method
    /// allows a non-panicking way to discover whether a given axis of a shape is mutable.
    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>>;

    /// Try to create an `ndim`-dimensional shape filled with `value`.
    ///
    /// This may fail if either the number of dimensions does not match the dimensionality
    /// of the shape, or if the shape has any `const` axis lengths.
    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>>;

    /// Convert this shape into a dynamic-dimensional shape.
    fn to_dyn(&self) -> DShape {
        self.as_slice().into()
    }

    /// Get the runtime dimensionality of the shape.
    ///
    /// Implementors of `Shape` must guarantee that this value will match [`Shape::Dimality`].
    fn ndim(&self) -> usize {
        self.as_slice().len()
    }

    /// Get the number of elements that the array contains.
    fn size(&self) -> usize {
        self.as_slice().iter().product()
    }

    /// Get the number of elements that the array contains, checking for overflow.
    fn size_checked(&self) -> Option<usize> {
        self.as_slice()
            .iter()
            .try_fold(1_usize, |acc, &i| acc.checked_mul(i))
    }

    /// Try to turn this shape into a constant `N`-dimensional shape.
    fn try_to_nshape<const N: usize>(&self) -> Result<NShape<N>, ShapeStrideError<NShape<N>>> {
        if self.ndim() == N {
            let mut values = [0; N];
            values.split_at_mut(N).0.copy_from_slice(&self.as_slice());
            Ok(values.into())
        } else {
            Err(ShapeStrideError::BadDimality(PhantomData, self.ndim()))
        }
    }
}

/// A shape whose `N`th dimension length is mutable.
pub trait AxisMut<const N: usize>: Shape
where
    NDim<N>: Dimensionality + DMax<Self::Dimality, Output = Self::Dimality>,
{
    /// Get a mutable reference to the shape at the constant index.
    ///
    /// # Panics
    /// This function panics if it is called on a dynamic-dimensionality shape whose
    /// runtime dimensionality is less than `N`.
    fn get_mut(&mut self) -> &mut usize;
}

/// A shape with no `const`-valued axis lengths; i.e., a mutable shape.
///
/// Implementing `ShapeMut` will automatically implement [`AxisMut<N>`] for all `N` less than
/// the dimensionality of the shape. Note that dynamic-dimensionality shapes (i.e.,
/// `Shape<Dimality = DDyn>`) will implement `AxisMut` for all values of `N`.
pub trait ShapeMut: Shape + IndexMut<usize, Output = usize>
// + AddAssign
// + for<'a> AddAssign<&'a Self>
// + SubAssign
// + for<'a> SubAssign<&'a Self>
// + MulAssign
// + for<'a> MulAssign<&'a Self>
// + MulAssign<usize>
{
}

/// Implement `AxisMut` for axes less than the dimensionality.
///
/// This also implements `AxisMut` for all axes if the shape has a dynamic dimensionality.
impl<const N: usize, T> AxisMut<N> for T
where
    T: ShapeMut,
    NDim<N>: Dimensionality + DMax<T::Dimality, Output = T::Dimality>,
{
    fn get_mut(&mut self) -> &mut usize {
        &mut self[N]
    }
}

/// A conversion trait for types that can turn into a shape.
pub trait IntoShape {
    type Dimality: Dimensionality;

    type Shape: Shape<Dimality = Self::Dimality>;

    fn into_shape(&self) -> Self::Shape;
}

impl<T> IntoShape for T
where
    T: Shape,
{
    type Dimality = T::Dimality;

    type Shape = T;

    fn into_shape(&self) -> Self::Shape {
        self.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstMatrixShape<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> Dimensioned for ConstMatrixShape<N, M> {
    type Dimality = NDim<2>;
}

impl<const N: usize, const M: usize> Index<usize> for ConstMatrixShape<N, M> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &N,
            1 => &M,
            _ => panic!("Index {index} out of bounds for ConstMatrixShape"),
        }
    }
}

impl<const N: usize, const M: usize> Shape for ConstMatrixShape<N, M> {
    fn as_slice(&self) -> Cow<'_, [usize]> {
        Cow::Borrowed(&[N, M])
    }

    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>> {
        if index < 2 {
            Err(ShapeStrideError::FixedIndex(PhantomData, index))
        } else {
            Err(ShapeStrideError::OutOfBounds(PhantomData, index))
        }
    }

    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>> {
        if ndim != 2 {
            Err(ShapeStrideError::BadDimality(PhantomData, ndim))
        } else if value != N {
            Err(ShapeStrideError::FixedIndex(PhantomData, 0))
        } else if value != N {
            Err(ShapeStrideError::FixedIndex(PhantomData, 1))
        } else {
            Ok(Self)
        }
    }
}
