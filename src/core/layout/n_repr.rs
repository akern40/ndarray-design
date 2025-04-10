use core::{
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use alloc::borrow::Cow;

use crate::{
    dimensionality::{Dimensionality, NDim},
    shape::{IntoShape, Shape, ShapeMut},
    strides::{DefaultC, DefaultF, IntoStrides, Strides, StridesMut},
    Dimensioned, ShapeStrideError,
};

/// A wrapper for fixed-length arrays that can be used for shape and strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeStrideN<T, const N: usize>([T; N]);

pub type NShape<const N: usize> = ShapeStrideN<usize, N>;

pub type NStrides<const N: usize> = ShapeStrideN<isize, N>;

impl<T, const N: usize> Deref for ShapeStrideN<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for ShapeStrideN<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize> Index<usize> for ShapeStrideN<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for ShapeStrideN<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const N: usize> From<[T; N]> for ShapeStrideN<T, N> {
    fn from(value: [T; N]) -> Self {
        Self { 0: value }
    }
}

impl<T, const N: usize> From<ShapeStrideN<T, N>> for [T; N] {
    fn from(value: ShapeStrideN<T, N>) -> Self {
        value.0
    }
}

impl<Rhs, T, const N: usize> PartialEq<Rhs> for ShapeStrideN<T, N>
where
    T: PartialEq,
    Rhs: AsRef<[T]>,
{
    fn eq(&self, other: &Rhs) -> bool {
        let other = other.as_ref();
        if other.len() != N {
            return false;
        }
        for i in 0..N {
            if self[i] != other[i] {
                return false;
            }
        }
        return true;
    }
}

impl<T, const N: usize> Dimensioned for ShapeStrideN<T, N>
where
    NDim<N>: Dimensionality,
{
    type Dimality = NDim<N>;
}

impl<const N: usize> IntoShape for [usize; N]
where
    NDim<N>: Dimensionality,
{
    type Dimality = NDim<N>;

    type Shape = NShape<N>;

    fn into_shape(&self) -> Self::Shape {
        (*self).into()
    }
}

macro_rules! impl_op {
    ($op_trait:ty, $op_fn:ident, $op_assign_trait:ty, $op_assign_fn:ident) => {
        impl<Rhs, const N: usize> $op_trait for NShape<N>
        where
            Rhs: IntoShape<Dimality = NDim<N>>,
        {
            type Output = NShape<N>;

            fn $op_fn(self, rhs: Rhs) -> Self::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        impl<Rhs, const N: usize> $op_assign_trait for NShape<N>
        where
            Rhs: IntoShape<Dimality = NDim<N>>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into_shape();
                for i in 0..N {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }

        impl<Rhs, const N: usize> $op_trait for NStrides<N>
        where
            Rhs: IntoStrides<Dimality = NDim<N>>,
        {
            type Output = NStrides<N>;

            fn $op_fn(self, rhs: Rhs) -> Self::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        impl<Rhs, const N: usize> $op_assign_trait for NStrides<N>
        where
            Rhs: IntoStrides<Dimality = NDim<N>>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into_strides();
                for i in 0..N {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }
    };
}

impl_op!(Add<Rhs>, add, AddAssign<Rhs>, add_assign);
impl_op!(Sub<Rhs>, sub, SubAssign<Rhs>, sub_assign);
impl_op!(Mul<Rhs>, mul, MulAssign<Rhs>, mul_assign);

impl<const N: usize> Add<usize> for NShape<N> {
    type Output = NShape<N>;

    fn add(self, rhs: usize) -> Self::Output {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<const N: usize> AddAssign<usize> for NShape<N> {
    fn add_assign(&mut self, rhs: usize) {
        for o in self.iter_mut() {
            *o += rhs;
        }
    }
}

impl<const N: usize> Mul<usize> for NShape<N> {
    type Output = NShape<N>;

    fn mul(self, rhs: usize) -> Self::Output {
        let mut output = self.clone();
        for o in output.iter_mut() {
            *o = *o * rhs;
        }
        output
    }
}

impl<const N: usize> MulAssign<usize> for NShape<N> {
    fn mul_assign(&mut self, rhs: usize) {
        for o in self.iter_mut() {
            *o += rhs;
        }
    }
}

impl<const N: usize> Shape for NShape<N>
where
    NDim<N>: Dimensionality,
{
    fn as_slice(&self) -> Cow<'_, [usize]> {
        Cow::Borrowed(self.deref())
    }

    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>> {
        if index < N {
            Ok(&mut self[index])
        } else {
            Err(ShapeStrideError::OutOfBounds(PhantomData, index))
        }
    }

    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>> {
        if ndim == N {
            Ok([value; N].into())
        } else {
            Err(ShapeStrideError::BadDimality(PhantomData, ndim))
        }
    }

    fn ndim(&self) -> usize {
        N
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn size_checked(&self) -> Option<usize> {
        self.iter().try_fold(1_usize, |acc, &i| acc.checked_mul(i))
    }
}

impl<const N: usize> ShapeMut for NShape<N> where NDim<N>: Dimensionality {}

impl<const N: usize> Strides for NStrides<N>
where
    NDim<N>: Dimensionality,
{
    fn as_slice(&self) -> Cow<'_, [isize]> {
        Cow::Borrowed(self.deref())
    }

    fn ndim(&self) -> usize {
        N
    }

    fn is_c_order(&self) -> bool {
        self.is_sorted_by(|a, b| a >= b)
    }

    fn is_f_order(&self) -> bool {
        self.is_sorted()
    }
}

impl<const N: usize> StridesMut for NStrides<N> where NDim<N>: Dimensionality {}

impl<const N: usize> DefaultC for NStrides<N>
where
    NDim<N>: Dimensionality,
{
    fn default_c<Sh>(shape: Sh) -> Self
    where
        Sh: IntoShape<Dimality = Self::Dimality>,
    {
        let shape = shape.into_shape();
        let mut strides = [1isize; N];
        for i in 1..N {
            strides[N - i - 1] = strides[N - i] * (shape[N - i] as isize);
        }
        return strides.into();
    }
}

impl<const N: usize> DefaultF for NStrides<N>
where
    NDim<N>: Dimensionality,
{
    fn default_f<Sh>(shape: Sh) -> Self
    where
        Sh: IntoShape<Dimality = Self::Dimality>,
    {
        let shape = shape.into_shape();
        let mut strides = [1isize; N];
        for i in 1..N {
            strides[i] = strides[i - 1] * (shape[i] as isize);
        }
        return strides.into();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        strides::{DefaultC, DefaultF},
        NStrides,
    };

    #[test]
    fn test_default_strides() {
        let shape = [2, 3, 4];
        let strides = NStrides::default_c(shape);
        assert_eq!(strides, [12, 4, 1]);

        let strides = NStrides::default_f(shape);
        assert_eq!(strides, [1, 3, 12]);
    }
}
