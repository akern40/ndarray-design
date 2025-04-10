use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use core::ops::{Deref, DerefMut, Index, IndexMut};

use crate::strides::StridesMut;
use crate::Dimensioned;
use crate::{
    dimensionality::DDyn,
    shape::{IntoShape, Shape, ShapeMut},
    strides::{IntoStrides, Strides},
    ShapeStrideError,
};

const CAP: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynAxesRepr<T> {
    Inline(usize, [T; CAP]),
    Alloc(Box<[T]>),
}

pub type DShape = DynAxesRepr<usize>;
pub type DStrides = DynAxesRepr<isize>;

impl<T> DynAxesRepr<T> {
    fn ndim(&self) -> usize {
        match self {
            DynAxesRepr::Inline(len, _) => *len,
            DynAxesRepr::Alloc(items) => items.len(),
        }
    }
}

impl<T> Deref for DynAxesRepr<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            DynAxesRepr::Inline(len, arr) => {
                debug_assert!(*len <= arr.len());
                unsafe { arr.get_unchecked(..*len) }
            }
            DynAxesRepr::Alloc(items) => items,
        }
    }
}

impl<T> DerefMut for DynAxesRepr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        todo!()
    }
}

impl<T> Index<usize> for DynAxesRepr<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &(**self)[index]
    }
}

impl<T> IndexMut<usize> for DynAxesRepr<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut (**self)[index]
    }
}

impl<Rhs, T> From<Rhs> for DynAxesRepr<T>
where
    Rhs: AsRef<[T]>,
    T: Default + Copy,
{
    fn from(value: Rhs) -> Self {
        let value = value.as_ref();
        let n = value.len();
        if n <= CAP {
            let mut inline = [T::default(); CAP];
            inline.split_at_mut(CAP).0.copy_from_slice(value);
            Self::Inline(n, inline)
        } else {
            Self::Alloc(value.into())
        }
    }
}

impl<T> Dimensioned for DynAxesRepr<T> {
    type Dimality = DDyn;
}

macro_rules! impl_op {
    ($op_trait:ty, $op_fn:ident, $op_assign_trait:ty, $op_assign_fn:ident) => {
        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_trait for DShape
        where
            Rhs: IntoShape<Dimality = DDyn>,
        {
            type Output = DShape;

            fn $op_fn(self, rhs: Rhs) -> <Self as $op_trait>::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_assign_trait for DShape
        where
            Rhs: IntoShape<Dimality = DDyn>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into_shape();
                for i in 0..self.ndim().max(other.ndim()) {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }

        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_trait for DStrides
        where
            Rhs: IntoStrides<Dimality = DDyn>,
        {
            type Output = DStrides;

            fn $op_fn(self, rhs: Rhs) -> <Self as $op_trait>::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        /// *Panics* if the two dimensionalities are different
        impl<Rhs> $op_assign_trait for DStrides
        where
            Rhs: IntoStrides<Dimality = DDyn>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into_strides();
                for i in 0..self.ndim().max(other.ndim()) {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }
    };
}

impl_op!(Add<Rhs>, add, AddAssign<Rhs>, add_assign);
impl_op!(Sub<Rhs>, sub, SubAssign<Rhs>, sub_assign);
impl_op!(Mul<Rhs>, mul, MulAssign<Rhs>, mul_assign);

impl Shape for DynAxesRepr<usize> {
    fn as_slice(&self) -> Cow<'_, [usize]> {
        Cow::Borrowed(self)
    }

    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, crate::ShapeStrideError<Self>> {
        if index < self.len() {
            Ok(&mut self[index])
        } else {
            Err(crate::ShapeStrideError::OutOfBounds(PhantomData, index))
        }
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn size_checked(&self) -> Option<usize> {
        self.iter().try_fold(1_usize, |acc, &i| acc.checked_mul(i))
    }

    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>>
    where
        Self: Sized,
    {
        if ndim <= CAP {
            Ok(Self::Inline(ndim, [value; CAP]))
        } else {
            let mut v = Vec::with_capacity(ndim);
            v.fill(value);
            Ok(Self::Alloc(v.into_boxed_slice()))
        }
    }
}

impl ShapeMut for DynAxesRepr<usize> {}

impl Strides for DynAxesRepr<isize> {
    fn as_slice(&self) -> Cow<'_, [isize]> {
        Cow::Borrowed(self)
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn is_c_order(&self) -> bool {
        todo!()
    }

    fn is_f_order(&self) -> bool {
        todo!()
    }
}

impl StridesMut for DynAxesRepr<isize> {}
