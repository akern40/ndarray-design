use alloc::borrow::Cow;
use core::marker::PhantomData;
use core::ops::Index;
use core::{fmt::Debug, ops::IndexMut};

use crate::{DStrides, Dimensioned, NStrides, ShapeStrideError};

use super::{dimensionality::Dimensionality, shape::IntoShape};

/// The trait for types that contain the strides of a multidimensional array.
///
/// Like elsewhere in the Rust core and ecosystem, strides in `ndarray` are measured
/// by the number of _elements_, not by the number of bytes.
///
/// Unlike [`Shape`](super::Shape), mixing `const`-valued strides with non-`const`
/// valued strides is not supported. Strides are either considered read-only (this trait)
/// or fully mutable ([`StridesMut`]).
pub trait Strides:
    Dimensioned + Index<usize, Output = isize> + Eq + Clone + Send + Sync + Debug
{
    /// Get the strides as a (possibly-borrowed) slice of `isize`.
    fn as_slice(&self) -> Cow<'_, [isize]>;

    /// Get the runtime dimensionality of the strides.
    ///
    /// Implementors of `Strides` must guarantee that this value will match [`Strides::Dimality`].
    fn ndim(&self) -> usize;

    fn is_c_order(&self) -> bool;

    fn is_f_order(&self) -> bool;

    fn to_dyn(&self) -> DStrides {
        self.as_slice().into()
    }

    /// Try to turn this shape into a constant `N`-dimensional shape.
    fn try_to_nstrides<const N: usize>(
        &self,
    ) -> Result<NStrides<N>, ShapeStrideError<NStrides<N>>> {
        if self.ndim() == N {
            let mut values = [0; N];
            values.split_at_mut(N).0.copy_from_slice(&self.as_slice());
            Ok(values.into())
        } else {
            Err(ShapeStrideError::BadDimality(PhantomData, self.ndim()))
        }
    }
}

/// Strides with no `const` values, i.e., all strides are mutable.
pub trait StridesMut: Strides + IndexMut<usize, Output = isize> {}

/// Default C-style (row-major) stride construction.
pub trait DefaultC: Strides {
    fn default_c<Sh>(shape: Sh) -> Self
    where
        Sh: IntoShape<Dimality = Self::Dimality>;
}

/// Default F-style (column-major) stride construction.
pub trait DefaultF: Strides {
    fn default_f<Sh>(shape: Sh) -> Self
    where
        Sh: IntoShape<Dimality = Self::Dimality>;
}

/// A conversion trait for types that can turn into strides.
pub trait IntoStrides {
    type Dimality: Dimensionality;

    type Strides: Strides<Dimality = Self::Dimality>;

    fn into_strides(&self) -> Self::Strides;
}

impl<T: Strides> IntoStrides for T {
    type Dimality = T::Dimality;

    type Strides = Self;

    fn into_strides(&self) -> Self::Strides {
        self.clone()
    }
}
