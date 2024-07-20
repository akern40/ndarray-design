//! ## Array Dereferencing
//!
//! So, with all that setup in place, I can move on to the real trick: `Deref` and `DerefMut`
//! implementations for the array types themselves. The idea here is that arrays really get
//! their mutability and data dereference safety from which `Deref` targets they have.
//! For example, a RawArrayView derefs to a `RawArrayRef`, while an `ArrayView` derefs to
//! an `ArrayRef`, and neither of them implement `DerefMut` (but `ArrayViewMut` does).
//! The orphan rule will prohibit users from breaking this safety design with their own `impl`s.

use std::ops::{Deref, DerefMut};

use crate::core::{
    Array, ArrayRef, ArrayView, ArrayViewMut, RawArrayRef, RawArrayView, RawArrayViewMut,
};

impl<A, L> Deref for Array<A, L> {
    type Target = ArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<A, L> DerefMut for Array<A, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<'a, A, L> Deref for ArrayView<'a, A, L> {
    type Target = ArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, L> Deref for ArrayViewMut<'a, A, L> {
    type Target = ArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, L> DerefMut for ArrayViewMut<'a, A, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<A, L> Deref for RawArrayView<A, L> {
    type Target = RawArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, L> Deref for RawArrayViewMut<A, L> {
    type Target = RawArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, L> DerefMut for RawArrayViewMut<A, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}
