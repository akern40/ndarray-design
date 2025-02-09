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
    ArrayRefBase, ArrayViewBase, ArrayViewBaseMut, Backend, RawArrayRefBase, RawArrayViewBase,
    RawArrayViewBaseMut,
};

use super::ArrayBase;

impl<L, B: Backend> Deref for ArrayBase<L, B> {
    type Target = ArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<L, B: Backend> DerefMut for ArrayBase<L, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.aref
    }
}

impl<'a, L, B: Backend> Deref for ArrayViewBase<'a, L, B> {
    type Target = ArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<'a, L, B: Backend> Deref for ArrayViewBaseMut<'a, L, B> {
    type Target = ArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<'a, L, B: Backend> DerefMut for ArrayViewBaseMut<'a, L, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.aref
    }
}

impl<L, B: Backend> Deref for RawArrayViewBase<L, B> {
    type Target = RawArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<'a, L, B: Backend> Deref for RawArrayViewBaseMut<L, B> {
    type Target = RawArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<'a, L, B: Backend> DerefMut for RawArrayViewBaseMut<L, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.aref
    }
}
