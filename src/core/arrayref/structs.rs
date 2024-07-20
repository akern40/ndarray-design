//! ## Array Reference Structs
//!
//! It may seem odd, but before I go on to defining array types themselves, I'm first going
//! define *references* to arrays. The point of this is that the references - not the arrays -
//! will hold most of the behavior of a multidimensional array.

use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// A reference to an array whose elements may not be safe to dereference.
#[derive(Debug)]
pub struct RawArrayRef<A, L> {
    pub(crate) layout: L,
    pub(crate) ptr: NonNull<A>,
}

/// A reference to an array whose elements are safe to dereference.
#[derive(Debug)]
pub struct ArrayRef<A, L>(pub(crate) RawArrayRef<A, L>);

/// Now to link these two: I'm going to implement `Deref` and `DerefMut` from an ArrayRef
/// to its inner `RawArrayRef`.

impl<A, L> Deref for ArrayRef<A, L> {
    type Target = RawArrayRef<A, L>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, L> DerefMut for ArrayRef<A, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
