//! ## Array Reference Structs
//!
//! It may seem odd, but before I go on to defining array types themselves, I'm first going
//! define *references* to arrays. The point of this is that the references - not the arrays -
//! will hold most of the behavior of a multidimensional array.

use std::ops::{Deref, DerefMut};

use crate::core::{storage::NonNullStorage, Backend};

/// A reference to an array whose elements may not be safe to dereference.
#[derive(Debug)]
pub struct RawArrayRefBase<L, B: Backend> {
    pub(crate) layout: L,
    pub(crate) storage: B::Ref,
}

/// A reference to an array whose elements are safe to dereference.
#[derive(Debug)]
pub struct ArrayRefBase<L, B: Backend>(pub(crate) RawArrayRefBase<L, B>);

pub type RawArrayRef<A, L> = RawArrayRefBase<L, NonNullStorage<A>>;
pub type ArrayRef<A, L> = ArrayRefBase<L, NonNullStorage<A>>;

/// Now to link these two: I'm going to implement `Deref` and `DerefMut` from an ArrayRef
/// to its inner `RawArrayRef`.

impl<L, B: Backend> Deref for ArrayRefBase<L, B> {
    type Target = RawArrayRefBase<L, B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<L, B: Backend> DerefMut for ArrayRefBase<L, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
