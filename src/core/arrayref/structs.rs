//! ## Array Reference Structs
//!
//! It may seem odd, but before I go on to defining array types themselves, I'm first going
//! define *references* to arrays. The point of this is that the references - not the arrays -
//! will hold most of the behavior of a multidimensional array.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

#[derive(Debug)]
pub struct RawArrayRefBase<A, L, P = NonNull<A>> {
    pub(crate) layout: L,
    pub(crate) ptr: P,
    phantom: PhantomData<A>,
}

#[derive(Debug)]
pub struct ArrayRefBase<A, L, P>(pub(crate) RawArrayRefBase<A, L, P>);

// Variations on the base array ref type
pub type RawArrayRef<A, L> = RawArrayRefBase<A, L, NonNull<A>>;
pub type ArrayRef<A, L> = ArrayRefBase<A, L, NonNull<A>>;
pub type RawArcArrayRef<A, L> = RawArrayRefBase<A, L, Arc<A>>;
pub type ArcArrayRef<A, L> = ArrayRefBase<A, L, Arc<A>>;

/// Now to link these two: I'm going to implement `Deref` and `DerefMut` from an ArrayRef
/// to its inner `RawArrayRef`.

impl<A, L, P> Deref for ArrayRefBase<A, L, P> {
    type Target = RawArrayRefBase<A, L, P>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, L, P> DerefMut for ArrayRefBase<A, L, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
