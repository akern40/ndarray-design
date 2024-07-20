//! ## Array Variants
//!
//! In this implementation, the array variants are not all the same generic type;
//! instead, each variant gets its own definition. This might seem cumbersome,
//! but it is critical to getting separation between "not-safely-derefencable" methods of
//! `RawNdArray` and the "safely-derefencable" methods of `NdArray`.
//!
//! I think that the separate structs implementation also makes it eminently clear what each variant
//! is doing differently. Finally, without a "third generic" (aside from element type and layout type),
//! we need different structs to indicate data mutability.
//!
//! You'll notice that the definitions are largely similar, and do not seem to inherently limit
//! the mutability or data dereference safety of their particular representations.
//! See [`crate::array_deref`] for how this is accomplished.

use crate::core::{ArrayRef, RawArrayRef};
use std::marker::PhantomData;

/// An owned array.
///
/// The representation here is slightly different from ArrayBase
/// in order to make it easier to implement Deref safely.
#[derive(Debug)]
pub struct Array<A, L> {
    pub(crate) meta: ArrayRef<A, L>,
    pub(crate) cap: usize,
    pub(crate) len: usize, // This may seem redundant, but we don't know what type `L` is;
                           // we won't even require it to be bound by Layout. As a result,
                           // we need to manually keep track of the "length" of elements in the array,
                           // even though this information is redundant with the layout in `ArrayRef`.
}

/// A view of an existing array.
#[derive(Debug)]
pub struct ArrayView<'a, A, L> {
    pub(crate) meta: ArrayRef<A, L>,
    pub(crate) life: PhantomData<&'a A>,
}

/// A mutable view of an existing array
#[derive(Debug)]
pub struct ArrayViewMut<'a, A, L> {
    pub(crate) meta: ArrayRef<A, L>,
    pub(crate) life: PhantomData<&'a mut A>,
}

/// A view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayView<A, L> {
    pub(crate) meta: RawArrayRef<A, L>,
    pub(crate) life: PhantomData<*const A>,
}

/// A mutable view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayViewMut<A, L> {
    pub(crate) meta: RawArrayRef<A, L>,
    pub(crate) life: PhantomData<*mut A>,
}
