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

use crate::core::{ArrayRef, ArrayRefBase, RawArrayRef};
use std::{marker::PhantomData, ptr::NonNull, sync::Arc};

#[derive(Debug)]
pub struct ArrayBase<A, L, P = NonNull<A>> {
    pub(crate) meta: ArrayRefBase<A, L, P>,
    pub(crate) cap: usize,
    pub(crate) len: usize, // This may seem redundant, but we don't know what type `L` is;
                           // we won't even require it to be bound by Layout. As a result,
                           // we need to manually keep track of the "length" of elements in the array,
                           // even though this information is redundant with the layout in `ArrayRef`.
}

type Array<A, L> = ArrayBase<A, L>;
type ArcArray<A, L> = ArrayBase<A, L, Arc<A>>;

/// A view of an existing array.
#[derive(Debug)]
pub struct ArrayViewBase<'a, A, L, P = NonNull<A>> {
    pub(crate) meta: ArrayRefBase<A, L, P>,
    pub(crate) life: PhantomData<&'a A>,
}

/// A mutable view of an existing array
#[derive(Debug)]
pub struct ArrayViewBaseMut<'a, A, L, P = NonNull<A>> {
    pub(crate) meta: ArrayRefBase<A, L, P>,
    pub(crate) life: PhantomData<&'a mut A>,
}

type ArrayView<'a, A, L> = ArrayViewBase<'a, A, L>;
type ArrayViewMut<'a, A, L> = ArrayViewBaseMut<'a, A, L>;
type ArcArrayView<'a, A, L> = ArrayViewBase<'a, A, L, Arc<A>>;
type ArcArrayViewMut<'a, A, L> = ArrayViewBaseMut<'a, A, L, Arc<A>>;

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
