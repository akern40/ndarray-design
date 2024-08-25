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

use crate::core::{
    storage::NonNullStorage, ArcBackend, ArrayRefBase, Backend, RawArrayRefBase, VecBackend,
};
use std::marker::PhantomData;

/// An owned array.
///
/// This representation is meant to encompass a whole variety of use cases.
/// As such, it is generic in three types:
///     1. `L`, the layout, representing the kind of shape it has
///     2. `S`, the storage, representing the underlying pointer
///     3. `O`, the ownership, representing any additional information needed for memory management.
#[derive(Debug)]
pub struct ArrayBase<L, B: Backend> {
    pub(crate) aref: ArrayRefBase<L, B>,
    pub(crate) own: B::Owned,
}

pub type Array<A, L> = ArrayBase<L, VecBackend<A>>;
pub type ArcArray<A, L> = ArrayBase<L, ArcBackend<A>>;

/// A view of an existing array.
#[derive(Debug)]
pub struct ArrayViewBase<'a, L, B: Backend> {
    pub(crate) aref: ArrayRefBase<L, B>,
    pub(crate) life: PhantomData<&'a B::Elem>,
}

/// A mutable view of an existing array
#[derive(Debug)]
pub struct ArrayViewBaseMut<'a, L, B: Backend> {
    pub(crate) aref: ArrayRefBase<L, B>,
    pub(crate) life: PhantomData<&'a mut B::Elem>,
}

/// A view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayViewBase<L, B: Backend> {
    pub(crate) aref: RawArrayRefBase<L, B>,
    pub(crate) life: PhantomData<*const B::Elem>,
}

/// A mutable view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayViewBaseMut<L, B: Backend> {
    pub(crate) aref: RawArrayRefBase<L, B>,
    pub(crate) life: PhantomData<*mut B::Elem>,
}
