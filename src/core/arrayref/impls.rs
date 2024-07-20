//! ## Trait Implementations
//!
//! Now that the array types dereference to the appropriate reference types,
//! I can implement the traits we started with for those reference types.

// I'll start with NdLayout:

use std::ptr::NonNull;

use crate::core::{ArrayRef, Layout, NdArray, NdLayout, RawArrayRef, RawNdArray};

impl<A, L: Layout> NdLayout<L> for RawArrayRef<A, L> {
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<A, L: Layout> NdLayout<L> for ArrayRef<A, L> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

// Now for RawNdArray:

impl<A, L: Layout> RawNdArray<A, L> for RawArrayRef<A, L> {
    fn ptr(&self) -> &NonNull<A> {
        &self.ptr
    }

    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        &mut self.ptr
    }
}

impl<A, L: Layout> RawNdArray<A, L> for ArrayRef<A, L> {
    fn ptr(&self) -> &NonNull<A> {
        self.0.ptr()
    }

    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        self.0.ptr_mut()
    }
}

// And finally NdArray

impl<A, L: Layout> NdArray<A, L> for ArrayRef<A, L> {}
