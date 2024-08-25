//! ## Trait Implementations
//!
//! Now that the array types dereference to the appropriate reference types,
//! I can implement the traits we started with for those reference types.

// I'll start with NdLayout:

// use crate::core::{ArrayRefBase, Layout, NdArray, NdLayout, RawArrayRefBase, RawNdArray, Storage};

// impl<L: Layout, S> NdLayout<L> for RawArrayRefBase<L, S> {
//     fn len(&self) -> usize {
//         self.layout.size()
//     }
// }

// // Now for RawNdArray:

// impl<L: Layout, S: Storage> RawNdArray<L, S> for RawArrayRefBase<L, S> {
//     fn as_ptr(&self) -> *const S::Elem {
//         unsafe { self.storage.as_ptr() }
//     }

//     fn as_mut_ptr(&mut self) -> *mut S::Elem {
//         unsafe { self.storage.as_ptr() }
//     }
// }

// // // And finally NdArray

// impl<L: Layout, S: Storage> NdArray<L, S, RawArrayRefBase<L, S>> for ArrayRefBase<L, S> {}

use crate::core::{Backend, Layout};

use super::RawArrayRefBase;

impl<L: Layout, B: Backend> RawArrayRefBase<L, B> {
    pub fn len(&self) -> usize {
        self.layout.size()
    }
}
