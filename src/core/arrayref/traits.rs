//! ## Trait Definitions
//!
//! This design uses a trait-based approach. I break up the concept of a multidimensional
//! array into three behaviors:
//!
//! 1. functions relating to its layout (which do not depend on or alter data),
//! 2. functions that can be done on any multidimensional array, even if its data is not safe to dereference
//! 3. functions that can only be done (safely) if the data is safe to dereference
//!
//! I give each of those traits: NdLayout, RawNdArray, and NdArray. They probably need better names.
//! I also require these three traits to build on each other; I think this is justifiable as it creates
//! a sort of "hierarchy" of expected behavior.
//! You have things that "look" like they have multidimensional shapes,
//! things that are unsafely multidimensional arrays,
//! and things that are safely multidimensional arrays.
//!
//! I'll also add a trait called Layout that acts similar to the existing Dimension trait in ndarray.

use std::ptr::NonNull;

/// A trait for shape- and stride- related functions.
pub trait NdLayout<L> {
    /// Return the total number of elements in the array.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A trait for functions that can operate safely on array data that may not be dereferencable.
///
/// To implement this trait, the functions `ptr` and `ptr_mut` must return references to the
/// NonNull that points to the "head" of that array's data. The implementation must also provide
/// methods for attempting to check or ensure uniqueness.
pub trait RawNdArray<A, L>: NdLayout<L> {
    fn ptr(&self) -> &NonNull<A>;

    fn ptr_mut(&mut self) -> &mut NonNull<A>;

    fn as_mut_ptr(&mut self) -> *mut A {
        self.ptr().as_ptr()
    }

    fn as_ptr(&self) -> *const A {
        self.ptr().as_ptr() as *const A
    }
}

/// A trait for functions that can only operate safely on array data that is safely dereferencable.
pub trait NdArray<A, L>: RawNdArray<A, L> {
    fn first(&self) -> Option<&A> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &*self.as_ptr() })
        }
    }

    fn first_mut(&mut self) -> Option<&mut A> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &mut *self.as_mut_ptr() })
        }
    }
}