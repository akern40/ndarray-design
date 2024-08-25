//! Storage types

use std::ptr::NonNull;

use super::{PointerStorage, Storage};

pub struct NonNullStorage<T> {
    ptr: NonNull<T>,
}

impl<T> Storage for NonNullStorage<T> {
    type Elem = T;
}

impl<T> PointerStorage for NonNullStorage<T> {
    unsafe fn ref_from_offset(&self, offset: usize) -> &T {
        self.ptr.add(offset).as_ref()
    }

    unsafe fn ref_mut_from_offset(&mut self, offset: usize) -> &mut T {
        self.ptr.add(offset).as_mut()
    }

    unsafe fn as_ptr(&self) -> *mut Self::Elem {
        self.ptr.as_ptr()
    }
}
