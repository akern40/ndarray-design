// Some setup
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use std::sync::Arc;

/// The existing dimension trait, just what we need for an example
trait Dimension {
    fn size(&self) -> usize;
}

////////////////////
// ARRAY VARIANTS //
////////////////////
// In this implementation, the array variants are not all the same generic type;
// instead, each variant gets its own definition. This might seem cumbersome,
// but it brings several advantages.

/// An owned array.
// The representation here is slightly different from ArrayBase
// in order to make it easier to implement Deref safely.
#[derive(Debug)]
pub struct Array<A, D> {
    meta: ArrayRef<A, D>,
    cap: usize,
}

/// A view of an existing array.
#[derive(Debug)]
pub struct ArrayView<'a, A, D> {
    meta: ArrayRef<A, D>,
    life: PhantomData<&'a A>,
}

/// A mutable view of an existing array
#[derive(Debug)]
pub struct ArrayViewMut<'a, A, D> {
    meta: ArrayRef<A, D>,
    life: PhantomData<&'a mut A>,
}

/// A view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayView<A, D> {
    meta: RawArrayRef<A, D>,
    life: PhantomData<*const A>,
}

/// A mutable view of an array without a lifetime, and whose elements are not safe to dereference.
#[derive(Debug)]
pub struct RawArrayViewMut<A, D> {
    meta: RawArrayRef<A, D>,
    life: PhantomData<*mut A>,
}

#[derive(Debug)]
pub struct ArcArray<A, D> {
    meta: ArcArrayRef<A, D>,
    cap: usize,
}

/// A reference to an array whose elements are safe to dereference.
#[derive(Debug)]
pub struct ArrayRef<A, D>(RawArrayRef<A, D>);

/// A reference to an array whose elements may not be safe to dereference.
#[derive(Debug)]
pub struct RawArrayRef<A, D> {
    dim: D,
    strides: D,
    ptr: NonNull<A>,
}

/// A reference to an array whose elements are safe to dereference.
#[derive(Debug)]
pub struct ArcArrayRef<A, D>(RawArcArrayRef<A, D>);

/// A reference to an array whose elements may not be safe to dereference.
#[derive(Debug)]
pub struct RawArcArrayRef<A, D> {
    dim: D,
    strides: D,
    ptr: Arc<NonNull<A>>,
}

// Define a new struct that holds references to the data
struct RawArrayRefRef<'a, A, D> {
    dim: &'a D,
    strides: &'a D,
    ptr: NonNull<A>,
}

impl<A, D> Deref for Array<A, D> {
    type Target = ArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> Deref for ArrayView<'a, A, D> {
    type Target = ArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> Deref for ArrayViewMut<'a, A, D> {
    type Target = ArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> DerefMut for ArrayViewMut<'a, A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<A, D> Deref for RawArrayView<A, D> {
    type Target = RawArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> Deref for RawArrayViewMut<A, D> {
    type Target = RawArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> DerefMut for RawArrayViewMut<A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<'a, A, D> Deref for ArcArray<A, D> {
    type Target = ArcArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<'a, A, D> DerefMut for ArcArray<A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<A, D> Deref for ArrayRef<A, D> {
    type Target = RawArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, D> DerefMut for ArrayRef<A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<A, D> Deref for ArcArrayRef<A, D> {
    type Target = RawArcArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, D> DerefMut for ArcArrayRef<A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

trait Layout<D> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<A, D: Dimension> Layout<D> for RawArrayRef<A, D> {
    fn len(&self) -> usize {
        self.dim.size()
    }
}

impl<A, D: Dimension> Layout<D> for ArrayRef<A, D> {
    fn len(&self) -> usize {
        self.0.dim.size()
    }
}

impl<A, D: Dimension> Layout<D> for RawArcArrayRef<A, D> {
    fn len(&self) -> usize {
        self.dim.size()
    }
}

trait RawNdArrayRef<A, D>: Layout<D> {
    fn ptr(&self) -> &NonNull<A>;

    fn as_ptr(&self) -> *const A {
        self.ptr().as_ptr() as *const A
    }
}

trait RawNdArrayRefMut<A, D>: RawNdArrayRef<A, D> {
    fn ptr_mut(&mut self) -> &mut NonNull<A>;

    fn try_ensure_unique(&mut self);

    fn try_is_unique(&mut self) -> Option<bool>;

    fn as_mut_ptr(&mut self) -> *mut A {
        self.try_ensure_unique();
        self.ptr().as_ptr()
    }
}

trait NdArrayRef<A, D>: RawNdArrayRef<A, D> {
    fn first(&self) -> Option<&A> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &*self.as_ptr() })
        }
    }
}

trait NdArrayRefMut<A, D>: NdArrayRef<A, D> + RawNdArrayRefMut<A, D> {
    fn first_mut(&mut self) -> Option<&mut A> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &mut *self.as_mut_ptr() })
        }
    }
}

impl<A, D: Dimension> RawNdArrayRef<A, D> for RawArrayRef<A, D> {
    fn ptr(&self) -> &NonNull<A> {
        &self.ptr
    }
}

impl<A, D: Dimension> RawNdArrayRefMut<A, D> for RawArrayRef<A, D> {
    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        &mut self.ptr
    }

    fn try_ensure_unique(&mut self) {}

    fn try_is_unique(&mut self) -> Option<bool> {
        None
    }
}

impl<A, D: Dimension> RawNdArrayRef<A, D> for RawArcArrayRef<A, D> {
    fn ptr(&self) -> &NonNull<A> {
        &self.ptr
    }
}

impl<A, D: Dimension> RawNdArrayRef<A, D> for ArrayRef<A, D> {
    fn ptr(&self) -> &NonNull<A> {
        &self.0.ptr
    }
}

impl<A, D: Dimension> RawNdArrayRefMut<A, D> for ArrayRef<A, D> {
    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        &mut self.0.ptr
    }

    fn try_ensure_unique(&mut self) {}

    fn try_is_unique(&mut self) -> Option<bool> {
        None
    }
}

impl<A, D: Dimension> RawNdArrayRefMut<A, D> for RawArcArrayRef<A, D> {
    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        self.try_ensure_unique();
        Arc::make_mut(&mut self.ptr)
    }

    fn try_ensure_unique(&mut self) {
        // All the stuff from OwnedArcRepr
    }

    fn try_is_unique(&mut self) -> Option<bool> {
        Some(Arc::get_mut(&mut self.ptr).is_some())
    }
}

fn ergonomic_raw<A, D>(arr: &RawArrayRef<A, D>) {}
fn ergonomic<A, D>(arr: &ArrayRef<A, D>) {}
fn powerful<A, D, T: RawNdArrayRef<A, D>>(arr: &T) {}

fn caller<A, D: Dimension>(arr: Array<A, D>, arc: ArcArray<A, D>) {
    ergonomic_raw(&arr);
    ergonomic(&arr);
    powerful(arr.deref());

    powerful(arc.deref().deref());
}
