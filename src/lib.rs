/// Some setup
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::sync::Arc;

//////////////////////
/// TEST FUNCTIONS ///
//////////////////////
/// This section is supposed to showcase what the new design can do.
/// I put them at the top so that it's clear what I'm trying to accomplish, and check whether it compiles.

fn ergonomic_raw<A, D>(arr: &RawArrayRef<A, D>) {}
fn ergonomic<A, D>(arr: &ArrayRef<A, D>) {}
fn powerful<A, D, T: RawNdArray<A, D>>(arr: &T) {}

fn caller<A, D: Layout>(arr: Array<A, D>) {
    ergonomic_raw(&arr);
    ergonomic(&arr);
    powerful(arr.deref());
}

/////////////////////////
/// TRAIT DEFINITIONS ///
/////////////////////////
/// This design uses a trait-based approach. I break up the concept of a multidimensional
/// array into three behaviors:
///
/// 1. functions relating to its layout (which do not depend on or alter data),
/// 2. functions that can be done on any multidimensional array, even if its data is not safe to dereference
/// 3. functions that can only be done (safely) if the data is safe to dereference
///
/// I give each of those traits: NdLayout, RawNdArray, and NdArray. They probably need better names.
/// I also require these three traits to build on each other; I think this is justifiable as it creates
/// a sort of "hierarchy" of expected behavior.
/// You have things that "look" like they have multidimensional shapes,
/// things that are unsafely multidimensional arrays,
/// and things that are safely multidimensional arrays.
///
/// I'll also add a trait called Layout that acts similar to the existing Dimension trait in ndarray.

/// A trait for the data that carries an array's layout information.
///
/// Arrays (and array references) should not implement this trait; it is meant to be implemented
/// by the struct *inside* the array (reference) that actually holds the layout information.
trait Layout {
    fn size(&self) -> usize;
}

/// A trait for shape- and stride- related functions.
trait NdLayout<D> {
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
trait RawNdArray<A, D>: NdLayout<D> {
    fn ptr(&self) -> &NonNull<A>;

    fn ptr_mut(&mut self) -> &mut NonNull<A>;

    fn try_ensure_unique(&mut self);

    fn try_is_unique(&mut self) -> Option<bool>;

    fn as_mut_ptr(&mut self) -> *mut A {
        self.try_ensure_unique();
        self.ptr().as_ptr()
    }

    fn as_ptr(&self) -> *const A {
        self.ptr().as_ptr() as *const A
    }
}

/// A trait for functions that can only operate safely on array data that is safely dereferencable.
trait NdArray<A, D>: RawNdArray<A, D> {
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

///////////////////////////////
/// ARRAY REFERENCE STRUCTS ///
///////////////////////////////
/// It may seem odd, but before I go on to defining array types themselves, I'm first going
/// define *references* to arrays. The point of this is that the references - not the arrays -
/// will hold most of the behavior of a multidimensional array.

/// A reference to an array whose elements may not be safe to dereference.
#[derive(Debug)]
pub struct RawArrayRef<A, D> {
    layout: D,
    ptr: NonNull<A>,
}

/// A reference to an array whose elements are safe to dereference.
#[derive(Debug)]
pub struct ArrayRef<A, D>(RawArrayRef<A, D>);

/// Now to link these two: I'm going to implement `Deref` and `DerefMut` from an ArrayRef
/// to its inner `RawArrayRef`.

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

//////////////////////
/// ARRAY VARIANTS ///
//////////////////////
/// In this implementation, the array variants are not all the same generic type;
/// instead, each variant gets its own definition. This might seem cumbersome,
/// but it is critical to getting separation between "not-safely-derefencable" methods of
/// `RawNdArray` and the "safely-derefencable" methods of `NdArray`.
///
/// I think that the separate structs implementation also makes it eminently clear what each variant
/// is doing differently. Finally, without a "third generic" (aside from element type and layout type),
/// we need different structs to indicate data mutability.
///
/// You'll notice that the definitions are largely similar, and do not seem to inherently limit
/// the mutability or data dereference safety of their particular representations.
/// See ARRAY DEREFERENCING for how this is accomplished.

/// An owned array.
///
/// The representation here is slightly different from ArrayBase
/// in order to make it easier to implement Deref safely.
#[derive(Debug)]
pub struct Array<A, D> {
    meta: ArrayRef<A, D>,
    cap: usize,
    len: usize, // This may seem redundant, but we don't know what type `D` is;
                // we won't even require it to be bound by Dimension. As a result,
                // we need to manually keep track of the "length" of elements in the array,
                // even though this information is redundant with the layout in `ArrayRef`.
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

///////////////////////////
/// ARRAY DEREFERENCING ///
///////////////////////////
/// So, with all that setup in place, I can move on to the real trick: `Deref` and `DerefMut`
/// implementations for the array types themselves. The idea here is that arrays really get
/// their mutability and data dereference safety from which `Deref` targets they have.
/// For example, a RawArrayView derefs to a `RawArrayRef`, while an `ArrayView` derefs to
/// an `ArrayRef`, and neither of them implement `DerefMut` (but `ArrayViewMut` does).
/// The orphan rule will prohibit users from breaking this safety design with their own `impl`s.

impl<A, D> Deref for Array<A, D> {
    type Target = ArrayRef<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<A, D> DerefMut for Array<A, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
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

/////////////////////////////
/// TRAIT IMPLEMENTATIONS ///
/////////////////////////////
/// Now that the array types dereference to the appropriate reference types,
/// I can implement the traits we started with for those reference types.
///
/// I'll start with NdLayout:

impl<A, D: Layout> NdLayout<D> for RawArrayRef<A, D> {
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<A, D: Layout> NdLayout<D> for ArrayRef<A, D> {
    fn len(&self) -> usize {
        self.len()
    }
}

/// Now for RawNdArray:

impl<A, D: Layout> RawNdArray<A, D> for RawArrayRef<A, D> {
    fn ptr(&self) -> &NonNull<A> {
        &self.ptr
    }

    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        &mut self.ptr
    }

    fn try_ensure_unique(&mut self) {
        todo!()
    }

    fn try_is_unique(&mut self) -> Option<bool> {
        None
    }
}

impl<A, D: Layout> RawNdArray<A, D> for ArrayRef<A, D> {
    fn ptr(&self) -> &NonNull<A> {
        self.ptr()
    }

    fn ptr_mut(&mut self) -> &mut NonNull<A> {
        self.ptr_mut()
    }

    fn try_ensure_unique(&mut self) {
        self.try_ensure_unique()
    }

    fn try_is_unique(&mut self) -> Option<bool> {
        self.try_is_unique()
    }
}

// impl<A, D: Layout> NdLayout<D> for RawArcArrayRef<A, D> {
//     fn len(&self) -> usize {
//         self.dim.size()
//     }
// }

// #[derive(Debug)]
// pub struct ArcArray<A, D> {
//     meta: ArcArrayRef<A, D>,
//     cap: usize,
// }

// /// A reference to an array whose elements are safe to dereference.
// #[derive(Debug)]
// pub struct ArcArrayRef<A, D>(RawArcArrayRef<A, D>);

// /// A reference to an array whose elements may not be safe to dereference.
// #[derive(Debug)]
// pub struct RawArcArrayRef<A, D> {
//     dim: D,
//     strides: D,
//     ptr: Arc<NonNull<A>>,
// }

// // Define a new struct that holds references to the data
// struct RawArrayRefRef<'a, A, D> {
//     dim: &'a D,
//     strides: &'a D,
//     ptr: NonNull<A>,
// }

// impl<'a, A, D> Deref for ArcArray<A, D> {
//     type Target = ArcArrayRef<A, D>;

//     fn deref(&self) -> &Self::Target {
//         &self.meta
//     }
// }

// impl<'a, A, D> DerefMut for ArcArray<A, D> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.meta
//     }
// }

// impl<A, D> Deref for ArcArrayRef<A, D> {
//     type Target = RawArcArrayRef<A, D>;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// impl<A, D> DerefMut for ArcArrayRef<A, D> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }

// impl<A, D: NdLayout> RawArrayable<A, D> for RawArcArrayRef<A, D> {
//     fn ptr(&self) -> &NonNull<A> {
//         &self.ptr
//     }
// }

// impl<A, D: NdLayout> RawNdArrayRefMut<A, D> for RawArcArrayRef<A, D> {
//     fn ptr_mut(&mut self) -> &mut NonNull<A> {
//         self.try_ensure_unique();
//         Arc::make_mut(&mut self.ptr)
//     }

//     fn try_ensure_unique(&mut self) {
//         // All the stuff from OwnedArcRepr
//     }

//     fn try_is_unique(&mut self) -> Option<bool> {
//         Some(Arc::get_mut(&mut self.ptr).is_some())
//     }
// }
