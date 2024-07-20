pub use array_refs::*;
pub use arrays::*;
pub use trait_defs::*;

mod test_functions {
    //! ## Test Functions
    //!
    //! This section is supposed to showcase what the new design can do.
    //! I put them at the top so that it's clear what I'm trying to accomplish, and check whether it compiles.

    use core::fmt::Debug;

    use crate::{
        Array, ArrayRef, ArrayView, ArrayViewMut, Layout, NdArray, RawArrayRef, RawArrayView,
        RawArrayViewMut, RawNdArray,
    };

    fn ergonomic_raw<A, L: Layout>(arr: &RawArrayRef<A, L>) {
        println!("{:?}", arr.ptr());
    }
    fn ergonomic<A: Debug, L: Layout>(arr: &ArrayRef<A, L>) {
        println!("{:?}", arr.first());
    }
    fn ergonomic_raw_mut<A, L: Layout>(arr: &mut RawArrayRef<A, L>) {
        println!("{:?}", arr.ptr_mut());
    }
    fn ergonomic_mut<A: Debug, L: Layout>(arr: &mut ArrayRef<A, L>) {
        println!("{:?}", arr.first_mut());
    }

    /// Scaffolding to call the above functions; arguments are move to simulate fully-owned values.
    fn caller<A, L: Layout>(
        mut arr: Array<A, L>,
        arr_view: ArrayView<A, L>,
        mut arr_view_mut: ArrayViewMut<A, L>,
        raw_view: RawArrayView<A, L>,
        mut raw_view_mut: RawArrayViewMut<A, L>,
    ) {
        ergonomic_raw(&arr);
        ergonomic(&arr);
        ergonomic_raw_mut(&mut arr);
        ergonomic_mut(&mut arr);

        ergonomic_raw(&arr_view);
        ergonomic(&arr_view);
        // Fails to compile because you can't get a mutable reference
        ergonomic_raw_mut(&mut arr_view);
        ergonomic_mut(&mut arr_view);

        ergonomic_raw(&arr_view_mut);
        ergonomic(&arr_view_mut);
        ergonomic_raw_mut(&mut arr_view_mut);
        ergonomic_mut(&mut arr_view_mut);

        ergonomic_raw(&raw_view);
        // Fails to compile because you can't get an ArrayRef from a RawArrayView
        ergonomic(&raw_view);
        // Fails to compile because you can't get a mutable raw ref from a RawArrayView
        ergonomic_raw_mut(&mut raw_view);
        // Fails to compile because you can't get an mutable ArrayRef from a RawArrayView
        ergonomic_mut(&mut raw_view);

        ergonomic_raw(&raw_view_mut);
        // Fails to compile because you can't get an ArrayRef from a RawArrayView
        ergonomic(&raw_view_mut);
        ergonomic_raw_mut(&mut raw_view_mut);
        // Fails to compile because you can't get a mutable ArrayRef from a RawArrayView
        ergonomic_mut(&mut raw_view_mut);
    }
}

mod trait_defs {
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

    /// A trait for the data that carries an array's layout information.
    ///
    /// Arrays (and array references) should not implement this trait; it is meant to be implemented
    /// by the struct *inside* the array (reference) that actually holds the layout information.
    pub trait Layout {
        fn size(&self) -> usize;
    }

    /// A trait for shape- and stride- related functions.
    pub trait NdLayout<L> {
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
}

mod array_refs {
    //! ## Array Reference Structs
    //!
    //! It may seem odd, but before I go on to defining array types themselves, I'm first going
    //! define *references* to arrays. The point of this is that the references - not the arrays -
    //! will hold most of the behavior of a multidimensional array.

    use std::{
        ops::{Deref, DerefMut},
        ptr::NonNull,
    };

    /// A reference to an array whose elements may not be safe to dereference.
    #[derive(Debug)]
    pub struct RawArrayRef<A, L> {
        pub(crate) layout: L,
        pub(crate) ptr: NonNull<A>,
    }

    /// A reference to an array whose elements are safe to dereference.
    #[derive(Debug)]
    pub struct ArrayRef<A, L>(pub(crate) RawArrayRef<A, L>);

    /// Now to link these two: I'm going to implement `Deref` and `DerefMut` from an ArrayRef
    /// to its inner `RawArrayRef`.

    impl<A, L> Deref for ArrayRef<A, L> {
        type Target = RawArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<A, L> DerefMut for ArrayRef<A, L> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
}

mod arrays {
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

    use std::marker::PhantomData;

    use crate::{ArrayRef, RawArrayRef};

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
}

mod array_deref {
    //! ## Array Dereferencing
    //!
    //! So, with all that setup in place, I can move on to the real trick: `Deref` and `DerefMut`
    //! implementations for the array types themselves. The idea here is that arrays really get
    //! their mutability and data dereference safety from which `Deref` targets they have.
    //! For example, a RawArrayView derefs to a `RawArrayRef`, while an `ArrayView` derefs to
    //! an `ArrayRef`, and neither of them implement `DerefMut` (but `ArrayViewMut` does).
    //! The orphan rule will prohibit users from breaking this safety design with their own `impl`s.

    use std::ops::{Deref, DerefMut};

    use crate::{
        Array, ArrayRef, ArrayView, ArrayViewMut, RawArrayRef, RawArrayView, RawArrayViewMut,
    };

    impl<A, L> Deref for Array<A, L> {
        type Target = ArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.meta
        }
    }

    impl<A, L> DerefMut for Array<A, L> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.meta
        }
    }

    impl<'a, A, L> Deref for ArrayView<'a, A, L> {
        type Target = ArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.meta
        }
    }

    impl<'a, A, L> Deref for ArrayViewMut<'a, A, L> {
        type Target = ArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.meta
        }
    }

    impl<'a, A, L> DerefMut for ArrayViewMut<'a, A, L> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.meta
        }
    }

    impl<A, L> Deref for RawArrayView<A, L> {
        type Target = RawArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.meta
        }
    }

    impl<'a, A, L> Deref for RawArrayViewMut<A, L> {
        type Target = RawArrayRef<A, L>;

        fn deref(&self) -> &Self::Target {
            &self.meta
        }
    }

    impl<'a, A, L> DerefMut for RawArrayViewMut<A, L> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.meta
        }
    }
}

mod trait_impl {
    //! ## Trait Implementations
    //!
    //! Now that the array types dereference to the appropriate reference types,
    //! I can implement the traits we started with for those reference types.

    // I'll start with NdLayout:

    use std::ptr::NonNull;

    use crate::{ArrayRef, Layout, NdArray, NdLayout, RawArrayRef, RawNdArray};

    impl<A, L: Layout> NdLayout<L> for RawArrayRef<A, L> {
        fn len(&self) -> usize {
            self.layout.size()
        }
    }

    impl<A, L: Layout> NdLayout<L> for ArrayRef<A, L> {
        fn len(&self) -> usize {
            self.len()
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

        fn try_ensure_unique(&mut self) {
            todo!()
        }

        fn try_is_unique(&mut self) -> Option<bool> {
            None
        }
    }

    impl<A, L: Layout> RawNdArray<A, L> for ArrayRef<A, L> {
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

    // And finally NdArray

    impl<A, L: Layout> NdArray<A, L> for ArrayRef<A, L> {}
}
