mod core;

mod test_functions {
    //! ## Test Functions
    //!
    //! This section is supposed to showcase what the new design can do.
    //! I put them at the top so that it's clear what I'm trying to accomplish, and check whether it compiles.

    use core::fmt::Debug;

    use crate::core::{
        Array, ArrayRef, ArrayView, ArrayViewBase, ArrayViewBaseMut, ArrayViewMut, Layout, NdArray,
        RawArrayRef, RawArrayView, RawArrayViewBase, RawArrayViewBaseMut, RawArrayViewMut,
        RawNdArray, Storage,
    };

    fn ergonomic_raw<A, L: Layout>(arr: &RawArrayRef<A, L>) {
        println!("{:?}", arr.as_ptr());
    }
    fn ergonomic<A: Debug, L: Layout>(arr: &ArrayRef<A, L>) {
        println!("{:?}", arr.first());
    }
    fn ergonomic_raw_mut<A, L: Layout>(arr: &mut RawArrayRef<A, L>) {
        println!("{:?}", arr.as_mut_ptr());
    }
    fn ergonomic_mut<A: Debug, L: Layout>(arr: &mut ArrayRef<A, L>) {
        println!("{:?}", arr.first_mut());
    }

    /// Scaffolding to call the above functions; arguments are move to simulate fully-owned values.
    fn caller<A: Debug, L: Layout>(
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
        // ergonomic_raw_mut(&mut arr_view);
        // ergonomic_mut(&mut arr_view);

        ergonomic_raw(&arr_view_mut);
        ergonomic(&arr_view_mut);
        ergonomic_raw_mut(&mut arr_view_mut);
        ergonomic_mut(&mut arr_view_mut);

        ergonomic_raw(&raw_view);
        // Fails to compile because you can't get an ArrayRef from a RawArrayView
        // ergonomic(&raw_view);
        // Fails to compile because you can't get a mutable raw ref from a RawArrayView
        // ergonomic_raw_mut(&mut raw_view);
        // Fails to compile because you can't get an mutable ArrayRef from a RawArrayView
        // ergonomic_mut(&mut raw_view);

        ergonomic_raw(&raw_view_mut);
        // Fails to compile because you can't get an ArrayRef from a RawArrayView
        // ergonomic(&raw_view_mut);
        ergonomic_raw_mut(&mut raw_view_mut);
        // Fails to compile because you can't get a mutable ArrayRef from a RawArrayView
        // ergonomic_mut(&mut raw_view_mut);
    }
}
