pub trait Layout {
    /// For fixed-size dimension representations (e.g. `Ix2`), this should be
    /// `Some(ndim)`, and for variable-size dimension representations (e.g.
    /// `IxDyn`), this should be `None`.
    const NDIM: Option<usize>;

    /// Returns the number of dimensions (number of axes).
    fn ndim(&self) -> usize;

    /// Compute the size of the dimension (number of elements)
    fn size(&self) -> usize {
        self.slice().iter().product()
    }

    /// Compute the size while checking for overflow.
    fn size_checked(&self) -> Option<usize> {
        self.slice()
            .iter()
            .try_fold(1_usize, |s, &a| s.checked_mul(a))
    }

    #[doc(hidden)]
    fn slice(&self) -> &[usize];

    #[doc(hidden)]
    fn slice_mut(&mut self) -> &mut [usize];

    // /// Borrow as a read-only array view.
    // fn as_array_view(&self) -> ArrayView1<'_, Ix>
    // {
    //     ArrayView1::from(self.slice())
    // }

    // /// Borrow as a read-write array view.
    // fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, Ix>
    // {
    //     ArrayViewMut1::from(self.slice_mut())
    // }

    #[doc(hidden)]
    fn equal(&self, rhs: &Self) -> bool {
        self.slice() == rhs.slice()
    }

    // /// Returns the strides for a standard layout array with the given shape.
    // ///
    // /// If the array is non-empty, the strides result in contiguous layout; if
    // /// the array is empty, the strides are all zeros.
    // #[doc(hidden)]
    // fn default_strides(&self) -> Self {
    //     // Compute default array strides
    //     // Shape (a, b, c) => Give strides (b * c, c, 1)
    //     let mut strides = Self::zeros(self.ndim());
    //     // For empty arrays, use all zero strides.
    //     if self.slice().iter().all(|&d| d != 0) {
    //         let mut it = strides.slice_mut().iter_mut().rev();
    //         // Set first element to 1
    //         if let Some(rs) = it.next() {
    //             *rs = 1;
    //         }
    //         let mut cum_prod = 1;
    //         for (rs, dim) in it.zip(self.slice().iter().rev()) {
    //             cum_prod *= *dim;
    //             *rs = cum_prod;
    //         }
    //     }
    //     strides
    // }

    // /// Returns the strides for a Fortran layout array with the given shape.
    // ///
    // /// If the array is non-empty, the strides result in contiguous layout; if
    // /// the array is empty, the strides are all zeros.
    // #[doc(hidden)]
    // fn fortran_strides(&self) -> Self
    // {
    //     // Compute fortran array strides
    //     // Shape (a, b, c) => Give strides (1, a, a * b)
    //     let mut strides = Self::zeros(self.ndim());
    //     // For empty arrays, use all zero strides.
    //     if self.slice().iter().all(|&d| d != 0) {
    //         let mut it = strides.slice_mut().iter_mut();
    //         // Set first element to 1
    //         if let Some(rs) = it.next() {
    //             *rs = 1;
    //         }
    //         let mut cum_prod = 1;
    //         for (rs, dim) in it.zip(self.slice()) {
    //             cum_prod *= *dim;
    //             *rs = cum_prod;
    //         }
    //     }
    //     strides
    // }

    // /// Creates a dimension of all zeros with the specified ndim.
    // ///
    // /// This method is useful for generalizing over fixed-size and
    // /// variable-size dimension representations.
    // ///
    // /// **Panics** if `Self` has a fixed size that is not `ndim`.
    // fn zeros(ndim: usize) -> Self;

    // #[doc(hidden)]
    // #[inline]
    // fn first_index(&self) -> Option<Self>
    // {
    //     for ax in self.slice().iter() {
    //         if *ax == 0 {
    //             return None;
    //         }
    //     }
    //     Some(Self::zeros(self.ndim()))
    // }

    // #[doc(hidden)]
    // /// Iteration -- Use self as size, and return next index after `index`
    // /// or None if there are no more.
    // // FIXME: use &Self for index or even &mut?
    // #[inline]
    // fn next_for(&self, index: Self) -> Option<Self>
    // {
    //     let mut index = index;
    //     let mut done = false;
    //     for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
    //         *ix += 1;
    //         if *ix == dim {
    //             *ix = 0;
    //         } else {
    //             done = true;
    //             break;
    //         }
    //     }
    //     if done {
    //         Some(index)
    //     } else {
    //         None
    //     }
    // }

    // #[doc(hidden)]
    // /// Iteration -- Use self as size, and create the next index after `index`
    // /// Return false if iteration is done
    // ///
    // /// Next in f-order
    // #[inline]
    // fn next_for_f(&self, index: &mut Self) -> bool
    // {
    //     let mut end_iteration = true;
    //     for (&dim, ix) in zip(self.slice(), index.slice_mut()) {
    //         *ix += 1;
    //         if *ix == dim {
    //             *ix = 0;
    //         } else {
    //             end_iteration = false;
    //             break;
    //         }
    //     }
    //     !end_iteration
    // }

    // /// Returns `true` iff `strides1` and `strides2` are equivalent for the
    // /// shape `self`.
    // ///
    // /// The strides are equivalent if, for each axis with length > 1, the
    // /// strides are equal.
    // ///
    // /// Note: Returns `false` if any of the ndims don't match.
    // #[doc(hidden)]
    // fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
    // where D: Dimension
    // {
    //     let shape_ndim = self.ndim();
    //     shape_ndim == strides1.ndim()
    //         && shape_ndim == strides2.ndim()
    //         && izip!(self.slice(), strides1.slice(), strides2.slice())
    //             .all(|(&d, &s1, &s2)| d <= 1 || s1 as isize == s2 as isize)
    // }

    // #[doc(hidden)]
    // /// Return stride offset for index.
    // fn stride_offset(index: &Self, strides: &Self) -> isize
    // {
    //     let mut offset = 0;
    //     for (&i, &s) in izip!(index.slice(), strides.slice()) {
    //         offset += stride_offset(i, s);
    //     }
    //     offset
    // }

    // #[doc(hidden)]
    // /// Return stride offset for this dimension and index.
    // fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    // {
    //     stride_offset_checked(self.slice(), strides.slice(), index.slice())
    // }

    // #[doc(hidden)]
    // fn last_elem(&self) -> usize
    // {
    //     if self.ndim() == 0 {
    //         0
    //     } else {
    //         self.slice()[self.ndim() - 1]
    //     }
    // }

    // #[doc(hidden)]
    // fn set_last_elem(&mut self, i: usize)
    // {
    //     let nd = self.ndim();
    //     self.slice_mut()[nd - 1] = i;
    // }

    // #[doc(hidden)]
    // fn is_contiguous(dim: &Self, strides: &Self) -> bool
    // {
    //     let defaults = dim.default_strides();
    //     if strides.equal(&defaults) {
    //         return true;
    //     }
    //     if dim.ndim() == 1 {
    //         // fast case for ndim == 1:
    //         // Either we have length <= 1, then stride is arbitrary,
    //         // or we have stride == 1 or stride == -1, but +1 case is already handled above.
    //         dim[0] <= 1 || strides[0] as isize == -1
    //     } else {
    //         let order = strides._fastest_varying_stride_order();
    //         let strides = strides.slice();

    //         let dim_slice = dim.slice();
    //         let mut cstride = 1;
    //         for &i in order.slice() {
    //             // a dimension of length 1 can have unequal strides
    //             if dim_slice[i] != 1 && (strides[i] as isize).unsigned_abs() != cstride {
    //                 return false;
    //             }
    //             cstride *= dim_slice[i];
    //         }
    //         true
    //     }
    // }

    // /// Return the axis ordering corresponding to the fastest variation
    // /// (in ascending order).
    // ///
    // /// Assumes that no stride value appears twice.
    // #[doc(hidden)]
    // fn _fastest_varying_stride_order(&self) -> Self
    // {
    //     let mut indices = self.clone();
    //     for (i, elt) in enumerate(indices.slice_mut()) {
    //         *elt = i;
    //     }
    //     let strides = self.slice();
    //     indices
    //         .slice_mut()
    //         .sort_by_key(|&i| (strides[i] as isize).abs());
    //     indices
    // }

    // /// Compute the minimum stride axis (absolute value), under the constraint
    // /// that the length of the axis is > 1;
    // #[doc(hidden)]
    // fn min_stride_axis(&self, strides: &Self) -> Axis
    // {
    //     let n = match self.ndim() {
    //         0 => panic!("min_stride_axis: Array must have ndim > 0"),
    //         1 => return Axis(0),
    //         n => n,
    //     };
    //     axes_of(self, strides)
    //         .rev()
    //         .min_by_key(|ax| ax.stride.abs())
    //         .map_or(Axis(n - 1), |ax| ax.axis)
    // }

    // /// Compute the maximum stride axis (absolute value), under the constraint
    // /// that the length of the axis is > 1;
    // #[doc(hidden)]
    // fn max_stride_axis(&self, strides: &Self) -> Axis
    // {
    //     match self.ndim() {
    //         0 => panic!("max_stride_axis: Array must have ndim > 0"),
    //         1 => return Axis(0),
    //         _ => {}
    //     }
    //     axes_of(self, strides)
    //         .filter(|ax| ax.len > 1)
    //         .max_by_key(|ax| ax.stride.abs())
    //         .map_or(Axis(0), |ax| ax.axis)
    // }

    // /// Convert the dimensional into a dynamic dimensional (IxDyn).
    // fn into_dyn(self) -> IxDyn
    // {
    //     IxDyn(self.slice())
    // }

    // #[doc(hidden)]
    // fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self>
    // {
    //     let mut s = Self::default();
    //     if s.ndim() == d.ndim() {
    //         for i in 0..d.ndim() {
    //             s[i] = d[i];
    //         }
    //         Some(s)
    //     } else {
    //         None
    //     }
    // }

    // #[doc(hidden)]
    // fn insert_axis(&self, axis: Axis) -> Self::Larger;

    // #[doc(hidden)]
    // fn try_remove_axis(&self, axis: Axis) -> Self::Smaller;

    // private_decl! {}
}
