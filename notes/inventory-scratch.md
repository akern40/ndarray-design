# Scratch inventory (base: rust-ndarray/ndarray@refactor 81756e66)

## Dimension trait (src/dimension/dimension_trait.rs), sealed via private_decl!
Supertraits: Clone+Eq+Debug+Send+Sync+Default+IndexMut<usize,Output=usize>+Add/Sub/Mul(+Assign, &Self, usize)
  + DimMax<{Ix0,Self,IxDyn,Smaller,Larger}> + DimAdd<{Self,Smaller,Larger,Ix0,Ix1,IxDyn}> + Ranked
Items: const NDIM; type Pattern: IntoDimension<Dim=Self>; type Smaller: Dimension; type Larger: Dimension+RemoveAxis;
  into_pattern, size, size_checked, slice, slice_mut, as_array_view(_mut), equal, default_strides, fortran_strides,
  zeros(ndim), first_index, next_for, next_for_f, strides_equivalent, stride_offset, stride_offset_checked,
  last_elem, set_last_elem, is_contiguous(dim,strides), _fastest_varying_stride_order, min_stride_axis,
  max_stride_axis, into_dyn, from_dimension, insert_axis, try_remove_axis.
Internal use counts (grep src): slice() 116, slice_mut() 60, zeros( 140, stride_offset 74, IntoDimension 52,
  remove_axis 48, ::Smaller 48, DimMax 47, NdIndex 43, insert_axis 39, into_dyn 28, Pattern 28, DimAdd 21,
  default_strides 20, NDIM 20, min_stride_axis 19, into_pattern 16, is_contiguous 14, ::Larger 14, next_for 13.

## Public exports touching D (lib.rs)
dim::* (Dim, Ix0..Ix6, IxDyn, Ix, Ixs), Axis, AxisDescription, Axes, Dimension, IntoDimension, RemoveAxis,
DimAdd, DimMax, IxDynImpl, NdIndex, ShapeError/ErrorKind, indices/indices_of, Order, slice::{SliceArg(D),
SliceInfo, SliceNextDim, MultiSliceArg, NewAxis}, shape_builder::{Shape<D>, ShapeArg, ShapeBuilder, StrideShape<D>},
zip::{NdProducer(type Dim; raw_dim; uget_ptr(&Dim)), IntoNdProducer}, layout::{Layout (deprecated alias), LayoutBitset},
stacking::{concatenate, stack -> D::Larger}, Dimension-generic free fns in dimension/mod.rs (mostly pub(crate)?).
NAME COLLISIONS: crate::Shape (shape_builder), crate::Layout (deprecated LayoutBitset alias).

## Public methods exposing D
LayoutRef: dim()->D::Pattern, raw_dim()->D, shape()->&[usize], strides()->&[isize], stride_of, ndim, len_of, ...
ArrayRef: index_axis(_mut) -> D::Smaller; rows/columns/lanes(_mut) -> Lanes<D::Smaller>; outer_iter/axis_iter(_mut)
  -> AxisIter<D::Smaller>; fold_axis/map_axis(_mut) -> Array<B, D::Smaller>; assign<E: Dimension>; flatten -> Ix1;
ArrayBase: index_axis_move, insert_axis -> D::Larger, remove_axis -> D::Smaller, into_dyn -> IxDyn, into_diag, into_flat;
  push(axis, ArrayView<D::Smaller>); views into_outer_iter/into_axis_iter -> D::Smaller.
Constructors all take Sh: ShapeBuilder<Dim=D> (from_elem, zeros, ones, default, from_shape_fn, from_shape_vec(_unchecked),
  uninit, build_uninit, view from_shape/from_shape_ptr, raw view from_shape_ptr). into_shape_with_order etc take ShapeArg.
Index ops: Index<I: NdIndex<D>>; get(I), uget(I).

## Other D-generic traits
NdProducer { type Dim: Dimension; raw_dim; equal_dim; uget_ptr(&Dim); stride_of(Axis); split_at } (sealed)
SliceArg<D> { type OutDim; in_ndim; out_ndim } (unsafe, sealed)
NdIndex<E> { index_checked(&self, dim:&E, strides:&E); index_unchecked(&self, strides:&E) } (unsafe, impl for D, tuples, arrays, usize, &[usize])
IntoDimension { type Dim; into_dimension } for usize, tuples, [usize;N], &[usize], Vec<usize>, IxDynImpl...

## Downstream usage (local clones)
ndarray-linalg: Dimension 21, .dim() 20, .shape() 17, D::Pattern 3, .strides() 3, ShapeBuilder 2, raw_dim 2
ndarray-stats: .shape() 31, RemoveAxis 23, Dimension 18, D::Smaller 16, D::Pattern 10, into_pattern 4, as_array_view 4, raw_dim 3, into_dyn 3
rust-numpy: Dimension 92, .shape() 50, IntoDimension 23, .strides() 22, Dim( 6, raw_dim 5, into_dyn 4, StrideShape 3, D::NDIM 2, .slice() 2, ShapeBuilder 1, NdIndex 1
=> D::Smaller/RemoveAxis, Pattern/dim(), IntoDimension, shape()/strides() slices are the load-bearing compat surface.

## User decisions (2026-10-18)
- `refactor` naming canonical: Rank/ConstRank/DynRank/R0..R12/RAdd/RMax/Ranked(NDim, ndim).
- crate::Shape and crate::Layout collisions: rename via deprecation first.
- No Cow in traits (no_std concern) -> iterator-based accessors.
- D -> layout mapping: associated type on Dimension.
