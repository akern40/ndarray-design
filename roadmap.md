# Roadmap: from `Dimension` to `Layout`

Working document for moving `ndarray` from `ArrayParts { ptr, dim: D, strides: D }` to `ArrayParts { ptr, layout: L }`.
The base is `rust-ndarray/ndarray@refactor` (after #1575), with `refactor` names treated as canonical.
Checkboxes track progress. Anything marked **Q** is an open decision.

## 0. End state

```rust
pub struct ArrayBase<S: RawData, L> { data: S, parts: ArrayPartsSized<S::Elem, L> }
pub struct ArrayRef<A, L>(LayoutRef<A, L>);         // likewise RawRef / LayoutRef
pub type Array<A, L> = ArrayBase<OwnedRepr<A>, L>;
pub type Array2<A> = Array<A, StridedLayout<R2>>;   // aliases keep their names
```

- Rank, shape, strides, index, and layout are separate traits (§2).
- A strided layout is just one kind of `Layout`. Most of today's methods move to `impl<A, L: Strided> ArrayRef<A, L>`. Only offset-based functionality goes on `impl<A, L: Layout>`.
- `Dimension`, `Dim`, `Ix*`, `IxDynImpl`, `IntoDimension`, `DimAdd`, `DimMax`, `RemoveAxis`, `ShapeBuilder`, `StrideShape` and the old `NdIndex` are deprecated, then removed one release later.
- No `Cow` in the traits (it isn't available without `alloc`). Accessors either return `&` to stored data (a ZST `&` is free, so `const` shapes work too) or return an iterator.

## 1. Where we are

- [x] Reference types `LayoutRef`/`ArrayRef`/`RawRef`, made sound via a DST (#1440, #1532)
- [x] `LayoutBitset` rename; `crate::Layout` is now a deprecated alias (#1563, #1567)
- [x] `Rank`, `ConstRank<N>`, `DynRank`, `R0..R12`, `RAdd`, `RMax` (#1568)
- [x] `Ranked { type NDim; fn ndim }` added as a supertrait of `Dimension` (#1575)
- [ ] Export `Ranked` from the prelude (#1621). Also consider a crate-root re-export, since `use ndarray::Dimension;` users still break on `.ndim()` for concrete types.
- [ ] `Shape` + `NShape`/`DShape` (fork branch `shape-trait`)

## 2. Trait designs

### 2.1 `Rank` gets type maps

Make `Rank` the single place that maps a rank to its default concrete types. All the rank-changing methods (`index_axis`, `insert_axis`, `stack`, broadcasting) then come for free through `R::Smaller` / `R::Larger` / `RMax` / `RAdd`, so per-shape `Smaller`/`Larger` types aren't needed.

```rust
pub trait Rank: /* existing bounds */ {
    const N: Option<usize>;
    type Smaller: Rank;
    type Larger: Rank;
    // new
    type Shape:   ShapeMut<NDim = Self>;      // NShape<N> | DShape
    type Strides: StridesMut<NDim = Self>;    // NStrides<N> | DStrides
    type Index:   ?Sized + IndexFor<Self>;    // [usize; N] | [usize]
}
pub type StridedLayout<R> = Strided<<R as Rank>::Shape, <R as Rank>::Strides>;
```

Custom shapes (e.g. a `const` `ConstMatrixShape<N, M>`) fall back to `R::Shape` when an operation changes the rank or an axis length.

**Q:** `Index` for dynamic rank: should it be unsized `[usize]` (as in `layout-v2`) or a sized `DShape`-like `DIndex`? Unsized avoids allocation for `arr[&[1, 2, 3][..]]`, but it can't be returned from iterators (`indexed_iter`). A sized inline-small-vector type is probably needed for iteration either way.

### 2.2 `Shape`

The `shape-trait` design, plus a mutable subtrait and conversions:

```rust
pub trait Shape: Ranked + Clone + Eq + Debug + Send + Sync {
    type Iter<'a>: ExactSizeIterator<Item = usize> + DoubleEndedIterator where Self: 'a;
    fn axis_len(&self, axis: Axis) -> usize;     // Axis, not usize, to match the rest of the API
    fn iter(&self) -> Self::Iter<'_>;
    fn size_checked(&self) -> Option<usize>;     // provided
    fn size_bytes_checked<T>(&self) -> Option<usize>; // provided
    fn to_rank_default(&self) -> <Self::NDim as Rank>::Shape; // provided; the fallback for custom shapes
}
pub trait ShapeMut: Shape + IndexMut<Axis, Output = usize> {
    fn from_elem(ndim: usize, v: usize) -> Result<Self, ShapeStrideError<Self>>; // replaces Dimension::zeros
    fn as_slice(&self) -> &[usize]; // NShape/DShape own their storage, so this is allowed here
}
pub trait IntoShape { type NDim: Rank; type Shape: Shape<NDim = Self::NDim>; fn into_shape(self) -> Self::Shape; }
// IntoShape impls: usize, tuples (Ix0..6 patterns), [usize; N], &[usize], Vec<usize>, NShape, DShape
// (and Dim<_> during the transition)
pub trait Patterned: Shape { type Pattern; fn into_pattern(self) -> Self::Pattern; } // NShape<0..=6> only
```

- `LayoutRef::shape() -> &[usize]` stays as a concrete method while the layout stores an `NShape`/`DShape`. Generic code should use `shape_iter()` / `axis_len()`.
- `dim() -> Pattern` stays, bounded on `L::Shape: Patterned`. It's popular downstream (ndarray-linalg: 20 call sites).

**Q:** Should `Shape: Index<Axis>` be required? It isn't possible for computed shapes that can't hand out a `&usize`. Proposal: no, use `axis_len` instead.

### 2.3 `Strides`

Mirrors `Shape`, but with `isize` (which also fixes today's usize-reinterpreted-as-isize strides):

```rust
pub trait Strides: Ranked + Clone + Eq + Debug + Send + Sync {
    type Iter<'a>: ExactSizeIterator<Item = isize> + DoubleEndedIterator where Self: 'a;
    fn stride(&self, axis: Axis) -> isize;
    fn iter(&self) -> Self::Iter<'_>;
}
pub trait StridesMut: Strides + IndexMut<Axis, Output = isize> { fn as_slice(&self) -> &[isize]; }
```

Stride constructors live on the layout (§2.4), not on `Strides`, because they need a shape.

### 2.4 `Layout` and `Strided`

```rust
/// Safety: offset_range_checked must be exact, and None iff it doesn't fit in isize (see layout-v2 docs).
pub unsafe trait Layout: Ranked + Clone + Debug + Send + Sync {
    type Shape: Shape<NDim = Self::NDim>;
    fn shape(&self) -> &Self::Shape;                                   // & is free for ZST const shapes
    fn offset(&self, idx: &<Self::NDim as Rank>::Index) -> isize;      // unchecked in release
    fn offset_checked(&self, idx: &<Self::NDim as Rank>::Index) -> Option<isize>;
    fn offset_range_checked(&self) -> Option<(isize, isize)>;          // min..=max offset
    fn bitset(&self) -> LayoutBitset;                                  // feeds Zip
    fn is_c_contiguous(&self) -> bool; fn is_f_contiguous(&self) -> bool;
}
pub trait Strided: Layout {
    type Strides: Strides<NDim = Self::NDim>;
    fn strides(&self) -> &Self::Strides;
}
/// The ops ndarray actually does to views today. Each preserves the invariants.
pub trait StridedMut: Strided {
    fn slice_axis_inplace(&mut self, axis: Axis, s: Slice) -> isize;   // returns ptr offset
    fn collapse_axis(&mut self, axis: Axis, index: usize) -> isize;
    fn invert_axis(&mut self, axis: Axis) -> isize;
    fn swap_axes(&mut self, a: Axis, b: Axis);
    fn merge_axes(&mut self, take: Axis, into: Axis) -> bool;
    fn permute(&mut self, perm: &<Self::NDim as Rank>::Index);
}
/// Rank-changing ops output the rank-default strided layout.
pub trait StridedOps: Strided {
    fn index_axis(&self, axis: Axis, i: usize) -> (StridedLayout<<Self::NDim as Rank>::Smaller>, isize);
    fn insert_axis(&self, axis: Axis) -> StridedLayout<<Self::NDim as Rank>::Larger>;
    fn into_dyn(self) -> StridedLayout<DynRank>;
    fn into_rank<R: Rank>(self) -> Result<StridedLayout<R>, ShapeStrideError<Self>>;
}
pub trait FromShape: Layout { fn c(shape: Self::Shape) -> Self; fn f(shape: Self::Shape) -> Self; }

pub struct Strided<Sh, St> { shape: Sh, strides: St }  // the one concrete strided layout
```

- `StridedMut`/`StridedOps` can be blanket-implemented for `Strided<Sh: ShapeMut, St: StridesMut>`. Kept as traits so a ZST "C-contiguous const" layout can opt out.
- `Index` uses `Rank::Index` so `arr[[i, j]]` keeps working. `NdIndex<E>` becomes `NdIndex<L: Layout>` with `fn offset_checked(&self, &L)`. The tuple/`usize`/array impls carry over.
- **Q:** Should `Layout` require the reversible "iteration order" API from the design branch (`index_linear_left/right`, `index_memory_order`), or should that stay in iterators? Proposal: leave it out of v1. Iterators keep using `Strided`, and non-strided layouts get iterators later.

### 2.5 The bridge: `Dimension` associated types

```rust
pub trait Dimension: ... + Ranked {
    // new (hidden at first, not behind `unstable`, because they're needed internally)
    type Shape:   ShapeMut<NDim = Self::NDim> + Into<Self> + From<Self>;  // NShape<N> | DShape
    type Strides: StridesMut<NDim = Self::NDim>;
    type Layout:  StridedOps + StridedMut + FromShape + Strided<Shape = Self::Shape, Strides = Self::Strides>;
}
// Dim<[usize; N]>: Shape = NShape<N>, Layout = StridedLayout<ConstRank<N>>; IxDyn: DShape / StridedLayout<DynRank>
// Require <Self::NDim as Rank>::Shape == Self::Shape so D-world and R-world agree.
```

`ArrayParts<A, D, T>` stores `layout: D::Layout`. The `_dim()`/`_strides()` accessors become `_layout().shape()` / `.strides()`. `raw_dim()` converts `D::Shape -> D`, which costs the same as today's clone.

## 3. Deprecation and rename map

| Today | Replacement | Deprecate in |
|---|---|---|
| `crate::Layout` (alias of `LayoutBitset`) | `LayoutBitset` | already deprecated; remove in next breaking release (frees the name) |
| `crate::Shape<D>` (shape_builder) | rename to `ShapeOrder<D>` (or similar) + deprecated alias | Phase 1 |
| `StrideShape<D>` | `Strided<Sh, St>` / `(shape, strides)` args | switch |
| `ShapeBuilder` (`.f()`, `.set_f()`, `.strides()`) | `IntoShape` + `Order` / `FromShape::c`/`f`; `(sh, Order::F)` via `ShapeArg` | switch (keep `.f()` sugar on `IntoShape`?) **Q** |
| `IntoDimension` | `IntoShape` | switch (blanket `IntoShape for T: IntoDimension` during transition) |
| `Dim(..)`, `Ix0()..Ix6()`, `IxDyn()` fns | `NShape::from`, `[a, b]`, `DShape::from` | switch |
| `Ix0..Ix6`, `IxDyn` types | deprecated aliases, see **Q-switch** | switch |
| `Dimension::{slice, slice_mut}` | `ShapeMut/StridesMut::as_slice` | switch |
| `default_strides`, `fortran_strides` | `FromShape::c/f` | switch |
| `zeros(ndim)` | `ShapeMut::from_elem(ndim, 0)` | switch |
| `stride_offset(_checked)`, `NdIndex<D>` | `Layout::offset(_checked)`, `NdIndex<L>` | switch |
| `first_index/next_for/next_for_f` | internal index iterator over `Rank::Index` | internal, Phase 4 |
| `is_contiguous`, `_fastest_varying_stride_order`, `min/max_stride_axis` | `Layout::is_*_contiguous`, `Strided` helpers | Phase 3 (pub(crate)) |
| `insert_axis/try_remove_axis/RemoveAxis` | `StridedOps::index_axis/insert_axis`, `Rank::Smaller/Larger` | switch |
| `D::Smaller/Larger` in signatures | `StridedLayout<R::Smaller>` etc. (rank-default) | switch |
| `DimAdd/DimMax` | `RAdd/RMax` | switch |
| `NDIM` | `Rank::N` | switch |
| `Pattern`/`into_pattern` | `Patterned` | switch |
| `IxDynImpl` | `DShape` | switch |
| `as_array_view(_mut)` | `aview1(shape.as_slice())` | switch (ndarray-stats uses it 4x) |
| `NdProducer::{Dim, raw_dim, uget_ptr(&Dim)}` | `type Layout: Strided` / `shape()` / `uget_ptr(&Index)` | switch (sealed, so only the bounds are visible) |
| `SliceArg<D>::OutDim` | `SliceArg<R: Rank>::OutRank` | switch |

## 4. Phases (each bullet is roughly one PR)

### Phase 1: free names, finish `Ranked` (non-breaking)
- [ ] #1621 `Ranked` in prelude (+ crate-root re-export?)
- [ ] Rename `shape_builder::Shape` with a deprecated alias. Plan to remove `crate::Layout` at the switch.
- [ ] Re-export new traits only under `ndarray::layout::*` until the switch, so nothing collides at the crate root.
- [ ] Settle the `unstable` feature policy. Proposal: new traits are public under `layout::` but `#[doc(hidden)]`/documented-unstable, and not gated by a feature. Gating `Dimension`'s associated types behind a feature makes internal code use two code paths.

### Phase 2: shapes and strides (non-breaking)
- [ ] `Shape` + `NShape`/`DShape` (`shape-trait`). Add `ShapeMut`, `IntoShape`, `Patterned`, `Axis` indexing.
- [ ] `Strides` + `NStrides`/`DStrides` (share `ShapeStrideN` / `DynAxesRepr`)
- [ ] `Rank::{Shape, Strides, Index}` associated types
- [ ] `Dimension::{Shape, Strides}` + `From`/`Into` between `Dim<[usize; N]>` ↔ `NShape<N>` and `IxDyn` ↔ `DShape`
- [ ] `impl IntoShape for T: IntoDimension` (temporary bridge)

### Phase 3: layout traits (non-breaking)
- [ ] `Layout`, `Strided`, `StridedMut`, `StridedOps`, `FromShape`, `Strided<Sh, St>` + tests (port the `dimension/mod.rs` free fns: `do_slice`, `do_collapse_axis`, `merge_axes`, `offset_from_low_addr_ptr_to_logical_ptr`, `max_abs_offset_check_overflow`)
- [ ] `Dimension::Layout`
- [ ] `NdIndex` bridge: `NdIndex<L>` impls alongside `NdIndex<D>`

### Phase 4: swap the guts (non-breaking, internal; the big one)
- [ ] `ArrayParts { ptr, layout: D::Layout }` + accessors. Keep `D` as the public generic. Can probably be done with `_dim()`/`_strides()` shims first.
- [ ] Migrate internals one module at a time, removing the shims as you go: constructors → slicing/axis ops (`impl_methods.rs`) → iterators/`Lanes`/`AxisIter` → `zip`/`NdProducer` → reshape/broadcast/stacking → `parallel` → `linalg`
- [ ] Benchmarks before and after (`benches/`). Watch `IxDyn` (it's `DShape` + `DStrides` now: two inline-small-vector reprs instead of two `IxDynImpl`s, which should be roughly neutral) and the `Ix1`/`Ix2` hot loops.

### Phase 5: introduce the new public API (non-breaking)
- [ ] `layout()`, `shape_iter()`, `strides_iter()` on `LayoutRef`
- [ ] Constructors that accept `IntoShape` (they already do, via the bridge from Phase 2)
- [ ] Write the migration guide in docs (`src/doc/`)

### Phase 6: the switch (breaking release)
- [ ] The generic changes from `D: Dimension` to `L: Layout` on `ArrayBase`/refs. Aliases `ArrayN`/`ArrayD`, `ArrayViewN`, ... are redefined over `StridedLayout<R>`.
- [ ] Methods move to `impl<A, L: Strided...>`; signatures switch from `D::Smaller` to `StridedLayout<<L::NDim as Rank>::Smaller>`
- [ ] Deprecate everything in §3. Remove `crate::Layout`; move the new `Shape`/`Layout` to the crate root.
- [ ] Downstream PRs: ndarray-linalg, ndarray-stats, rust-numpy (rust-numpy is the heaviest user: `Dimension` 92 call sites, `IntoDimension` 23, `.strides()` 22)

**Q-switch: what happens to `Array<A, Ix2>` written out by hand?** `Ix2` is both a type alias (type namespace) and a fn (value namespace). Proposal: `#[deprecated] type Ix2 = StridedLayout<R2>;` and `fn Ix2(a, b) -> NShape<2>`. Then `Array<f64, Ix2>` and `Ix2(3, 4)` still compile, while `let d: Ix2 = a.raw_dim()` breaks (`raw_dim` gets deprecated and returns `L::Shape`). Alternatively `Ix2` aliases `NShape<2>` and arrays take `StridedLayout<_>`, which breaks every explicit `Array<A, IxN>` instead. I'd pick the first.

## 5. Risks / notes

- **Sealed traits.** `Dimension`, `NdProducer` and `SliceArg` are sealed, so adding associated types or supertraits doesn't break implementors. It *can* still break callers, as `ndim` showed: watch for methods moving between traits on concrete types.
- **`Dimension` arithmetic supertraits** (`Add`, `Mul`, ...) are used internally for shape math. `NShape` has them; decide whether `ShapeMut` requires them or whether they stay as inherent impls.
- **Compile times.** `Rank` already brings 13×13 `RAdd`/`RMax` impls. Adding associated types to `Rank` is cheap, but generic `StridedLayout<R::Smaller>` signatures add projection depth.
- **MSRV.** GATs (`Shape::Iter<'a>`) need 1.65, so 1.87 is fine.
