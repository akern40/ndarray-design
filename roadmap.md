# Roadmap: from `Dimension` to `Layout` (draft)

This is mostly a list of open design questions. Only §1 has been decided (§1a records the output-type decision, §1b the `Layout`/`Strided` split, each with its reasoning). §2 is the one next step that's concrete. Everything after that is still undecided, and the aim is to answer each question before writing the code that depends on it.

Guiding principle: introduce **minimal traits that are clearly right**, then grow them as real needs come up. Having less functionality than today's API is fine. Locking in a bad decision isn't.

## 1. Decided

- **Goal:** `ArrayParts` stores a layout instead of explicit `dim`/`strides`. `Dimension` is the integration point during migration and reaches the layout through an **associated type**. A later breaking switch deprecates the `D`-based APIs.
- **Names:** the names on the `rust-ndarray/ndarray@refactor` branch are canonical (`Rank`, `ConstRank`, `DynRank`, `RAdd`, `RMax`, `Ranked`).
- **`Rank` only encodes the number of axes.** `Ranked` connects it to the runtime `ndim()`. `Rank` doesn't know about shapes, strides, indices or any representation.
- **No `Cow` in the traits.** Shape and stride access goes through iterators (or `&` to stored data).
- **Shapes are read-only, with no public safe API for mutating them.** Changing a shape almost always requires checking invariants about the strides and the allocation. So any change happens through operations that keep those invariants, never through "set axis length".
- **Name collisions** (`crate::Shape` from `shape_builder`, and `crate::Layout` as the deprecated alias of `LayoutBitset`) are resolved by deprecation/renaming before the new names reach the crate root.
- **Landed so far:** the reference types (#1440, #1532), the `LayoutBitset` rename, `Rank` (#1568) and `Ranked` (#1575). In review: #1621 (`Ranked` in the prelude) and #1622 (lint fixes).

### 1a. Output types: how an operation names the layout type it returns

A layout type is a record of what the compiler knows about an array's layout. Today that is one fact, the rank. Operations have to name their output type from their input types alone, and two constraints shape every option: axes are runtime `usize` arguments, so an output type can never depend on *which* axis was touched; and stable Rust has no arithmetic on const generics, so `NLayout<{N - 1}>` can't be written.

Decided (mockups comparing the alternatives are in `mockups/output-type/`, with a table in its README):

- **The rule: output types depend on *counts* and *static names*, never on runtime values.** This is the rule nalgebra follows (`remove_fixed_rows::<D>(i)` vs `remove_rows(i, n)`). For rank the count is almost always a static 1 (`remove_axis`, `insert_axis`, `index_axis`, `stack`, `axis(k)` iteration), so today's `Smaller`/`Larger` *are* the fixed-count API; a runtime count means dynamic rank. Where the whole shape is an argument (`into_shape`, `SliceArg`), the output type follows the argument's type, as it already does.
- **Rank-changing operations keep per-operation associated types** (`Smaller`, `Larger`, dyn; today's `RemoveAxis` pattern), not a GAT. The single-GAT alternative (`type WithRank<R: Rank>`) needs a rank→container map implemented on the rank types, which leaks as bounds on every rank expression in generic signatures, and the GAT doesn't compose (`S::WithRank<A>::WithRank<B>` isn't known to equal `S::WithRank<B>`), so chained operations need equality bounds. Per-operation types chain for free and are what downstream code already uses.
- **v1 layouts encode rank only.** `const` axis lengths are a *separate shape family* that can be added later as a tuple of per-axis `Const<N>` / `Dyn` slots (nalgebra's `(R, C)` generalised to N axes). Today's `Ix<N>` is then "all axes `Dyn`". Nothing in the rank-only traits needs to anticipate it, provided no trait method returns `Self` after changing a length or rank.
- **Erasure rule for the const family:** an operation preserves a `const` length only if it names the axis statically *and* doesn't change that axis's length (indexing, element-wise ops, `dot`, static transposes, `fixed_slice::<AXIS, LEN>`). Any runtime axis argument erases *all* lengths to `Dyn`; any runtime length argument erases *that* axis. So `const` shapes are entry points, not full citizens; that is acceptable because the audience (small fixed vectors and matrices) mostly indexes, multiplies and transposes.
- **Length arithmetic (`concatenate`, `split_at`, const `reshape`) erases to `Dyn`** until Rust has const-generic arithmetic. The macro-table trick that works for rank (`RAdd`/`RMax`, formerly `Dimensionality::{DAdd, DMax}`) doesn't transfer: rank is bounded by 12, lengths aren't. Typenum is rejected for the reasons in the `Dimensionality` doc comment. Macros can't help either: a macro sees tokens, not the inputs' types.
- **Where the caller states the result, the compiler can still check it** without `generic_const_exprs`, via an inline `const` assertion (stable since 1.79):
  ```rust
  fn concatenate<const A: usize, const B: usize, const S: usize>(a: Array<(Const<A>, Dyn)>, b: Array<(Const<B>, Dyn)>) -> Array<(Const<S>, Dyn)> {
      const { assert!(A + B == S) };
      ...
  }
  let c: Array<(Const<6>, Dyn)> = concatenate(a, b);  // `S` inferred from the annotation; 3 + 3 == 6 is checked, a wrong `S` is a compile error
  ```
  `S` can't be inferred from `A + B`, and the check is post-monomorphisation, both acceptable. If `generic_const_exprs` lands, `S` gets the default `{A + B}` without an API break. A bounded `DimAdd` impl table for small lengths would add inference and is a pure addition that can be decided later.

### 1b. `Layout`, an axis-operations trait, and `Strided`; design targets include sparse and block layouts

Decided:

- **Design targets.** Besides today's dense strided arrays, the layout traits must accommodate (1) layouts with *computed* strides (C/F-contiguous and `const`-shape layouts that store no strides), and (2) **block/tiled** layouts (Morton/Z-order, GPU and image tilings). Not every operation has to be efficient, or even available, on every layout, but the traits must not make them unrepresentable. (Arrow's `Tensor` is strided and is not a separate target.) **Sparse** arrays are also a design target, but they are a *storage*, not a layout: see §1c.
- **Three levels, each bound by the weakest trait that suffices:**
  - **`Layout`**: the shape, index → location, the valid offset range (soundness), and iteration over stored locations in the layout's preferred order. Iteration belongs here: the strided impl walks pointers, a general impl walks indices. Fast paths are an implementation detail of each layout, not a property of the trait.
  - **An axis-operations trait (name open; `AxisOps` as a placeholder), `: Layout`**: the structural operations with output types from §1a — `index_axis → Smaller`, `insert_axis → Larger`, permutation, broadcasting, slicing — and therefore the home of the `axis(k)` handle. None of these need strides: `index_axis` on any layout is "remove slot `k` from the shape, pin coordinate `i` in the location function", and each family chooses its `Smaller`. It is a separate trait because at least one non-strided family can implement it in full: **tiled** layouts (`offset = Σ_k (q_k·S_k + r_k·s_k)`, `(q_k, r_k) = divmod(i_k + start_k, t_k)`) are piecewise strided and closed under every one of these ops. Morton/Z-order (per-axis bit masks) is closed under all of them *except* offset slicing, so slicing may later split off on its own; not decided. Block layouts *may* implement this trait; nothing requires them to.
  - **`Strided: AxisOps`** is where strides are observable, stored or computed: `strides()`, contiguity and `as_slice`, `is_standard_layout`, `from_shape_strides`, raw-view constructors, BLAS handoff. A short list, so the method-by-method audit (§3.4) is cheap.
- **Rejected:** a single strided `Layout` trait (rules out the design targets), and "`Layout` internal-only in v1" (pays the audit cost without the benefit).

### 1c. Sparse is a storage, not a layout (mdspan-style: layout = index → offset, accessor = offset → element)

Decided:

- **Sparse arrays live in `S`, not `L`.** A sparse storage is "a `Vec` with holes": a virtual dense buffer whose stored elements are kept as sorted `(physical offset, value)` pairs, optionally with an acceleration index (row pointers). The layout stays a *total* map from index to physical offset, exactly as for dense arrays, and the storage answers `access(offset) -> Option<&A>` (missing → the fill value). Consequences: every structural op (transpose, slice, broadcast, `index_axis`, `reversed_axes`) is free on a sparse array because it only changes the layout — CSR sorted by C-order offset, transposed, *is* CSC, with the same buffers, as in scipy. One `ArrayBase<S, L>`, one set of methods, no second generic story.
- **Storage stops assuming a pointer.** `ArrayParts { ptr: NonNull<A>, layout }` becomes `{ handle: S::Handle, layout }` with `S::access(handle, offset)`; this is mdspan's `data_handle_type` + `AccessorPolicy`. For dense storage the handle is `NonNull<A>` and access is `ptr.offset(o)`, zero cost. This is a rewrite of the `data_traits` layer and it has to happen *in the same step as the storage swap* (§4): retrofitting it afterwards means touching every `dim`/`strides` site twice.
- **Two generics stay two.** The accessor carries the element type (`Acc::Elem`), so `ArrayRef<A, D>` becomes an alias for an `ArrayRefBase<Dense<A>, L>` — the same trick as `ArrayBase<S, D>` and mdspan's `default_accessor<T>`. Dense code sees no change.
- **Storage and layout are orthogonal; compatibility is a runtime invariant.** The only coupling is "the layout's offset range lies inside the storage's (virtual) buffer", checked at construction and preserved by every structural op, as today. The accessor takes an offset and never sees the layout, so there are 5 storage kinds × N layouts with one set of impls, not a combined `Backend`. The one *type-level* compatibility case is static sizing (inline `[A; 6]` for a `(Const<2>, Const<3>)` layout); that is a bound on the constructor, not on the struct.
- **Uniqueness gates mutability.** Non-unique layouts already exist (broadcast views: a stride-0 axis maps many indices to one offset) and ndarray already forbids mutating through them. Generalise to `Layout::is_unique()` (static when built from a shape, runtime-checked for arbitrary strides; mdspan's `is_unique`) and the rule: anything that hands out `&mut A` per index — `iter_mut`, mutable producers in `Zip`, `IndexMut`, sparse insertion — is only constructible over a unique layout. Reads through non-unique layouts are fine for dense and sparse alike (packed symmetric storage is the classic case).
- **Insertion** (`IndexMut` on a hole) is an owned/unique-storage operation through `DataMut::try_ensure_unique`, like dense `push`. A mutable view of a sparse array holds a `&mut` to the sparse buffer, not a `*mut A` — which is the handle abstraction again.
- **Known costs, accepted:** O(nnz) iteration is driven by the storage, so the layout must *invert*, `index_of(offset) -> Option<Index>` (well-defined for unique layouts; a new `Layout` requirement). A sliced sparse view iterates the underlying nnz and filters, O(nnz_total) not O(nnz_view), unless storage and layout cooperate (row-pointer tricks that only help when the layout order matches the storage sort order; scipy has the same caveat). Sparse ⊗ sparse `Zip` is format-specific in every option and is out of generic scope.
- **Mockup:** `mockups/storage-handle/` is a compiling, Miri-clean model of all of the above (dense + sparse accessors, `Deref` by cast, layout-only transpose/slice on sparse, insertion through `&mut ArrayRef`, CoW handle refresh, branch-free dense codegen). Two findings from it: (1) the stable-handle rule is stricter under Stacked Borrows than "lives on the heap" — an owned storage must not hold its buffer through a `Box` (moving it is a `Unique` retag that invalidates the handle); raw pointer + `Drop`, as `OwnedRepr` already does; (2) slicing needs `Accessor::offset(handle, by)` (mdspan's `accessor.offset`), because a sparse handle can't be bumped like a data pointer; the alternative is a base offset inside every layout.
- **Rejected:** sparse as a `Layout` (would make `offset(index)` fallible, layouts heap-owning with a borrowed form for views, and dense code would pay for all of it); and "dense `ArrayRef` plus separate sparse structs sharing array-level traits" (two ways to write generics). The latter remains a fallback if the handle abstraction proves too costly in the dense hot paths (§4, performance).

## 2. Next step: a minimal `Shape`

The `shape-trait` branch has `Shape: Ranked` with `axis_len`, `iter`, and provided methods `size_checked`, `size_bytes_checked` and `iter_isize`. Questions to settle before it lands:

- Should `axis_len` take `usize` or `Axis`? `Axis` matches the public API.
- Does `iter_isize` belong here? It's an offset/strides concern.
- Which bounds should it have (`Clone`, `Eq`, `Debug`, `Send`, `Sync`)? Keep them minimal and add them when a user of the trait needs them.
- How should it connect to `Dimension`? Either `Dimension: Shape` (the `Dim` types already *are* shapes, so this is cheap and doesn't commit to a representation) or an associated `Dimension::Shape` (which commits to a representation now).
- **Leave out on purpose:** mutation, construction, rank-changing operations, and pattern/tuple conversion.

## 3. Open questions: trait design

### 3.1 Where do rank-changing operations live?
Answered in §1a: per-operation associated types on the shape/layout trait (or per-operation traits), with `const` shapes erasing to the plain family. Still open: whether they sit on `Shape`, on `Layout`, or on the `axis(k)` handle (see §3.4).

### 3.2 How do shapes and strides change at all?
There's no public safe setter, but slicing, collapsing, inverting, permuting and reshaping all have to happen somehow.
- What's the smallest set of invariant-preserving operations? Are they on the layout, or free functions over concrete types?
- Does the unsafe "raw mutation" surface need to be a trait so generic internal code can use it, or can it live on each concrete representation?
- Partially-mutable representations (some axes `const`): answered in §1a, they erase to the plain family on any runtime-axis or runtime-length operation.
- Constructors should check (shape, strides, allocation length) together. Where does that check live?

### 3.3 Strides
- Is `Strides` a standalone trait, or does it only exist inside a strided layout? Does anything need generic strides outside a layout?
- Strides should be signed `isize`, which also gets rid of today's practice of storing strides as `usize` and reinterpreting them as `isize`.
- Should strides ever be `const`?

### 3.4 Layout
The split is decided in §1b. What remains is the contract, driven by the design targets:
- **Index → location.** Closed by §1c: `offset(index) -> isize` stays total and infallible on `Layout`, because sparse is a storage. New in its place: `Layout` needs an inverse, `index_of(offset) -> Option<Index>`, for storage-driven iteration, and `is_unique()` to gate mutable access. Open: are these on `Layout` itself, or on a sub-trait that dense `Strided` and sparse-capable layouts implement? (`mockups/storage-handle/` puts both on `Layout`; `index_of` is only defined when `is_unique()`.)
- **Closure under structural ops.** Decided in §1b: they live on the axis-operations trait, which a layout may or may not implement. Still open: whether slicing needs to split off from it (Morton), and whether a layout can implement it *partially* (CSR can index rows cheaply, columns not at all) — sub-traits, `Option`-returning methods, or "then don't implement it".
- **Minimal surface.** Probably: the shape, index → location, the offset range (soundness), and offset iteration. Does `shape()` return `&Self::Shape` (works for stored shapes and zero-sized `const` ones), or does `Layout` expose shape methods directly so shapes can be computed?
- **Audit.** Done; see the table below. Headline: ~60% of the method surface needs only `Layout`, the view/slicing/axis-iteration family needs the axis-ops trait, and `Strided` is small — strides accessors, the reshape family's fast path, and three stride-constructed view families (`windows`, `exact_chunks`, `diag`). A fourth bound shows up that §1b didn't name: **dense-only** (`Acc = Dense<A>`), for everything that hands out pointers, slices, or reinterprets memory.

**Rule for the audit (decided):** minimise the ripple of the redesign. Where today's bound is stricter than the weakest sufficient one, keep today's bound if loosening it later is non-breaking (it always is for a bound on an `impl` block or method), and only pay for the weaker bound now where it is needed for a §1 decision to work at all. The table still records the weakest bound, because that is what the eventual API should converge to.

#### Audit: weakest bound per method (against `upstream/refactor`)

Columns are the *layout* bound; the storage bound (`RawData`/`Data`/`DataMut`/owned) is orthogonal and unchanged from today. "Fallback" means the method stays available on the weaker bound by copying.

| Bound | Methods | Notes |
|---|---|---|
| **`Layout`** (shape + `offset` + `offset_range` + `indices`) | `len`, `len_of`, `is_empty`, `ndim`, `dim`, `raw_dim`, `shape`; `first/last(_mut)`, `get(_mut)`, `uget(_mut)`, `Index`/`IndexMut`, `swap`, `uswap`; `iter(_mut)`, `indexed_iter(_mut)`, `into_iter`, `for_each`, `fold`, `map`, `map_mut`, `mapv`, `mapv_into(_any)`, `map_inplace`, `mapv_inplace`, `zip_mut_with`, `assign`, `assign_to`, `fill`; `view(_mut)`, `to_owned`, `to_shared`, `into_owned`, `into_shared`, `try_into_owned_nocopy`, `to_vec`, `into_scalar`, `into_dyn`, `into_dimensionality`, `move_into(_uninit)`; `sum`, `product`, `mean`, `var`, `std`, `pow2`, `clamp`, `angle`, `to_complex_re/im`, `partition`, `select` (copies); arithmetic ops; `from_elem`, `zeros`, `ones`, `default`, `from_shape_fn`, `from_shape_simple_fn`, `uninit`, `build_uninit`, `eye`, `from_diag(_elem)`; `nrows`, `ncols`, `is_square`; `is_view`, `is_owned`, `is_unique` | Iteration order is the layout's; the strided impl walks pointers, a generic layout walks `indices()`. With `Acc::DENSE == false`, `map`/`fold`/`sum`/`iter` are storage-driven (`stored` + `index_of`) instead. |
| **axis-ops** (`index_axis → Smaller`, `insert_axis → Larger`, slice, permute, broadcast) | `slice(_mut)`, `slice_move`, `slice_collapse`, `slice_axis(_mut/_inplace/_move)`, `slice_each_axis(_mut/_inplace)`, `multi_slice_mut`, `multi_slice_move`, `split_at`; `index_axis(_mut/_move/_inplace)`, `collapse_axis`, `remove_index`; `rows(_mut)`, `columns(_mut)`, `lanes(_mut)`, `outer_iter(_mut)`, `axis_iter(_mut)`, `axis_chunks_iter(_mut)`, `into_*_iter`, `fold_axis`, `map_axis(_mut)`, `accumulate_axis_inplace`, `sum/product/mean/var/std_axis`, `diff`, `cumprod`, `row(_mut)`, `column(_mut)`, `triu`, `tril`; `broadcast`, `insert_axis(_inplace)`, `remove_axis`, `squeeze`; `swap_axes`, `permuted_axes`, `permute_axes`, `reversed_axes`, `reverse_axes`, `t`, `invert_axis` (slice with step −1); `stack`, `concatenate` | All expressible as "new layout, same handle", with `Accessor::offset` absorbing the base shift (§1c). Nothing here reads a stride. |
| **`Strided`** | `strides`, `stride_of`, `axes`, `max_stride_axis`, `min_stride_axis`, `is_standard_layout`, `merge_axes`; reshape family — `to_shape`, `into_shape(_with_order/_clone)`, `reshape`, `flatten(_with_order)`, `into_flat`, `as_standard_layout` — *fast path*, with a `Layout` copy fallback (what `to_shape`/`CowArray` already is); `windows(_with_stride)`, `axis_windows(_with_stride)`, `exact_chunks(_mut)`, `diag(_mut)`, `into_diag`; `from_shape_vec`/`from_shape_ptr` with explicit strides; `Zip`'s inner-loop fast path (`layout()`, `stride_of`, `contiguous_stride`) | `windows`/`exact_chunks`/`diag` construct a view by writing strides (duplicated, multiplied, summed) and have no stride-free definition that a tiled layout is closed under, so they stay `Strided` unless the axis-ops trait grows `diagonal`/`split_axis` ops — not proposed. |
| **dense-only** (`Acc = Dense<A>`) | `as_ptr`, `as_mut_ptr`, `get_ptr`, `get_mut_ptr`, `raw_view(_mut)`, `from_shape_ptr`, `deref_into_view(_mut)`, `cast`, `split_complex`, `cell_view`, `into_cell_view`, `assume_init`; `as_slice*`, `to_slice*`, `into_slice*`, `into_raw_vec(_and_offset)`, `from_vec`, `from_iter`, `from_shape_vec(_unchecked)`; `push`, `append`, `push_row/column`, `reserve*` (+ `Strided`: they need a known memory order); BLAS paths of `dot`, `general_mat_mul`, `general_mat_vec_mul`, `scaled_add` (+ `Strided`; `Layout` fallback exists today); `NdProducer::{Ptr, as_ptr, uget_ptr}` | These are the real cost of §1c inside the crate: today they're `impl ArrayRef<A, D>`, tomorrow `impl ArrayRef<Dense<A>, L>`. Several (`cast`, `split_complex`, `assume_init`, `cell_view`) are *accessor reinterpretations* and could become one `Accessor` method instead of dense-only; by the minimise-ripple rule they stay dense-only for now. |

What doesn't fit cleanly, each needing a decision before §4:

1. **`Zip`/`NdProducer`.** Today: one index space, N producers, two loop shapes (flat loop when all producers are contiguous in the same order; otherwise walk outer indices with `first_index`/`next_for`, call `uget_ptr(index)` per producer, bump cursors by `stride_of(unroll_axis)` in the inner loop). `split_at` exists only for rayon. Two observations: (i) `NdProducer::Ptr` is already an abstract cursor (`Offset::stride_offset` is all the loop uses; the `Indices` producer uses an index as its "pointer"), so `Ptr = Acc::Handle` + `Accessor::offset` is a rename, and the storage side of §1c costs `Zip` almost nothing. (ii) The layout side is the question: `raw_dim`/`uget_ptr` need `Layout`, `split_at` needs axis-ops, and `layout()`/`stride_of`/`contiguous_stride` need `Strided` — purely for the fast path. So `Zip` (and `map`/`fold`/arithmetic built on it) is `Strided`-bound today, contradicting §1b's "iteration needs only `Layout`". **Decided: minimise ripple — `Zip` stays `Strided`-bound and `split_at` stays on `NdProducer` for now.** Relaxing a bound from `Strided` to `Layout` later is non-breaking for every caller, so nothing is lost by deferring; the options below are recorded for when that happens:
   - *Fast path.* (A) bound `Zip` by `Strided` — makes `Layout`-without-`Strided` useless in practice; (B) runtime capability query on `Layout` (`stride_of(axis) -> Option<isize>` + contiguity query, default `None`, `Strided` returns `Some`), asked once per `Zip`, per-element `layout.offset(index)` fallback — mdspan's `is_strided()`/`stride(r)`; (C) lockstep offset iterators per producer — loses the shared inner loop. When it is opened up, **B**: a query for optimisation *inside* a generic algorithm, versus traits for capabilities that change the API surface. Forward-compatible with a later `Layout::runs() -> (offset, len, stride)` generalisation that would give tiled layouts a fast path too.
   - *Sparse.* `Zip` means "visit every logical index" and a closure can't be assumed a no-op on fill, so storage-driven iteration is a different operation, not an optimisation of `Zip`. **Decided:** `Zip` over sparse works unchanged through the accessor (correct, O(shape)); storage-driven iteration is explicit (`stored_iter`, perhaps `Zip::stored(sparse).and(dense)`) and used by the library's own `sum`/`dot`/`map` when they know the fill algebra. No silent switch on `Acc::DENSE`.
   - *`split_at`.* Only parallel code needs it; if `Zip` is opened to `Layout`, move it to a sub-trait that `par_*` bounds by, so a producer needn't have axis-ops to be zipped.
2. **Whole-array mutation on sparse** (`fill`, `assign`, `map_inplace`, `+=`) type-checks on `Layout + DataMut` but densifies the storage. **Decided: allow and document** (scipy does the same, with a warning). The methods are correct, just expensive, and sparse users know what they're asking for; bounding them by `Acc::DENSE` would make the generic surface lie about what sparse supports.
3. **Reshape family**: today `into_shape*` *fails* when not contiguous; under the split it should either be bound by `Strided` or be `to_shape`-shaped (copy fallback). **Decided by the minimise-ripple rule:** bound by `Strided`, behaviour unchanged; `to_shape` stays the generic one.
4. **`push`/`append`** are dense-and-owned growth; sparse growth is entry insertion. **Decided:** dense-only; sparse gets its own (format-specific) surface.

### 3.5 Index type
- Should it be an associated type per layout? Sized or unsized for dynamic rank? Iterators like `indexed_iter` need something sized.
- How does `NdIndex<E>` (tuples, arrays, `usize`, `&[usize]`) map onto it?

### 3.6 Iteration order
- §1b puts "iterate stored locations in the layout's preferred order" on `Layout`. Still open: how callers ask for a *specific* order (logical C order for `iter()`, memory order for `map`/`fold`), and whether that is a second method, a parameter, or a separate trait.

### 3.7 Patterns, construction and other D-generic traits
- `dim() -> Pattern` (tuples) is used a lot downstream. Should that become a conversion on concrete shapes, or stay only in the `D` world?
- What replaces `IntoDimension`/`ShapeBuilder`/`StrideShape` for building shapes from user input, and how are order and strides specified?
- What do `NdProducer` (`type Dim`, `raw_dim`, `uget_ptr`), `SliceArg<D>` (`OutDim`) and broadcasting (`DimMax`→`RMax`) actually need from the new traits?

## 4. Transition

### 4.1 Sequence (decided)

Governed by the minimise-ripple rule (§3.4): each step compiles and passes the test suite on its own, changes nothing public until step 4, and is a separate PR series.

1. **Move the fields.** `Dimension` gains `type Layout: Strided` (sealed, so non-breaking). For `Ix0..IxDyn` it is `StridedLayout<D> { dim: D, strides: D }` — today's two fields moved into a struct, **exactly as today** (same representation, same two allocations for `IxDyn`, `raw_dim()` still free; performance work is step-independent and benchmarked separately). `ArrayParts { ptr, dim, strides }` becomes `{ handle: NonNull<A>, layout: D::Layout }`. The ~450 internal `.dim`/`.strides` accesses become a mechanical rename. *Done when:* no `dim`/`strides` field on `ArrayParts`, public API and benchmarks unchanged.
2. **Teach the struct the traits.** Implement `Layout`, the axis-ops trait and `Strided` (§1b) with the §1a output types on `StridedLayout<D>`, then migrate `impl_methods`/`impl_views`/iterators block by block from `D` arithmetic to layout methods, following the §3.4 table for which bound each block gets — *but* keeping today's bound wherever the table's weaker one is not needed yet (`Zip` stays `Strided`). *Done when:* `Dimension`'s stride/offset methods have no callers outside `StridedLayout`.
3. **Handle and accessor.** `RawData` gains `type Acc: Accessor` (§1c) with `Dense<A>` for every existing storage; `ArrayParts.handle` becomes `<S::Acc as Accessor>::Handle`; `ArrayRef<A, D>` becomes an alias over `ArrayRefBase<Dense<A>, D::Layout>`, `Deref` by the same cast. The `RawData` invariant from the mockup — the handle points at heap memory the storage never holds through a uniqueness-asserting type — is documented on the trait. *Gate:* benchmark the dense `Handle = NonNull<A>` path against step 2 on the `Ix1`/`Ix2`/`IxDyn` hot loops **before** any sparse storage exists; the step is not done until it is zero-cost.
4. **Additions, then the switch.** New layout families (`(Const<N>, Dyn)` tuples, §1a), sparse storage (§1c), as pure additions. Then the breaking release: `ArrayBase<S, L>` with `L` the layout directly, `Dimension`'s superseded surface deprecated, `Ix*` kept as aliases where they can be.

**Step 1 as actually done (fork branch `akern40/ndarray` `layout-spike`, off `upstream/refactor`):** `ArrayParts { ptr, layout: StridedLayout<D> }` with `StridedLayout<D> { dim: D, strides: D }` (`src/layout/strided.rs`, `pub(crate)` fields), not `D::Layout`. Reason: with an associated type, generic code over `D: Dimension` can't see that `D::Layout` *is* `StridedLayout<D>`, so none of the ~270 `parts.dim`/`parts.strides` sites would compile; the concrete struct is the exact field move. `Dimension::Layout` (if it's ever needed) belongs with step 2, once there are trait methods to call through it. `ptr` is not yet renamed `handle` (step 3's job, when the handle stops being a pointer). The crate-level `Layout` name is taken by the deprecated `LayoutBitset` alias, so the trait in step 2 needs either that alias removed first or a different name.

**Step 2 in progress (same branch).** `src/layout/traits.rs` now has `Layout` (shape/len/len_of, `offset`/`offset_checked`, `first_index`/`next_index`), `AxisOps` (`collapse_axis`/`slice_axis`/`invert_axis` returning the origin offset, `swap_axes`/`reverse_axes`/`merge_axes`, `insert_axis`/`remove_axis`/`permute_axes`/`into_dyn` with `Smaller`/`Larger`/`Dyn` output types) and `Strided` (strides, contiguity, `low_addr_offset`, `has_overlap`), all implemented for `StridedLayout<D>` by delegating to the existing `Dimension`/`dimension::` code; `ArrayBase`'s structural ops, `Baseiter` and the indexing paths call them. Findings so far:

- *In-place origin moves return an offset.* `collapse_axis`/`slice_axis`/`invert_axis` on the layout return the element offset and `LayoutRef` advances `ptr` by it. That is exactly the `Accessor::offset(handle, by)` contract from the mockup, so step 3 has one place per op to swap `ptr.offset(o)` for `acc.offset(handle, o)`.
- *Rank-changing ops take `&self`.* `insert_axis(&self)`/`remove_axis(&self)`/`permute_axes(&self)` rather than `self`: `self.with_layout(self.parts.layout.insert_axis(..))` moves `self` before evaluating the argument, and `Dimension` already builds the new value from a borrow. `into_dyn(self)` stays by value.
- *`NdIndex<D>` is stride-specific.* Its hidden methods are `index_checked(dim, strides)`/`index_unchecked(strides)`; everything now funnels through `StridedLayout::index_offset{,_checked}` so the trait is touched in one place when it becomes `L::Index`-based (§3.5).
- *`Dimension::Layout` was never needed* for step 2 either; `StridedLayout<D>` is the layout and `with_layout()` the internal constructor.

**Step 3 as actually done (same branch, `src/accessor.rs`).** `RawData::Acc: Accessor<Elem = Self::Elem>` with `Dense<A>` (`Handle = NonNull<A>`, `offset` = pointer offset) for all seven storages; `ArrayParts { handle, layout }`; `LayoutRefBase`/`ArrayRefBase`/`RawRefBase<Acc, D>` with `ArrayRef<A, D> = ArrayRefBase<Dense<A>, D>` etc. as aliases, so every existing `impl … for ArrayRef<A, D>` compiles unchanged (impls on a type alias are fine). Findings:

- *Variance.* With `parts: ArrayPartsSized<<S::Acc as Accessor>::Handle, D>` the projection makes `ArrayBase` invariant in `S`, and `ArrayView<'a, …>` stops being covariant in `'a` (`tests/variance.rs` fails to compile). Fix: the handle is a defaulted fourth parameter, `ArrayBase<S, D, A = S::Elem, H = <S::Acc as Accessor>::Handle>`, the same trick the existing `A` parameter already is. `ArrayRefBase<Acc, D>` still holds the projection, so `ArrayRef<&'a T, D>` is now invariant in `'a`; untested today, but the same `H` default would fix it if it matters.
- *Exactly 11 generic sites assumed the handle is a pointer*, and they sort cleanly: `RawDataClone::clone_with_ptr`/`clone_from_with_ptr` and the internal `from_data_ptr` become handle-typed (generic, no dense assumption); `DataOwned` (constructs from a `Vec`) and `RawDataSubst` (`MaybeUninit` reinterpretation) are dense by supertrait bound `Acc = Dense<Elem>`, which is the §3.4 "dense-only" row made precise; `as_mut_ptr`/`raw_view_mut` (on `S: RawDataMut`, no `Deref`) reach the pointer through the `AsRef<RawRef<A, D>>` cast.
- *Where the dense assumption now lives:* the `Deref`/`AsRef` casts in `impl_ref_types.rs` cast `ArrayParts<<S::Acc as Accessor>::Handle, D>` to the dense alias for any `S`. It is sound only because every `Acc` is `Dense` today. The honest `Target = ArrayRefBase<S::Acc, D>` is the next experiment: it will reject every call from a generic `impl<S: Data>` block into an `ArrayRef<A, D>`-only method, i.e. it measures how much of `impl_methods` must be re-homed onto `ArrayRefBase<Acc, D>` with `Layout`/`AxisOps`/`Strided` bounds before a non-dense storage can exist.
- *Benchmark gate (passed).* `cargo +nightly bench --bench bench1` on `upstream/refactor` vs `layout-spike` (ns/iter, same machine, sequential runs; noise is ±10% between runs of the *same* binary):

  | bench | refactor | layout-spike |
  |---|---:|---:|
  | `iter_sum_2d_regular` | 166 | 163 |
  | `iter_sum_1d_regular` | 184 | 175 |
  | `sum_2d_regular` | 106 | 103 |
  | `add_2d_regular` | 275 | 274 |
  | `add_2d_strided` | 1662 | 1714 |
  | `add_2d_broadcast_1_to_2` | 329 | 308 |
  | `add_2d_zip_cutout` | 446 | 331 |
  | `fold_sum_i32_2d_regular` | 210 | 163 |
  | `fold_sum_i32_2d_stride` | 1313 | 1094 |
  | `iadd_scalar_2d_regular` | 223 | 212 |
  | `iter_sum_2d_cutout` | 3254 | 2925 |
  | `iter_sum_2d_transpose_regular` | 3040 | 3647 |
  | `bench_row_iter` | 315 | 271 |
  | `mean_axis0` | 2028 | 1728 |
  | `into_dyn_ix3` | 38 | 31 |
  | `scalar_add_1` | 327 | 302 |

  The one apparent regression, `iter_sum_2d_transpose_regular` (+15–20%, reproducible, and bisecting to ~+8% after the pure field move of step 1 and ~+10% more after step 2 with no change to the hot code), disappears when both binaries are built with `-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6` (3.35k vs 3.29k): it is loop-alignment noise, not a cost of the design. Dense `uget`/`Zip`/iteration paths are unchanged in cost, as expected for a `NonNull<A>` handle with `#[inline(always)]` `offset`.

**Measured: the honest `Deref` target.** Changing `impl Deref for ArrayBase<S, D>` from `Target = ArrayRef<S::Elem, D>` (the dense alias) to `Target = ArrayRefBase<S::Acc, D>` — the type it would have to be before any non-dense `S` can exist — gives **258 compile errors** in the library, distributed: `impl_ops.rs` 213 (one macro family: `&a + &b`, `a += &b`, scalar ops, all `impl … for ArrayRef<A, D>` and used from generic `ArrayBase` code), `linalg/impl_linalg.rs` 24, `arraytraits.rs` 17 (`Index`, `PartialEq`, `IntoIterator`…), `lib.rs` 17, `free_functions.rs` 10, `impl_methods.rs` 7, `arrayformat.rs` 6, serde/approx 4, the rest ≤2 each. Reading: the §3.4 "move impls to `ArrayRefBase<Acc, L>` with the right bound" cost is real but bounded and lopsided — the operator macros and the trait impls (`Index`, `Debug`, `PartialEq`, `Add`…) are where it lands, not `impl_methods`, because on `refactor` the methods already live on the ref types and `ArrayBase` reaches them through the user's `Deref`, not from generic crate code. Each site resolves one of two ways: generalise the impl to `ArrayRefBase<Acc, D>` with a `Layout`/`AxisOps`/`Strided` bound (right for `Index`, `Debug`, `PartialEq`, iteration, elementwise ops), or bound it `Acc = Dense<A>` (BLAS, `as_slice`, `Vec` construction). That is the work item that sits between step 3 and step 4; it is not needed for steps 1–3 and was not done on the branch.

**Done: the honest `Deref` target (commit on `layout-spike`).** `ArrayBase<S, D>: Deref<Target = ArrayRefBase<S::Acc, D>>`, with `AsRef`/`AsMut`/`Borrow` likewise, the ref-type `Deref`/`AsRef` chains generic over `Acc`, every `impl LayoutRef<A, D>` block (13 of them: shape/strides accessors, in-place slicing, axis ops, `layout_impl`, `nrows`/`ncols`, dyn insert/remove) re-homed as `impl<Acc: Accessor, D> LayoutRefBase<Acc, D>` with origin moves through `Acc::offset`, and `RawData::_is_pointer_inbounds` taking the handle. Everything else that generic code reached through the old dense cast now carries an explicit `Acc = Dense<A>` bound: **126 bounds**, 98 in the library (free_functions/meshgrid 20, arraytraits 20, impl_ops 12, impl_methods 12, data_traits 11, arrayformat 6, approx 6, linalg 4, the rest 1–2 each) and 28 in tests, examples and `ndarray-rand`. Build, clippy `-D warnings`, fmt, all tests and doctests green; bench parity unchanged in kind (no hot path touched).

What this measured, beyond the count:

- *Did `Layout`/`AxisOps`/`Strided` suffice?* For everything on `LayoutRefBase` — yes, with no new trait methods: those blocks compile for any `Acc` because they never touch an element. Nothing on `ArrayRefBase` could be generalised yet, and the reason is uniform: every element-touching path goes through one of three dense-only primitives — `Baseiter`/`Iter` (hold `NonNull<A>` and deref it), `view()`/`view_mut()` (produce `ArrayBase<ViewRepr<&A>, D>`, a *dense* view storage), and `as_slice*`/BLAS (contiguous memory). So the next layer of the design is not more layout methods but two accessor-side additions: `Accessor::get(handle, offset) -> &Elem` / `get_mut` (so `Index`, `PartialEq`, `Debug`, `Hash`, `fold`/`map` and the iterators can be written once over `Acc`), and a per-accessor view storage (`Acc::View<'a>: Data<Acc = Acc>`, a GAT) so `view()`/`iter()` exist for non-dense arrays. `as_slice*`, `cast`, BLAS, `Vec` construction stay `Acc = Dense<A>` by design.
- *Downstream breakage is real and specific.* Any user function generic over `S: Data` that calls an `ArrayRef` method (`fn f<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) { a.sum() }`) stops compiling until it says `S: Data<Elem = f64, Acc = Dense<f64>>` — 28 such sites in our own tests/examples/`ndarray-rand`, including the `functions_and_traits` example whose whole point is that pattern. Until `Accessor::get` + the view GAT land and the `ArrayRefBase` methods are generic, this bound is required; after they land it becomes unnecessary for most methods but never harmful. That ordering (generalise the `ArrayRefBase` surface *before* flipping `Deref` in a release) is the lesson for the real migration; on the spike branch the flip came first so the cost could be counted.
- *Bounds that are dense by nature* and should stay so: `DataOwned` (`Vec` construction), `RawDataSubst` (`MaybeUninit` reinterpretation), `Data::to_shared`/`into_owned` (clone into a `Vec`), `as_ptr`/`as_slice*`/`raw_view*`/`into_shape*` (strided memory), BLAS and `dot`, `meshgrid`.

**Done: `Accessor::get`/`get_mut` and generic indexing (next commit on `layout-spike`).** `Accessor` gained `unsafe fn get<'a>(handle, offset) -> &'a Elem` / `get_mut` (dense: pointer offset + deref, `#[inline(always)]`). `get`, `get_mut`, `uget`, `uget_mut` moved from `impl ArrayRef<A, D>` to `impl<Acc: Accessor, D: Dimension> ArrayRefBase<Acc, D>`, and `Index`/`IndexMut` are implemented for `ArrayRefBase<Acc, D>`; the `Acc = Dense` bound on `Index`/`IndexMut for ArrayBase` is gone. All tests green. This confirms the pattern: a method generalises as soon as its only element access is "handle + offset → reference". Still dense: anything that iterates (`Baseiter`), produces a view (`ViewRepr<&A>`), or hands out a pointer/slice — the next step is the per-accessor view storage and a generic `Baseiter`.

**Next (in progress): generic views without a GAT.** MSRV is 1.64 and GATs need 1.65, but a GAT isn't needed: `ViewRepr<A>` is already a `PhantomData` marker whose only job is to carry a lifetime/mutability (`ViewRepr<&'a A>`, `ViewRepr<&'a mut A>`). Make it carry the *accessor* instead: `impl<'a, Acc: Accessor> RawData for ViewRepr<&'a Acc> { type Elem = Acc::Elem; type Acc = Acc; }` (and `&'a mut Acc` for `RawDataMut`), with `ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a Dense<A>>, D>`. Then `view()`/`view_mut()`, `iter()`/`iter_mut()`, `Baseiter`/`Iter` generalise to `ArrayRefBase<Acc, D>` as `ArrayBase<ViewRepr<&'a Acc>, D>` — one family of view storages for every accessor, no new trait item. Cost: the public *shape* of the alias changes (`ViewRepr<&A>` → `ViewRepr<&Dense<A>>`), which only matters to code that names `ViewRepr` directly. `Iter<'a, A, D>` has the dense slice fast path (`ElementsRepr::Slice`); the generic iterator is the `Counted` arm only, so `Iter` becomes `IterBase<'a, Acc, D>` with the slice arm gated on `Acc = Dense`, or the slice fast path becomes an accessor query (`Accessor::as_slice(handle, len) -> Option<&[Elem]>`).

**Finding while starting generic views: `Data` is dense by signature.** `Data::into_owned`/`try_into_owned_nocopy` return `Array<Elem, D>` = `ArrayBase<OwnedRepr<Elem>, D>`, and `ArrayBase: Deref` requires `S: Data`. So a view storage `ViewRepr<&'a Acc>` can implement `RawData` generically but not `Data` — and without `Data` it can't deref, so a generic `view()` would be useless. The owned counterpart of a storage family is an accessor-level fact, so the fix is on `Accessor`: `type Owned: RawData<Elem = Self::Elem, Acc = Self>` (`Dense<A>::Owned = OwnedRepr<A>`) and `fn into_owned<D>(view: ArrayBase<ViewRepr<&Self>, D>) -> ArrayBase<Self::Owned, D>` — the accessor decides how to materialise (dense: today's `to_owned`; sparse: copy the entries). `Data::into_owned` then returns `ArrayBase<<Self::Acc as Accessor>::Owned, D>`, which normalises to `Array<A, D>` for every existing dense storage, so dense callers don't change. Same for `to_shared`/`ArcArray` later (`Acc::Shared`). Doing this on the spike.

**Done: generic views and generic base iteration (`layout-spike` commits `7001e9c8`, `11d07160`, `a25c68c1`).** `ViewRepr<&'a Acc>`/`ViewRepr<&'a mut Acc>` with the storage traits written once over `Acc`; `Accessor::Owned` + `Accessor::into_owned`, and `Data::into_owned`/`try_into_owned_nocopy` return `ArrayBase<<Self::Acc as Accessor>::Owned, D>` (normalises to `Array<A, D>` for dense, no caller changed). `view()`/`view_mut()` live on `ArrayRefBase<Acc, D>` via `from_handle`/`from_handle_mut`. `Baseiter<Acc, D>` holds `Acc::Handle`, steps with `Accessor::offset`, and `ElementsBase(Mut)` dereference with `Accessor::get(_mut)`; `into_base_iter` exists for any accessor view. `Iter`/`IterMut` (public, with the slice fast path) are still `Dense<A>`-only — generalising them means `IterBase<'a, Acc, D>` with `Iter` as the dense alias, same move as `ArrayRef`. All green (build, clippy -D warnings, fmt, tests, doctests). Bench after the iterator change: `iter_sum_2d_regular` 163, `iter_sum_1d_regular` 164, `fold_sum_i32_2d_regular` 163, `iter_sum_2d_cutout` 2896, `iter_sum_2d_transpose_regular` 3438 ns — parity with the step-3 numbers (the transpose bench is inside its known alignment band). Two small trait costs surfaced: `Accessor: Sized` (needed to name `ViewRepr<&Self>` in the trait) and `Handle: Debug` (iterators derive `Debug`). Send/Sync for the generic element iterators follow today's rule on `Acc::Elem` rather than on the handle.

### 4.2 Still open

- **Step 1 details:** what `raw_dim()`, `dim()` and the slice-returning `shape()`/`strides()` cost once they read through `D::Layout` (expected: nothing, same fields). Do the `Dim` types also implement the new traits directly, or only via `StridedLayout<D>`?
- **The switch:** what happens to hand-written `Array<A, Ix2>`, `Ix2(..)` values, and `raw_dim()`? How long is the deprecation cycle? What's the order for downstream crates (ndarray-linalg, ndarray-stats, rust-numpy; usage counts are in `notes/inventory-scratch.md`)?
- **Names:** what should `shape_builder::Shape` be renamed to? Removing `crate::Layout` is a breaking change; when does it go?
- **Sealing:** `Dimension`, `NdProducer` and `SliceArg` are sealed, so adding supertraits or associated types doesn't break implementors. It can still break callers when a method moves between traits on concrete types (as `ndim` did in #1575). Watch for this at every step.
- **Performance:** check before and after each storage change, especially `IxDyn` and the `Ix1`/`Ix2` hot loops.

## 5. Inventory: today's API and which concept it relies on

This doesn't decide replacements. It's the input to §3.4 and §3.7.

| Today | Relies on |
|---|---|
| `Dimension::{NDIM, ndim}`, `DimAdd`, `DimMax` | rank |
| `Dimension::{size, size_checked, slice, equal, zeros}`, `IxDynImpl`, `Pattern`/`into_pattern`, `as_array_view` | shape (+ construction) |
| `default_strides`, `fortran_strides`, `strides_equivalent`, `is_contiguous`, `_fastest_varying_stride_order`, `min/max_stride_axis` | shape + strides |
| `stride_offset(_checked)`, `NdIndex<E>`, `first_index`/`next_for(_f)` | index + layout |
| `slice_mut`, `set_last_elem`, `IndexMut<usize>`, arithmetic supertraits | mutation (internal shape maths) |
| `insert_axis`, `try_remove_axis`, `RemoveAxis`, `Smaller`/`Larger`, `into_dyn`, `from_dimension` | rank-changing operations |
| `IntoDimension`, `ShapeBuilder`, `ShapeArg`, `StrideShape`, `shape_builder::Shape` | construction from user input |
| `LayoutRef::{dim, raw_dim, shape, strides, stride_of, len_of}` | shape/strides access |
| `index_axis`, lanes/rows/columns, `axis_iter`, `fold_axis`/`map_axis`, `insert_axis`/`remove_axis` methods, `push` | rank-changing operations on strided layouts |
| `NdProducer`, `Zip`, `SliceArg`, `stack`/`concatenate`, broadcasting | layout + rank operations |

The most-used compatibility surface downstream: `D::Smaller`/`RemoveAxis`, `Pattern`/`dim()`, `IntoDimension`, and the `shape()`/`strides()` slices.

### 5.1 What doesn't survive the switch (against `upstream/refactor`)

Today `D` plays three roles at once: the *type* parameter of `ArrayBase` (rank/family tag), a *shape value* (`raw_dim()`, `Ix2(3, 4)` as a constructor argument) and an *index value* (`a[Ix2(1, 2)]`, `NdIndex<D>`), plus the internal strides representation. The layout design keeps only the first (as `L`); shape values move to the layout's shape type, index values to `L::Index` (§3.5), strides inside `StridedLayout`. Everything below is a consequence of that split. The §3.4 audit says which *bound* a method needs; this list says which methods lose their *meaning*.

**Gone: no counterpart, the concept doesn't exist after the switch.**

| Today | Why it goes | What takes the role (if any) |
|---|---|---|
| `Dimension` as public API: `default_strides`, `fortran_strides`, `first_index`/`next_for(_f)`, `stride_offset(_checked)`, `is_contiguous`, `strides_equivalent`, `_fastest_varying_stride_order`, `min/max_stride_axis(strides)`, `slice()`/`slice_mut()`, `last_elem`/`set_last_elem`, `zeros`, `equal`, `size(_checked)`, `as_array_view(_mut)`, `from_dimension`, `into_pattern`, `insert_axis`/`try_remove_axis` | internal strided shape maths on a bare shape; a layout owns dim *and* strides, so there is nothing to call these on | private methods of `StridedLayout<_>` |
| `DimensionExt`, `IxDynImpl`, `Dim<I>` + `Dim(..)`, `Convert` | representation of `D` | none public |
| `RemoveAxis`, `DimAdd`, `DimMax` (traits on `D`) | per-`D` traits | `Smaller`/`Larger`/`Dyn` on the layout (§1a); rank-level `RAdd`/`RMax` for `stack`/broadcast arithmetic. `D: RemoveAxis` bounds are common downstream — a top compat item |
| `IntoDimension`, `Dimension::Pattern` | the tuple ↔ `Dim` bridge | §3.7 open; see the `dim()` decision below |
| `raw_dim() -> D` | there is no `D` value to return | shape by value: either `shape()` changes return type or a new accessor (decision below) |
| `dim() -> D::Pattern` | `Pattern` is a `Dimension` associated type | decision below |
| `into_dimensionality::<D2>()` | cross-`D` conversion with runtime rank check | still needed (dyn → static): a fallible layout conversion (`TryFrom<DynLayout> for StridedLayout<2>` or `into_layout::<L2>()`); the name goes |
| `Ix0()..Ix6()`, `IxDyn(&[..])`, `Dim(..)` *value* constructors | build a `D` value for shape or index use | shape use → shape constructors / tuples; index use → `L::Index`. The *type* aliases `Ix0..IxDyn` stay as layout aliases (§4.1 step 4) |
| `into_shape`, `reshape`, `into_raw_vec` | already `#[deprecated]` since 0.16 | `into_shape_with_order`/`to_shape`, `into_raw_vec_and_offset` |

**Changed signature, same name and meaning.**

- Every `E: IntoDimension` / `Sh: ShapeBuilder` parameter and `-> ArrayBase<S, E::Dim>` result: `to_shape`, `into_shape_with_order`, `into_shape_clone`, `broadcast`, `windows(_with_stride)`, `exact_chunks(_mut)`, `permuted_axes`, `from_elem`, `from_shape_fn`, `from_shape_simple_fn`, `from_shape_vec(_unchecked)`, `build_uninit`, `zeros`/`ones`/`default`/`uninit`, `ArrayView::from_shape(_ptr)`, `RawArrayView::from_shape_ptr`, `indices(shape)`. The argument becomes "something that builds a layout" and the result type follows it (§1a). `ShapeBuilder`/`Shape`/`StrideShape`/`ShapeArg`, `.f()`, `.strides(..)` are replaced by whatever that construction API is (§3.7).
- `Index`/`IndexMut`, `get(_mut)`, `uget(_mut)`, `get_ptr(_mut)`: `I: NdIndex<D>` → indexed by `L::Index` or `NdIndex<L>` (§3.5).
- `slice*`, `multi_slice_*`: `SliceArg<D>` → `SliceArg<L>`, `SliceInfo<T, Din, Dout>` over layouts; `s![]` unchanged.
- `assign<E: Dimension>`, `zip_mut_with`, `Zip::and_broadcast`, arithmetic ops: `E: Dimension`/`DimMax` → `L2: Layout` with rank-level broadcasting.
- `into_dyn()` → `-> ArrayBase<S, L::Dyn>`. `ndim()` → rank. `NdProducer::Dim` → a layout (still `Strided`, §3.4).
- `strides()`, `stride_of`, `axes()`: exist only for `L: Strided` (same today in practice).
- Dense-only (`as_ptr`, `as_slice*`, `raw_view*`, `into_raw_vec_and_offset`, `cast`, `split_complex`, `cell_view`, `assume_init`, `push`/`append`/`reserve`, BLAS): still exist, but on `ArrayRef<Dense<A>, L>` rather than every `ArrayRef` (§3.4).

**Not touched by this design.** The `Axis(usize)` newtype and the ~40 `*_axis`/`axis_*` methods (`index_axis*`, `slice_axis*`, `collapse_axis`, `axis_iter*`, `lanes`, `sum_axis`, `fold_axis`, `map_axis`, `insert_axis`/`remove_axis`, `split_at`, `stack`/`concatenate`, …) survive unchanged here; the axis-handle redesign is a separate, undecided track and must not be conflated with the layout switch. In-place rank-changing ops on `IxDyn` (`insert_axis_inplace`, `index_axis_inplace`) stay, on the dyn layout only, as today.

**Survives as is.** Everything else — the whole `Layout` block of the audit (`len`, `is_empty`, `iter*`, `indexed_iter*`, `for_each`, `fold`, `map*`, `view*`, `to_owned`, `into_shared`, `into_owned`, `t`/`reversed_axes`/`swap_axes`, reductions, `dot`) plus the axis-ops block.

**Shape accessors: what the data says (open).** Counts of call sites (`grep`, 2026-10-19; ndarray = `upstream/refactor` src / tests+examples+benches; downstream = current masters):

| | ndarray src | ndarray tests | ndarray-stats | ndarray-linalg |
|---|---|---|---|---|
| `.shape()` | 107 | 107 | 31 | 17 |
| `.dim()` | 38 | 101 | 2 | 20 |
| `.raw_dim()` | 53 | 16 | 3 | 2 |
| `.ndim()` | 166 | 5 | 16 | 0 |
| `.len_of(` | 43 | 8 | 5 | 11 |
| `RemoveAxis` | — | — | 23 | 0 |

So `shape()` is the most-used accessor, not `dim()`. More telling is *what* `dim()` is used for: of the 101 test/example uses, 65 pass it straight into a constructor (`Array::zeros(a.dim())`, `from_shape_fn(a.dim(), …)`), 9 are `assert_eq!(a.dim(), (m, n))`, and only 12 unpack it (`let (m, n) = a.dim()`); ndarray-linalg is 4 / 7 / 4. That is, `dim()` is mostly used in `raw_dim()`'s role — *the shape as an owned value* — and tuples are just the convenient literal for it. So the two accessors answer one need, and the language question ("`dim` is overloaded; the operation is unpacking a shape") has a clean answer: there is a **shape value** (owned, usable as a constructor argument, comparable to a literal, unpackable to a tuple when the rank is static) and a **slice/trait view** of it for rank-generic code.

Decisions this leaves:

1. **What `shape()` returns.** `&[usize]` is not possible in general: a `(Const<3>, Dyn)` shape has no `[usize; 2]` in memory (`Const<3>` has no storage), so returning a slice would require the layout to keep a materialised shadow copy next to the typed shape — redundant state, and it makes `Const` pointless as a space saving. Options: (a) `shape() -> &L::Shape`, with the §2 `Shape` trait (`axis_len(k)`, `ndim`, `iter`) as the rank-generic reader, `as_slice()` only on shapes that have one (`StridedLayout<N>`, dyn), and `[usize; N]` by value for rank-static shapes; (b) keep `&[usize]` and accept the shadow copy on the const family only (`StridedLayout` stores `[usize; N]` anyway). Lean: (a) for the long run; (b) is a legitimate minimise-ripple choice for the transition since the const family doesn't exist yet. To be settled by the spike.
2. **The shape value and tuple unpacking.** The 65 "pass the shape to a constructor" sites want an owned shape value; `raw_dim()` is that today, under a `Dimension` name. The 12+4 unpack sites want `Into<(usize, usize)>` on rank-static shapes. Both live on the shape type, not on the array; `dim()` and `raw_dim()` as names go (`dim` is overloaded; the operation is unpacking a shape). Whether the array keeps a by-value `shape()` *and* a borrowed one, and what they're called, is a naming question for the spike. `ndarray-stats`' 23 `RemoveAxis` bounds confirm `D: RemoveAxis` → layout `Smaller` as the other top compat item.
