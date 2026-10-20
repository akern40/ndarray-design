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

### 1b. `Layout` and `Strided` are two traits, and the design targets include sparse and block layouts

Decided:

- **Design targets.** Besides today's dense strided arrays, the layout traits must accommodate (1) layouts with *computed* strides (C/F-contiguous and `const`-shape layouts that store no strides), (2) **block/tiled** layouts (Morton/Z-order, GPU and image tilings), and (3) **sparse** layouts. Not every operation has to be efficient, or even available, on every layout, but the traits must not make them unrepresentable. (Arrow's `Tensor` is strided and is not a separate target.)
- **`Layout` is the general contract and carries nearly all of `ArrayBase`:** the shape, index → location, the valid offset range (soundness), iteration over stored locations in the layout's preferred order, and the structural operations with output types from §1a (`index_axis → Smaller`, `insert_axis → Larger`, slicing, permutation, broadcasting). None of these need strides: `index_axis` on any layout is "remove slot `k` from the shape, pin coordinate `i` in the location function", and each family chooses its `Smaller` — the strided family is closed and returns itself, a block or sparse family may return a derived wrapper. Iteration belongs on `Layout` too: the strided impl walks pointers, a general impl walks indices. Fast paths are an implementation detail of each layout, not a property of the trait.
- **`Strided: Layout` is where strides are observable**, stored or computed: `strides()`, contiguity and `as_slice`, `is_standard_layout`, `from_shape_strides`, raw-view constructors, BLAS handoff. This is a short list, so the method-by-method audit (§3.4) is cheap.
- **Rejected:** a single strided `Layout` trait (rules out the design targets), and "`Layout` internal-only in v1" (pays the audit cost without the benefit).

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
- **Index → location for sparse.** `offset(index) -> isize` assumes every index has a stored element. Sparse layouts don't: an index maps to a stored offset *or* an implicit value. Options: `offset(index) -> Option<isize>` on `Layout` with dense layouts always `Some` (cost: every dense access pays a branch, unless dense layouts also implement a `Dense: Layout` marker with the infallible version); or a two-level design where `Layout` only promises *iteration* over `(index, offset)` pairs and random access is a sub-trait. The first is simpler; the second matches what sparse formats can actually do efficiently.
- **Closure under structural ops.** Which of `index_axis`, slicing, permutation and broadcasting must every layout support, and which may be unavailable (CSR can index rows cheaply, columns not at all)? If some are optional, are they sub-traits or `Option`-returning methods? Derived wrappers (`Sub<L>`, `Sliced<L>`) are always *possible*; the question is whether the trait requires them.
- **Minimal surface.** Probably: the shape, index → location, the offset range (soundness), and offset iteration. Does `shape()` return `&Self::Shape` (works for stored shapes and zero-sized `const` ones), or does `Layout` expose shape methods directly so shapes can be computed?
- **Audit.** Which of today's `ArrayBase`/`ArrayRef` methods need only `Layout`, and which need `Strided`? Expected: `Strided` for `as_slice*`, `strides()`, `is_standard_layout`, `from_shape_strides`, raw views, BLAS; `Layout` for everything else. §5 is the starting point; the result should become a table here.

### 3.5 Index type
- Should it be an associated type per layout? Sized or unsized for dynamic rank? Iterators like `indexed_iter` need something sized.
- How does `NdIndex<E>` (tuples, arrays, `usize`, `&[usize]`) map onto it?

### 3.6 Iteration order
- §1b puts "iterate stored locations in the layout's preferred order" on `Layout`. Still open: how callers ask for a *specific* order (logical C order for `iter()`, memory order for `map`/`fold`), and whether that is a second method, a parameter, or a separate trait.

### 3.7 Patterns, construction and other D-generic traits
- `dim() -> Pattern` (tuples) is used a lot downstream. Should that become a conversion on concrete shapes, or stay only in the `D` world?
- What replaces `IntoDimension`/`ShapeBuilder`/`StrideShape` for building shapes from user input, and how are order and strides specified?
- What do `NdProducer` (`type Dim`, `raw_dim`, `uget_ptr`), `SliceArg<D>` (`OutDim`) and broadcasting (`DimMax`→`RMax`) actually need from the new traits?

## 4. Open questions: transition

- **The `D` → layout associated type:** what bounds does it have? Do the `Dim` types also implement the new traits directly? What do `raw_dim()` and the like cost?
- **Storage swap:** can `ArrayParts` store `D::Layout` while the public API still uses `D`? `shape() -> &[usize]` and `strides() -> &[isize]` need the concrete transitional layouts to be backed by slices. Is that acceptable until the switch?
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
