# Roadmap: from `Dimension` to `Layout` (draft)

This is mostly a list of open design questions. Only §1 has been decided. §2 is the one next step that's concrete. Everything after that is still undecided, and the aim is to answer each question before writing the code that depends on it.

Guiding principle: introduce **minimal traits that are clearly right**, then grow them as real needs come up. Having less functionality than today's API is fine. Locking in a bad decision isn't.

## 1. Decided

- **Goal:** `ArrayParts` stores a layout instead of explicit `dim`/`strides`. `Dimension` is the integration point during migration and reaches the layout through an **associated type**. A later breaking switch deprecates the `D`-based APIs.
- **Names:** the names on the `rust-ndarray/ndarray@refactor` branch are canonical (`Rank`, `ConstRank`, `DynRank`, `RAdd`, `RMax`, `Ranked`).
- **`Rank` only encodes the number of axes.** `Ranked` connects it to the runtime `ndim()`. `Rank` doesn't know about shapes, strides, indices or any representation.
- **No `Cow` in the traits.** Shape and stride access goes through iterators (or `&` to stored data).
- **Shapes are read-only, with no public safe API for mutating them.** Changing a shape almost always requires checking invariants about the strides and the allocation. So any change happens through operations that keep those invariants, never through "set axis length".
- **Name collisions** (`crate::Shape` from `shape_builder`, and `crate::Layout` as the deprecated alias of `LayoutBitset`) are resolved by deprecation/renaming before the new names reach the crate root.
- **Landed so far:** the reference types (#1440, #1532), the `LayoutBitset` rename, `Rank` (#1568) and `Ranked` (#1575). In review: #1621 (`Ranked` in the prelude) and #1622 (lint fixes).

## 2. Next step: a minimal `Shape`

The `shape-trait` branch has `Shape: Ranked` with `axis_len`, `iter`, and provided methods `size_checked`, `size_bytes_checked` and `iter_isize`. Questions to settle before it lands:

- Should `axis_len` take `usize` or `Axis`? `Axis` matches the public API.
- Does `iter_isize` belong here? It's an offset/strides concern.
- Which bounds should it have (`Clone`, `Eq`, `Debug`, `Send`, `Sync`)? Keep them minimal and add them when a user of the trait needs them.
- How should it connect to `Dimension`? Either `Dimension: Shape` (the `Dim` types already *are* shapes, so this is cheap and doesn't commit to a representation) or an associated `Dimension::Shape` (which commits to a representation now).
- **Leave out on purpose:** mutation, construction, rank-changing operations, and pattern/tuple conversion.

## 3. Open questions: trait design

### 3.1 Where do rank-changing operations live?
These are `index_axis`, `insert_axis`, `remove_axis`, `into_dyn`, broadcasting and `stack`.
- (a) The type being changed has an associated output type (like today's `Smaller`/`Larger`, but on the shape/layout trait, or as per-operation traits).
- (b) A separate "representation family" trait that maps a rank to a concrete type (e.g. a GAT `Of<R: Rank>`), kept *outside* `Rank`.
- (c) Rank-changing operations always produce a general-purpose representation.
- Rejected: `Rank` as a type map, because it puts the dependency the wrong way round.
- Not yet answered: what happens to partially-`const` shapes when you remove a `const` axis, or insert an axis next to one?

### 3.2 How do shapes and strides change at all?
There's no public safe setter, but slicing, collapsing, inverting, permuting and reshaping all have to happen somehow.
- What's the smallest set of invariant-preserving operations? Are they on the layout, or free functions over concrete types?
- Does the unsafe "raw mutation" surface need to be a trait so generic internal code can use it, or can it live on each concrete representation?
- How do partially-mutable representations (some axes `const`) take part, or do they opt out of some operations?
- Constructors should check (shape, strides, allocation length) together. Where does that check live?

### 3.3 Strides
- Is `Strides` a standalone trait, or does it only exist inside a strided layout? Does anything need generic strides outside a layout?
- Strides should be signed `isize`, which also gets rid of today's practice of storing strides as `usize` and reinterpreting them as `isize`.
- Should strides ever be `const`?

### 3.4 Layout
- What's the minimal surface? Probably: the shape, the offset of an index, and the offset range (needed for soundness). Does `shape()` return `&Self::Shape` (works for stored shapes and for zero-sized `const` ones), or does `Layout` expose shape methods directly so shapes can be computed?
- How does `Strided: Layout` relate to non-strided layouts? Make sure nothing in `Layout` assumes strides.
- Which of today's methods only need `Layout`, and which need `Strided`? This needs a method-by-method audit (§5 is the starting point).

### 3.5 Index type
- Should it be an associated type per layout? Sized or unsized for dynamic rank? Iterators like `indexed_iter` need something sized.
- How does `NdIndex<E>` (tuples, arrays, `usize`, `&[usize]`) map onto it?

### 3.6 Iteration order and non-strided layouts
- Should iteration order be part of `Layout` or something separate? It can probably wait, as long as `Layout` doesn't rule it out.

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
