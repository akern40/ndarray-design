# Mockup: sparse as a storage behind a handle + accessor (roadmap §1c)

The question this answers: can `ArrayBase<S, L>` keep its shape — `Deref` to a
two-generic `ArrayRef` by pointer cast, `S` as ownership, `L` as a *total*
index → offset map — once sparse arrays are a storage reached through an
accessor instead of a bare `NonNull<A>`?

Run `cargo test`; `cargo +nightly miri test` also passes (it caught one bug, below).

## What is here

| piece | file | notes |
|---|---|---|
| `Layout` | `layout.rs` | total `offset(index)`, `offset_range` (soundness), `is_unique`, inverse `index_of(offset)`, layout-order `indices()`. `AxisOps: Layout` has `reversed_axes`, `slice_axis`, `broadcast_axis`. One impl, `Strided<N>`. |
| `Accessor` | `accessor.rs` | `Elem`, `Handle: Copy`, `DENSE`, `offset(handle, by)`, `get`, `get_mut`, `fill`, `stored`. `Dense<A>` (handle = `NonNull<A>`), `Sparse<A>` (handle = pointer to a `SparseBuf` of sorted `(offset, value)` + a base offset). |
| storages | `storage.rs` | `RawData { type Acc }`, `Data`, `DataMut { ensure_unique(&mut handle) }`. `OwnedRepr`, `OwnedArcRepr` (dense); `OwnedSparse`; `ViewRepr<'a, Acc>` / `ViewMutRepr<'a, Acc>` generic over the family. |
| arrays | `array.rs` | `ArrayParts { handle, layout, _dst }`, `ArrayBase<S, L>`, `ArrayRef<Acc, L>` (repr(transparent) cast), `DenseRef<A, L>` = today's `ArrayRef<A, D>`. Methods generic over `Acc`: `get`, `get_mut`, `stored_iter`, `indexed_iter`, `sum`, `view`, `t`, `slice_axis(_mut)`, `broadcast_axis`. |

## What the tests show

- One generic surface: `fn trace<Acc: Accessor<Elem = f64>>(a: &ArrayRef<Acc, Strided<2>>)` runs on dense and sparse.
- Transpose/slice on a sparse array are layout-only: same handle address, strides swapped; a C-order-sorted buffer read through the transposed layout *is* CSC.
- Storage-driven iteration: `stored_iter` walks the buffer's entries and maps offsets back to indices with `index_of`, which also filters out entries outside a sliced view.
- Insertion through `&mut ArrayRef` (directly on the owned array and through a sliced `ArrayViewMut`); the entries `Vec` reallocates, the handle does not change.
- CoW on `ArcArray` re-derives the handle in `ensure_unique`.
- Broadcast views are `!is_unique()`, `index_of` refuses them, and no `&mut` path to them exists by construction.
- The offset-range invariant is checked once, in the owned constructors.

## Codegen

`dense_uget` (`lib.rs`), release, x86-64 — the accessor indirection costs nothing:

```asm
imulq   24(%rdi), %rdx
imulq   32(%rdi), %rcx
shlq    $3, %rcx
addq    (%rdi), %rcx
movsd   (%rcx,%rdx,8), %xmm0
retq
```

## Findings that belong in the roadmap

1. **The handle must be stable across moves of `ArrayBase` — and Stacked Borrows makes that stricter than "lives on the heap".** The first version held the sparse buffer in a `Box`; Miri rejected it: moving the `Box` into the storage struct is a `Unique` retag that invalidates the handle derived before the move. The owned storage has to hold the buffer through a raw pointer (`Box::into_raw` + `Drop`), exactly why ndarray's `OwnedRepr` is `ptr/len/cap` and not a `Vec`. Rule for the `RawData` docs: an owned storage never holds its buffer through a type that asserts uniqueness on move.
2. **Slicing needs `Accessor::offset(handle, by)`.** Today slicing bumps the data pointer. A sparse handle can't be bumped, so either the layout carries a base offset or the accessor knows how to advance a handle. The mockup does the latter (mdspan's `accessor.offset`), keeping `Strided` base-free and the dense path a pointer add; `AxisOps::slice_axis` returns `(layout, shift)`.
3. **`Layout` gains two members because of sparse: `index_of` and `is_unique`.** Both are properties of the map, not of storage, and both are only meaningful together: `index_of` is defined only on unique layouts. Whether they sit on `Layout` or on a sub-trait is still open (§3.4).
4. **O(nnz) is O(nnz of the buffer), not of the view.** `stored_iter` on a sliced sparse view walks every entry and filters. Making it O(nnz_view) needs storage/layout cooperation (row pointers only help when the layout order matches the sort order); accepted in §1c.
5. **The mockup's `is_unique`/`index_of` for `Strided` are sufficient-condition versions** (nested axes, non-negative strides). The real implementation needs the general overlap test ndarray already has for mutable constructors, and negative strides.

Not covered: shared *sparse* storage (CoW on `Arc<SparseBuf>` is the dense pattern plus handle refresh), `Zip` of two sparse arrays (format-specific, out of generic scope), sparse `index_axis`/`axis_iter` (layout-only, like `t`).
