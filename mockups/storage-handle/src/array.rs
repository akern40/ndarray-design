use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::accessor::{Accessor, Dense, Sparse, SparseBuf};
use crate::layout::{AxisOps, Layout};
use crate::storage::{Data, DataMut, OwnedArcRepr, OwnedRepr, OwnedSparse, RawData, ViewMutRepr, ViewRepr};

/// Handle + layout; sized inside `ArrayBase`, unsized (zero-length tail)
/// inside `ArrayRef` so that it can only ever be reached by reference. Same
/// trick as ndarray's `ArrayParts`, with `ptr: NonNull<A>` replaced by the handle.
pub struct ArrayParts<Acc: Accessor, L, T: ?Sized>
{
    handle: Acc::Handle,
    layout: L,
    _dst: T,
}

type ArrayPartsSized<Acc, L> = ArrayParts<Acc, L, [usize; 0]>;
type ArrayPartsUnsized<Acc, L> = ArrayParts<Acc, L, [usize]>;

pub struct ArrayBase<S: RawData, L>
{
    data: S,
    parts: ArrayPartsSized<S::Acc, L>,
}

/// The common reference type of one accessor family. Two generics, as today:
/// the element type is `Acc::Elem`.
#[repr(transparent)]
pub struct ArrayRef<Acc: Accessor, L>(ArrayPartsUnsized<Acc, L>);

/// Today's `ArrayRef<A, D>`.
pub type DenseRef<A, L> = ArrayRef<Dense<A>, L>;

pub type Array<A, L> = ArrayBase<OwnedRepr<A>, L>;
pub type ArcArray<A, L> = ArrayBase<OwnedArcRepr<A>, L>;
pub type SparseArray<A, L> = ArrayBase<OwnedSparse<A>, L>;
pub type ArrayView<'a, Acc, L> = ArrayBase<ViewRepr<'a, Acc>, L>;
pub type ArrayViewMut<'a, Acc, L> = ArrayBase<ViewMutRepr<'a, Acc>, L>;

#[derive(Debug, PartialEq, Eq)]
pub enum ShapeError
{
    OffsetRangeExceedsStorage,
    DuplicateEntry,
}

// ---- Deref: the cast --------------------------------------------------------

impl<S: Data, L> Deref for ArrayBase<S, L>
{
    type Target = ArrayRef<S::Acc, L>;

    fn deref(&self) -> &Self::Target
    {
        let parts: &ArrayPartsUnsized<S::Acc, L> = &self.parts;
        // SAFETY: ArrayRef is repr(transparent) over ArrayPartsUnsized.
        unsafe { &*(parts as *const ArrayPartsUnsized<S::Acc, L> as *const ArrayRef<S::Acc, L>) }
    }
}

impl<S: DataMut, L: Layout> DerefMut for ArrayBase<S, L>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        self.data.ensure_unique(&mut self.parts.handle);
        // Invariant of every constructor of a mutable array, not a runtime check
        // in the real thing; asserted here to make the rule visible.
        debug_assert!(self.parts.layout.is_unique(), "&mut ArrayRef over a non-unique layout");
        let parts: &mut ArrayPartsUnsized<S::Acc, L> = &mut self.parts;
        unsafe { &mut *(parts as *mut ArrayPartsUnsized<S::Acc, L> as *mut ArrayRef<S::Acc, L>) }
    }
}

// ---- constructors (owned kinds; this is where the offset-range invariant is checked) ----

fn check_span<L: Layout>(layout: &L, span: usize) -> Result<(), ShapeError>
{
    match layout.offset_range() {
        Some((lo, hi)) if lo < 0 || hi as usize >= span => Err(ShapeError::OffsetRangeExceedsStorage),
        _ => Ok(()),
    }
}

impl<A, L: Layout> Array<A, L>
{
    pub fn from_shape_vec(layout: L, v: Vec<A>) -> Result<Self, ShapeError>
    {
        check_span(&layout, v.len())?;
        let (data, handle) = OwnedRepr::new(v);
        Ok(ArrayBase { data, parts: ArrayParts { handle, layout, _dst: [] } })
    }

    pub fn into_shared(self) -> ArcArray<A, L>
    {
        let ArrayBase { data, parts } = self;
        // The Vec's heap block is unchanged by moving the Vec into the Arc.
        ArrayBase { data: OwnedArcRepr(Arc::new(data.0)), parts }
    }
}

impl<A: Clone, L: Layout> Clone for ArcArray<A, L>
{
    fn clone(&self) -> Self
    {
        ArrayBase {
            data: OwnedArcRepr(Arc::clone(&self.data.0)),
            parts: ArrayParts { handle: self.parts.handle, layout: self.parts.layout.clone(), _dst: [] },
        }
    }
}

impl<A: Clone, L: Layout> SparseArray<A, L>
{
    pub fn from_entries(layout: L, fill: A, entries: Vec<(L::Index, A)>) -> Result<Self, ShapeError>
    {
        let span = layout.offset_range().map_or(0, |(_, hi)| hi as usize + 1);
        let mut entries: Vec<(usize, A)> = entries
            .into_iter()
            .map(|(i, a)| (layout.offset(i) as usize, a))
            .collect();
        entries.sort_by_key(|e| e.0);
        if entries.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(ShapeError::DuplicateEntry);
        }
        let (data, handle) = OwnedSparse::new(SparseBuf { span, fill, entries });
        Ok(ArrayBase { data, parts: ArrayParts { handle, layout, _dst: [] } })
    }

    pub fn nnz(&self) -> usize
    {
        self.data.buf().nnz()
    }
}

// ---- methods generic over the accessor --------------------------------------

impl<Acc: Accessor, L: Layout> ArrayRef<Acc, L>
{
    pub fn layout(&self) -> &L
    {
        &self.0.layout
    }

    pub fn shape(&self) -> &[usize]
    {
        self.0.layout.shape()
    }

    pub fn is_unique(&self) -> bool
    {
        self.0.layout.is_unique()
    }

    /// # Safety
    /// `index` must be in bounds.
    #[inline]
    pub unsafe fn uget(&self, index: L::Index) -> &Acc::Elem
    {
        Acc::get(self.0.handle, self.0.layout.offset(index))
    }

    fn in_bounds(&self, index: L::Index) -> bool
    where L::Index: AsRef<[usize]>
    {
        index.as_ref().iter().zip(self.shape()).all(|(i, n)| i < n)
    }

    pub fn get(&self, index: L::Index) -> Option<&Acc::Elem>
    where L::Index: AsRef<[usize]>
    {
        self.in_bounds(index).then(|| unsafe { self.uget(index) })
    }

    /// Only reachable through `&mut ArrayRef`, i.e. unique writable storage
    /// over a unique layout. On a sparse hole this inserts.
    pub fn get_mut(&mut self, index: L::Index) -> Option<&mut Acc::Elem>
    where L::Index: AsRef<[usize]>
    {
        self.in_bounds(index)
            .then(|| unsafe { Acc::get_mut(self.0.handle, self.0.layout.offset(index)) })
    }

    pub fn fill_value(&self) -> Option<&Acc::Elem>
    {
        unsafe { Acc::fill(self.0.handle) }
    }

    /// Stored elements with their indices. Dense: layout-driven, every
    /// element. Sparse: storage-driven, O(nnz of the underlying buffer),
    /// filtered to this layout through `index_of`.
    pub fn stored_iter(&self) -> Box<dyn Iterator<Item = (L::Index, &Acc::Elem)> + '_>
    {
        let h = self.0.handle;
        let layout = &self.0.layout;
        if Acc::DENSE {
            Box::new(layout.indices().map(move |i| (i, unsafe { Acc::get(h, layout.offset(i)) })))
        } else {
            Box::new(
                unsafe { Acc::stored(h) }
                    .into_iter()
                    .filter_map(move |(o, a)| layout.index_of(o).map(|i| (i, a))),
            )
        }
    }

    /// Every logical element, in layout order; total for dense and sparse.
    pub fn indexed_iter(&self) -> impl Iterator<Item = (L::Index, &Acc::Elem)> + '_
    {
        let h = self.0.handle;
        let layout = &self.0.layout;
        layout.indices().map(move |i| (i, unsafe { Acc::get(h, layout.offset(i)) }))
    }

    pub fn view(&self) -> ArrayView<'_, Acc, L>
    {
        ArrayBase {
            data: ViewRepr(PhantomData),
            parts: ArrayParts { handle: self.0.handle, layout: self.0.layout.clone(), _dst: [] },
        }
    }

    pub fn view_mut(&mut self) -> ArrayViewMut<'_, Acc, L>
    {
        ArrayBase {
            data: ViewMutRepr(PhantomData),
            parts: ArrayParts { handle: self.0.handle, layout: self.0.layout.clone(), _dst: [] },
        }
    }
}

impl<Acc: Accessor<Elem = f64>, L: Layout> ArrayRef<Acc, L>
{
    /// Exact for dense and sparse: stored values plus the fill for every hole.
    pub fn sum(&self) -> f64
    {
        let mut n = 0usize;
        let stored: f64 = self.stored_iter().map(|(_, a)| { n += 1; *a }).sum();
        stored + self.fill_value().map_or(0.0, |f| f * (self.0.layout.size() - n) as f64)
    }
}

// ---- structural ops: layout-only, so they cost the same for every family ----

impl<Acc: Accessor, L: AxisOps> ArrayRef<Acc, L>
{
    fn with_layout(&self, layout: L, shift: isize) -> ArrayView<'_, Acc, L>
    {
        ArrayBase {
            data: ViewRepr(PhantomData),
            parts: ArrayParts { handle: unsafe { Acc::offset(self.0.handle, shift) }, layout, _dst: [] },
        }
    }

    pub fn t(&self) -> ArrayView<'_, Acc, L>
    {
        self.with_layout(self.0.layout.reversed_axes(), 0)
    }

    pub fn slice_axis(&self, axis: usize, start: usize, len: usize) -> ArrayView<'_, Acc, L>
    {
        let (l, shift) = self.0.layout.slice_axis(axis, start, len);
        self.with_layout(l, shift)
    }

    /// Slicing preserves uniqueness, so a mutable view is fine.
    pub fn slice_axis_mut(&mut self, axis: usize, start: usize, len: usize) -> ArrayViewMut<'_, Acc, L>
    {
        let (layout, shift) = self.0.layout.slice_axis(axis, start, len);
        ArrayBase {
            data: ViewMutRepr(PhantomData),
            parts: ArrayParts { handle: unsafe { Acc::offset(self.0.handle, shift) }, layout, _dst: [] },
        }
    }

    /// Never mutable: the result is not unique.
    pub fn broadcast_axis(&self, axis: usize, len: usize) -> Option<ArrayView<'_, Acc, L>>
    {
        self.0.layout.broadcast_axis(axis, len).map(|l| self.with_layout(l, 0))
    }
}

impl<A: Clone, L: Layout> ArrayRef<Sparse<A>, L>
{
    pub fn stored_count(&self) -> usize
    {
        self.stored_iter().count()
    }

    /// For the tests: where the handle points.
    pub fn handle_addr(&self) -> usize
    {
        self.0.handle.buf.as_ptr() as usize
    }
}
