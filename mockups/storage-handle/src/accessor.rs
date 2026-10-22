use std::marker::PhantomData;
use std::ptr::NonNull;

/// offset → element. The `Handle` is what an array carries instead of a bare
/// pointer; it must stay valid when the owning `ArrayBase` is moved (so it
/// points at heap memory, never into the `ArrayBase` itself) and is refreshed
/// by whatever mutates the storage (CoW clone, reallocation) — the two rules
/// `OwnedRepr`'s pointer already follows today.
///
/// # Safety
/// `get`/`get_mut`/`stored` are called only with offsets inside the range the
/// handle was constructed for (the layout/storage invariant), and `get_mut`
/// only through a `&mut ArrayRef`, which guarantees unique, writable storage
/// and a unique layout.
pub unsafe trait Accessor
{
    type Elem;
    type Handle: Copy;

    /// Dense means "every offset in range is stored", so iteration can be
    /// layout-driven. Sparse iteration is storage-driven via `stored`.
    const DENSE: bool;

    /// # Safety
    /// `by` must keep the handle inside the storage's range.
    unsafe fn offset(h: Self::Handle, by: isize) -> Self::Handle;
    /// # Safety
    /// `offset` in range; the storage outlives `'a`.
    unsafe fn get<'a>(h: Self::Handle, offset: isize) -> &'a Self::Elem;
    /// Sparse: a hole becomes a stored element (the fill value) first.
    /// # Safety
    /// As `get`, and the caller holds a `&mut ArrayRef` (unique storage, unique layout).
    unsafe fn get_mut<'a>(h: Self::Handle, offset: isize) -> &'a mut Self::Elem;
    /// The value a hole reads as; `None` for dense.
    /// # Safety
    /// The storage outlives `'a`.
    unsafe fn fill<'a>(h: Self::Handle) -> Option<&'a Self::Elem>;
    /// Stored `(offset relative to the handle, element)` pairs. Offsets may
    /// fall outside the current layout's range (e.g. a sliced view); the array
    /// filters them with `Layout::index_of`. Dense: unused.
    /// # Safety
    /// The storage outlives `'a`.
    unsafe fn stored<'a>(h: Self::Handle) -> Vec<(isize, &'a Self::Elem)>;
}

/// Today's arrays: the handle is the data pointer.
pub struct Dense<A>(PhantomData<A>);

unsafe impl<A> Accessor for Dense<A>
{
    type Elem = A;
    type Handle = NonNull<A>;
    const DENSE: bool = true;

    #[inline(always)]
    unsafe fn offset(h: NonNull<A>, by: isize) -> NonNull<A>
    {
        NonNull::new_unchecked(h.as_ptr().offset(by))
    }

    #[inline(always)]
    unsafe fn get<'a>(h: NonNull<A>, offset: isize) -> &'a A
    {
        &*h.as_ptr().offset(offset)
    }

    #[inline(always)]
    unsafe fn get_mut<'a>(h: NonNull<A>, offset: isize) -> &'a mut A
    {
        &mut *h.as_ptr().offset(offset)
    }

    unsafe fn fill<'a>(_: NonNull<A>) -> Option<&'a A>
    {
        None
    }

    unsafe fn stored<'a>(_: NonNull<A>) -> Vec<(isize, &'a A)>
    {
        unreachable!("dense iteration is layout-driven")
    }
}

/// "A `Vec` with holes": a virtual buffer of `span` slots of which only
/// `entries` are stored, sorted by offset. Sorted-by-C-order-offset *is* CSR;
/// read through a transposed layout it is CSC, with the same buffer.
#[derive(Debug, Clone)]
pub struct SparseBuf<A>
{
    pub span: usize,
    pub fill: A,
    pub entries: Vec<(usize, A)>,
}

impl<A> SparseBuf<A>
{
    pub fn nnz(&self) -> usize
    {
        self.entries.len()
    }
}

/// The sparse handle: a pointer to the (heap-stable) buffer plus a base offset,
/// because slicing can't advance a buffer pointer the way it advances `*A`.
pub struct SparseHandle<A>
{
    pub(crate) buf: NonNull<SparseBuf<A>>,
    pub(crate) base: isize,
}

impl<A> Clone for SparseHandle<A>
{
    fn clone(&self) -> Self
    {
        *self
    }
}
impl<A> Copy for SparseHandle<A> {}

pub struct Sparse<A>(PhantomData<A>);

unsafe impl<A: Clone> Accessor for Sparse<A>
{
    type Elem = A;
    type Handle = SparseHandle<A>;
    const DENSE: bool = false;

    unsafe fn offset(h: SparseHandle<A>, by: isize) -> SparseHandle<A>
    {
        SparseHandle { buf: h.buf, base: h.base + by }
    }

    unsafe fn get<'a>(h: SparseHandle<A>, offset: isize) -> &'a A
    {
        let buf = &*h.buf.as_ptr();
        let o = (h.base + offset) as usize;
        match buf.entries.binary_search_by_key(&o, |e| e.0) {
            Ok(i) => &buf.entries[i].1,
            Err(_) => &buf.fill,
        }
    }

    unsafe fn get_mut<'a>(h: SparseHandle<A>, offset: isize) -> &'a mut A
    {
        let buf = &mut *h.buf.as_ptr();
        let o = (h.base + offset) as usize;
        let i = match buf.entries.binary_search_by_key(&o, |e| e.0) {
            Ok(i) => i,
            Err(i) => {
                // May reallocate `entries`; the Box address, and so the handle, is unchanged.
                buf.entries.insert(i, (o, buf.fill.clone()));
                i
            }
        };
        &mut buf.entries[i].1
    }

    unsafe fn fill<'a>(h: SparseHandle<A>) -> Option<&'a A>
    {
        Some(&(*h.buf.as_ptr()).fill)
    }

    unsafe fn stored<'a>(h: SparseHandle<A>) -> Vec<(isize, &'a A)>
    {
        let buf = &*h.buf.as_ptr();
        buf.entries.iter().map(|(o, a)| (*o as isize - h.base, a)).collect()
    }
}
