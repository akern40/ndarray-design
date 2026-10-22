use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::accessor::{Accessor, Dense, Sparse, SparseBuf, SparseHandle};

/// A storage names its accessor family. A family (dense, this sparse format)
/// is defined by its accessor; the ownership kinds are storages *for* that
/// accessor, which is what makes `&ArrayRef<Acc, L>` the common type across
/// owned, shared and view arrays of one family.
///
/// # Safety
/// The handle handed to the array must point at memory that stays valid and
/// at a fixed address while `self` is alive and unmoved-from, regardless of
/// moves of `self` (the stable-handle rule).
pub unsafe trait RawData
{
    type Acc: Accessor;
}

/// # Safety
/// The handle is safe to read through.
pub unsafe trait Data: RawData {}

/// # Safety
/// After `ensure_unique`, the handle is safe to write through and no other
/// array shares the storage.
pub unsafe trait DataMut: Data
{
    /// May replace the storage (CoW) and must then refresh the handle.
    fn ensure_unique(&mut self, h: &mut <Self::Acc as Accessor>::Handle);
}

// ---- dense ----------------------------------------------------------------

pub struct OwnedRepr<A>(pub(crate) Vec<A>);

impl<A> OwnedRepr<A>
{
    pub(crate) fn new(mut v: Vec<A>) -> (Self, NonNull<A>)
    {
        let h = NonNull::new(v.as_mut_ptr()).unwrap();
        (OwnedRepr(v), h)
    }
}

unsafe impl<A> RawData for OwnedRepr<A>
{
    type Acc = Dense<A>;
}
unsafe impl<A> Data for OwnedRepr<A> {}
unsafe impl<A> DataMut for OwnedRepr<A>
{
    fn ensure_unique(&mut self, _: &mut NonNull<A>) {}
}

pub struct OwnedArcRepr<A>(pub(crate) Arc<Vec<A>>);

unsafe impl<A> RawData for OwnedArcRepr<A>
{
    type Acc = Dense<A>;
}
unsafe impl<A> Data for OwnedArcRepr<A> {}
unsafe impl<A: Clone> DataMut for OwnedArcRepr<A>
{
    fn ensure_unique(&mut self, h: &mut NonNull<A>)
    {
        if Arc::get_mut(&mut self.0).is_some() {
            return;
        }
        // Clone-on-write moves the data; re-derive the handle from the new buffer.
        let off = unsafe { h.as_ptr().offset_from(self.0.as_ptr()) };
        let fresh = Arc::make_mut(&mut self.0);
        *h = unsafe { NonNull::new_unchecked(fresh.as_mut_ptr().offset(off)) };
    }
}

// ---- sparse ---------------------------------------------------------------

/// The buffer lives on the heap so the handle has a stable address while
/// `ArrayBase` moves; the `Vec`s inside may reallocate on insertion without
/// affecting the handle. It is held through a raw pointer rather than a `Box`
/// because moving a `Box` asserts uniqueness (a Stacked Borrows retag) and
/// invalidates every pointer derived from it — Miri catches the `Box` version.
/// This is the same reason ndarray's `OwnedRepr` is a raw `ptr/len/cap`
/// triple and not a `Vec`.
pub struct OwnedSparse<A>(NonNull<SparseBuf<A>>);

impl<A> OwnedSparse<A>
{
    pub(crate) fn new(buf: SparseBuf<A>) -> (Self, SparseHandle<A>)
    {
        let ptr = NonNull::new(Box::into_raw(Box::new(buf))).unwrap();
        (OwnedSparse(ptr), SparseHandle { buf: ptr, base: 0 })
    }

    pub(crate) fn buf(&self) -> &SparseBuf<A>
    {
        unsafe { self.0.as_ref() }
    }
}

impl<A> Drop for OwnedSparse<A>
{
    fn drop(&mut self)
    {
        unsafe { drop(Box::from_raw(self.0.as_ptr())) }
    }
}

unsafe impl<A: Clone> RawData for OwnedSparse<A>
{
    type Acc = Sparse<A>;
}
unsafe impl<A: Clone> Data for OwnedSparse<A> {}
unsafe impl<A: Clone> DataMut for OwnedSparse<A>
{
    fn ensure_unique(&mut self, _: &mut SparseHandle<A>) {}
}

// ---- views, one per mutability, generic over the family ---------------------

pub struct ViewRepr<'a, Acc: Accessor>(pub(crate) PhantomData<&'a Acc::Handle>);
pub struct ViewMutRepr<'a, Acc: Accessor>(pub(crate) PhantomData<&'a mut Acc::Handle>);

unsafe impl<'a, Acc: Accessor> RawData for ViewRepr<'a, Acc>
{
    type Acc = Acc;
}
unsafe impl<'a, Acc: Accessor> Data for ViewRepr<'a, Acc> {}

unsafe impl<'a, Acc: Accessor> RawData for ViewMutRepr<'a, Acc>
{
    type Acc = Acc;
}
unsafe impl<'a, Acc: Accessor> Data for ViewMutRepr<'a, Acc> {}
unsafe impl<'a, Acc: Accessor> DataMut for ViewMutRepr<'a, Acc>
{
    fn ensure_unique(&mut self, _: &mut Acc::Handle) {}
}
