use core::fmt::Debug;
use std::{marker::PhantomData, mem, ptr::NonNull, sync::Arc};

use super::{
    ownership::{ArcOwner, VecOwner},
    storage::NonNullStorage,
    Backend,
};

pub struct VecBackend<T> {
    phantom: PhantomData<T>,
}

unsafe impl<T: Debug> Backend for VecBackend<T> {
    type Ref = NonNull<T>;

    type Owned = VecOwner<T>;

    type Elem = T;

    fn ensure_unique<L>(arr: &mut super::ArrayBase<L, Self>)
    where
        Self: Sized,
        L: super::Layout,
    {
    }

    fn is_unique<L>(arr: &mut super::ArrayBase<L, Self>) -> bool
    where
        Self: Sized,
    {
        true
    }

    fn ref_from_owner_offset(owner: &mut Self::Owned, offset: isize) -> Self::Ref {
        todo!()
    }
}

pub struct ArcBackend<T> {
    phantom: PhantomData<T>,
}

unsafe impl<T: Debug> Backend for ArcBackend<T> {
    type Ref = NonNull<T>;

    type Owned = ArcOwner<T>;

    type Elem = T;

    fn ensure_unique<L>(arr: &mut super::ArrayBase<L, Self>)
    where
        Self: Sized,
        L: super::Layout,
    {
        if Arc::get_mut(&mut arr.own.0).is_some() {
            return;
        }
        if arr.layout.size() <= arr.own.0.len / 2 {
            todo!(".to_owned().to_shared()");
        }
        let ptr = arr.as_ptr();
        let rcvec = &mut arr.own.0;
        let a_size = mem::size_of::<Self::Elem>() as isize;
        let our_off = if a_size != 0 {
            (ptr as isize - Arc::as_ptr(&rcvec) as isize) / a_size
        } else {
            0
        };
        arr.storage = Self::ref_from_owner_offset(&mut arr.own, our_off);
    }

    fn is_unique<L>(arr: &mut super::ArrayBase<L, Self>) -> bool
    where
        Self: Sized,
    {
        todo!()
    }

    fn ref_from_owner_offset(owner: &mut Self::Owned, offset: isize) -> Self::Ref {
        todo!()
    }
}
