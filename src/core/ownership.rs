//! Ownership types

use std::{mem, ptr::NonNull, sync::Arc};

#[derive(Debug)]
pub struct VecOwner<A> {
    pub(crate) ptr: NonNull<A>,
    pub(crate) len: usize,
    pub(crate) cap: usize,
}

#[derive(Debug)]
pub struct ArcOwner<A>(pub(crate) Arc<VecOwner<A>>);

impl<A> Clone for VecOwner<A> {
    fn clone(&self) -> Self {
        todo!()
    }
}
