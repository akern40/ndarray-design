use crate::core::{Backend, Layout};

use super::ArrayBase;

impl<L: Layout, B: Backend> ArrayBase<L, B> {
    fn ensure_unique(&mut self) {
        B::ensure_unique(self)
    }
}
