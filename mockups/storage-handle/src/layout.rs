use std::cmp::Reverse;

/// A total map from logical index to physical offset. Nothing here knows how
/// elements are stored; sparsity is the storage's business (§1c).
pub trait Layout: Clone
{
    type Index: Copy + std::fmt::Debug + PartialEq;

    fn shape(&self) -> &[usize];

    fn size(&self) -> usize
    {
        self.shape().iter().product()
    }

    /// Total: every in-bounds index has an offset. Debug-checks bounds.
    fn offset(&self, index: Self::Index) -> isize;

    /// Inclusive bounds of the offsets reachable from any in-bounds index, or
    /// `None` for an empty layout. The storage-compatibility invariant is
    /// "this range lies inside the storage's (virtual) buffer".
    fn offset_range(&self) -> Option<(isize, isize)>;

    /// Whether distinct indices map to distinct offsets. Mutable access is only
    /// constructible over unique layouts (broadcast views are the non-unique case).
    fn is_unique(&self) -> bool;

    /// Inverse of `offset`: which index, if any, maps to this offset. Needed for
    /// storage-driven iteration, where the storage knows *offsets* and the array
    /// has to report *indices*. Only meaningful on unique layouts.
    fn index_of(&self, offset: isize) -> Option<Self::Index>;

    /// All indices in the layout's preferred order (layout-driven iteration).
    fn indices(&self) -> Box<dyn Iterator<Item = Self::Index> + '_>;
}

/// Structural operations that only rewrite the layout. `slice_axis` also
/// reports how far the handle must be advanced (mdspan's `accessor.offset`),
/// since with a non-pointer handle the base offset can't just be added to a pointer.
pub trait AxisOps: Layout
{
    fn reversed_axes(&self) -> Self;
    fn slice_axis(&self, axis: usize, start: usize, len: usize) -> (Self, isize);
    /// `None` unless `shape[axis] == 1`. The result is not unique.
    fn broadcast_axis(&self, axis: usize, len: usize) -> Option<Self>;
}

/// Plain strided layout, non-negative strides only (enough for the mockup).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strided<const N: usize>
{
    shape: [usize; N],
    strides: [isize; N],
}

impl<const N: usize> Strided<N>
{
    pub fn c_order(shape: [usize; N]) -> Self
    {
        let mut strides = [0isize; N];
        let mut acc = 1isize;
        for ax in (0..N).rev() {
            strides[ax] = acc;
            acc *= shape[ax] as isize;
        }
        Strided { shape, strides }
    }

    pub fn from_shape_strides(shape: [usize; N], strides: [isize; N]) -> Self
    {
        assert!(strides.iter().all(|&s| s >= 0), "mockup: non-negative strides only");
        Strided { shape, strides }
    }

    pub fn strides(&self) -> &[isize; N]
    {
        &self.strides
    }

    /// Axes of length > 1, largest stride first.
    fn axes_by_stride(&self) -> Vec<usize>
    {
        let mut axes: Vec<usize> = (0..N).filter(|&a| self.shape[a] > 1).collect();
        axes.sort_by_key(|&a| Reverse(self.strides[a]));
        axes
    }
}

impl<const N: usize> Layout for Strided<N>
{
    type Index = [usize; N];

    fn shape(&self) -> &[usize]
    {
        &self.shape
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn offset(&self, index: [usize; N]) -> isize
    {
        let mut off = 0isize;
        for ax in 0..N {
            debug_assert!(index[ax] < self.shape[ax], "index out of bounds");
            off += index[ax] as isize * self.strides[ax];
        }
        off
    }

    fn offset_range(&self) -> Option<(isize, isize)>
    {
        if self.size() == 0 {
            return None;
        }
        let max = (0..N).map(|ax| (self.shape[ax] as isize - 1) * self.strides[ax]).sum();
        Some((0, max))
    }

    /// Sufficient condition: the axes nest (each stride is at least the extent
    /// of the next-smaller axis). Covers every view derived from a contiguous
    /// array by slicing/transposing; a general overlap test is a separate problem.
    fn is_unique(&self) -> bool
    {
        let axes = self.axes_by_stride();
        axes.iter().all(|&a| self.strides[a] > 0)
            && axes
                .windows(2)
                .all(|w| self.strides[w[0]] >= self.shape[w[1]] as isize * self.strides[w[1]])
    }

    fn index_of(&self, offset: isize) -> Option<[usize; N]>
    {
        if offset < 0 || self.size() == 0 || !self.is_unique() {
            return None;
        }
        let mut idx = [0usize; N];
        let mut rem = offset;
        for a in self.axes_by_stride() {
            let i = rem / self.strides[a];
            if i as usize >= self.shape[a] {
                return None;
            }
            idx[a] = i as usize;
            rem -= i * self.strides[a];
        }
        (rem == 0).then_some(idx)
    }

    fn indices(&self) -> Box<dyn Iterator<Item = [usize; N]> + '_>
    {
        let shape = self.shape;
        Box::new((0..self.size()).map(move |mut k| {
            let mut idx = [0usize; N];
            for ax in (0..N).rev() {
                idx[ax] = k % shape[ax];
                k /= shape[ax];
            }
            idx
        }))
    }
}

impl<const N: usize> AxisOps for Strided<N>
{
    fn reversed_axes(&self) -> Self
    {
        let mut l = self.clone();
        l.shape.reverse();
        l.strides.reverse();
        l
    }

    fn slice_axis(&self, axis: usize, start: usize, len: usize) -> (Self, isize)
    {
        assert!(start + len <= self.shape[axis], "slice out of bounds");
        let mut l = self.clone();
        l.shape[axis] = len;
        (l, start as isize * self.strides[axis])
    }

    fn broadcast_axis(&self, axis: usize, len: usize) -> Option<Self>
    {
        if self.shape[axis] != 1 {
            return None;
        }
        let mut l = self.clone();
        l.shape[axis] = len;
        l.strides[axis] = 0;
        Some(l)
    }
}
