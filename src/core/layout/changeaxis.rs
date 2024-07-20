use super::Layout;

/// Array shape with a next smaller dimension.
///
/// `RemoveAxis` defines a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *Smaller*.
pub trait RemoveAxis: Layout {
    /// Next smaller dimension (if applicable)
    type Smaller: Layout + AddAxis;

    fn remove_axis(&self, axis: usize) -> Self::Smaller;
}

pub trait AddAxis: Layout {
    /// Next larger dimension
    type Larger: Layout + RemoveAxis;

    /// Get the larger dimension corresponding to adding a dimension of `length` at `axis`.
    ///
    /// If `axis` is greater than the number of existing dimensions, adds the extra dimension
    /// after the last existing dimension.
    fn add_axis(&self, axis: usize, length: usize) -> Self::Larger;
}
