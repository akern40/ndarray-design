use super::Layout;

/// A trait that provides compatibility with a pattern-matching type.
pub trait Patterned: Layout + Sized {
    /// Pattern matching friendly form of the dimension value.
    ///
    /// - For a one-dimensional Layout: `usize`,
    /// - For a two-dimensional Layout: `(usize, usize)`
    /// - and so on..
    type Pattern: Into<Self>;

    /// Convert the dimension into a pattern matching friendly value.
    fn as_pattern(&self) -> Self::Pattern;
}
