use core::fmt::Debug;

/// A trait representing a dimensionality, i.e., an unsigned integer indicating how many axes an array has.
///
/// `ndarray` has a history of encoding arrays' dimensionality in the type system; this turns out to be
/// useful for debugging and for writing complex array libraries.
/// However, some arrays don't (or can't) have their dimensionality known at compile time.
/// One good example of this is the output of [`squeeze`](`crate::ArrayBase::squeeze`): even if
/// the array's dimensionality is known before the operation, the dimensionality of the output
/// is dependent on the number of axes that have a length of one. So, this trait is used to unify
/// both the known- and unknown-dimensionality cases.
///
/// `ndarray` currently limits compile-time dimensionalities to values between 0 and 12, inclusive.
/// Any dimensionality above 12, even if it's known at compile time, must be represented with [`DDyn`].
/// See both [`NDim`] and [`DDyn`] for some suggestions on how to use them effectively.
/// See [below](Dimensionality#why-can-I-only-have-dimensionalities-up-to-12?) for information on
/// alternative solutions.
///
/// # A note on dynamic dimensionalities
/// Close readers of the `Dimensionality` or [`DDyn`] code will notice that there is no way
/// to get the number of dimensions at runtime from a type that implements `Dimensionality`.
/// This is an intentional decision; a dynamic dimensionality here is not "the dimensionality
/// can be known at runtime" but rather "the dimensionality _cannot_ be known at compile time".
/// This design was chosen so that users of `ndarray` do not need to worry about synchronizing
/// the runtime value of the "dimensionality" with the actual runtime dimensionality of the array's
/// shape and strides.
///
/// # Why can I only have dimensionalities up to 12?
/// `ndarray` currently limits compile-time dimensionalities to values between 0 and 12, inclusive.
/// Any dimensionality above 12, even if it's known at compile time, must be represented with [`DDyn`].
/// See both [`NDim`] and [`DDyn`] for some suggestions on how to use them effectively.
/// Below is a quick explanation of why the two clearest solutions - const generics and the
/// [`typenum` crate](https://docs.rs/typenum/latest/typenum/index.html) - don't suffice.
///
/// ## Const Generics
/// Using const generics seems like the obvious solution to compile-time dimensionalities;
/// indeed, the library makes use of them whenever and wherever it can. However, const generics
/// in Rust are, as of the time of this writing, not mature enough on their own to enable all of
/// the capability that we'd want from compile-time dimensionalities.
///
/// As a quick example, take broadcasting: the dimensionality of broadcasting a 2D array with a
/// 3D array will be 3D. However, [`usize::max`] is not a `const` function. So, using const generics
/// alone, there is no way to write this maximum for any arbitrary pair of `N`D and `M`D arrays. Instead,
/// the library must resort to implementing its own [`DMax`] trait "by hand" for each possible pair
/// of dimensionalities.
///
/// A similar problem arises with using dimensionality sums, e.g., for concatenation. Rust's ability to
/// pass expressions like `{N + M}` is still locked behind the [`generic_const_exprs` feature flag](
/// https://doc.rust-lang.org/beta/unstable-book/language-features/generic-const-exprs.html), so
/// typing concatenation must be done using `ndarray`'s own [`DAdd`] trait implemented "by hand".
///
/// ## Typenum
/// The incredible [`typenum` crate](https://docs.rs/typenum/latest/typenum/index.html) seems like
/// the other obvious solution to having any-size dimensionalities. It elegantly solves the issues
/// of maximums and sums mentioned above for const generics. However, moving into the `typenum` world
/// is a one-way trip: you cannot go _from_ `typenum` _to_ const generics, only the other way around.
/// You also can't use associated constants (which `typenum` has) to do things like define an array's
/// length (which `ndarray` has to do quite frequently). Additionally, `typenum` tends to introduce a fair
/// number of trait bounds. Finally, `ndarray` would have to augment `typenum` with the "dynamic
/// dimensionality escape hatch", which would include quite a bit of patching. As a result, the library
/// uses a more limited style of type-encoded integers instead of relying on `typenum` directly.
pub trait Dimensionality:
    Copy
    + Eq
    + Debug
    + Send
    + Sync
    + DMax<D0, Output = Self>
    + DMax<Self, Output = Self>
    + DMax<DDyn, Output = DDyn>
    + DMax<Self::Smaller, Output = Self>
    + DMax<Self::Larger, Output = Self::Larger>
    + DAdd<Self>
    + DAdd<Self::Smaller>
    + DAdd<Self::Larger>
    + DAdd<D0, Output = Self>
    + DAdd<D1, Output = Self::Larger>
    + DAdd<DDyn, Output = DDyn>
{
    /// The dimensionality as a constant usize, if it's not dynamic.
    const N: Option<usize>;

    type Smaller: Dimensionality;

    type Larger: Dimensionality; // And more
}

pub trait DAdd<D> {
    type Output: Dimensionality;
}

pub trait DMax<D> {
    type Output: Dimensionality;
}

/// The N-dimensional static dimensionality.
///
/// This type indicates dimensionalities that are known at compile-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NDim<const N: usize>;

pub type D0 = NDim<0>;
pub type D1 = NDim<1>;
pub type D2 = NDim<2>;
pub type D3 = NDim<3>;
pub type D4 = NDim<4>;
pub type D5 = NDim<5>;
pub type D6 = NDim<6>;
pub type D7 = NDim<7>;
pub type D8 = NDim<8>;
pub type D9 = NDim<9>;
pub type D10 = NDim<10>;
pub type D11 = NDim<11>;
pub type D12 = NDim<12>;

macro_rules! impl_add {
    ($left:literal, ($($right:literal),+), ddyn: ($($rightd:literal),*)) => {
        $(
            impl DAdd<NDim<$right>> for NDim<$left>
            {
                type Output = NDim<{$left + $right}>;
            }
        )+

        $(
            impl DAdd<NDim<$rightd>> for NDim<$left>
            {
                type Output = DDyn;
            }
        )*
    };
}

// There's got to be a macro way to do this in one line to help with
// any future additions of extra dimenions, although it might
// also slow down compile times.
impl_add!(0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ddyn: ());
impl_add!(1, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), ddyn: (12));
impl_add!(2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ddyn: (11, 12));
impl_add!(3, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), ddyn: (10, 11, 12));
impl_add!(4, (0, 1, 2, 3, 4, 5, 6, 7, 8), ddyn: (9, 10, 11, 12));
impl_add!(5, (0, 1, 2, 3, 4, 5, 6, 7), ddyn: (8, 9, 10, 11, 12));
impl_add!(6, (0, 1, 2, 3, 4, 5, 6), ddyn: (7, 8, 9, 10, 11, 12));
impl_add!(7, (0, 1, 2, 3, 4, 5), ddyn: (6, 7, 8, 9, 10, 11, 12));
impl_add!(8, (0, 1, 2, 3, 4), ddyn: (5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(9, (0, 1, 2, 3), ddyn: (4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(10, (0, 1, 2), ddyn: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(11, (0, 1), ddyn: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(12, (0), ddyn: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

macro_rules! impl_max {
    // Base case, just a target with some lowers
    ($($lower:literal),+, target: $target:literal) => {
        $(
            impl DMax<NDim<$lower>> for NDim<$target>
            {
                type Output = NDim<$target>;
            }
        )+
    };
    // General case: at least one lower, at least one upper
    ($($lower:literal),+$(,)? target: $target:literal, $first_upper:literal$(, $($upper:literal),+)?) => {
        $(
            impl DMax<NDim<$lower>> for NDim<$target>
            {
                type Output = NDim<$target>;
            }
        )+
        impl DMax<NDim<$first_upper>> for NDim<$target>
        {
            type Output = NDim<$first_upper>;
        }
        $(
            $(
                impl DMax<NDim<$upper>> for NDim<$target>
                {
                    type Output = NDim<$upper>;
                }
            )+
        )?
        impl_max!($($lower),+, $target, target: $first_upper$(, $($upper),+)?);
    };
    // Helper syntax: zero lowers, target, at least one upper
    (target: $target:literal, $first_upper:literal, $($upper:literal),+) => {
        impl DMax<NDim<$first_upper>> for NDim<$target>
        {
            type Output = NDim<$first_upper>;
        }
        $(
            impl DMax<NDim<$upper>> for NDim<$target>
            {
                type Output = NDim<$upper>;
            }
        )+
        impl_max!($target, target: $first_upper, $($upper),+);
    };
}

impl_max!(target: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

impl<const N: usize> DMax<NDim<N>> for NDim<N>
where
    NDim<N>: Dimensionality,
{
    type Output = Self;
}

macro_rules! impl_dimensionality {
    ($($d:literal),+) => {
        $(
            impl Dimensionality for NDim<$d>
            {
                const N: Option<usize> = Some($d);

                type Smaller = NDim<{$d - 1}>;

                type Larger = NDim<{$d + 1}>;
            }
        )+
    };
}

impl Dimensionality for D0 {
    const N: Option<usize> = Some(0);

    type Smaller = Self;

    type Larger = D1;
}

impl_dimensionality!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

impl Dimensionality for NDim<12> {
    const N: Option<usize> = Some(12);

    type Smaller = D11;

    type Larger = DDyn;
}

/// The dynamic dimensionality.
///
/// This type indicates dimensionalities that can only be known at runtime.
/// See [`Dimensionality`] for more information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DDyn;

impl Dimensionality for DDyn {
    const N: Option<usize> = None;

    type Smaller = Self;

    type Larger = Self;
}

impl DAdd<DDyn> for DDyn {
    type Output = DDyn;
}

impl<const N: usize> DAdd<NDim<N>> for DDyn {
    type Output = DDyn;
}

impl<const N: usize> DAdd<DDyn> for NDim<N> {
    type Output = DDyn;
}

impl DMax<DDyn> for DDyn {
    type Output = DDyn;
}

impl<const N: usize> DMax<NDim<N>> for DDyn {
    type Output = DDyn;
}

impl<const N: usize> DMax<DDyn> for NDim<N> {
    type Output = DDyn;
}
