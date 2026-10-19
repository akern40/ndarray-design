//! Shared base: `Rank`/`Ranked` as on `rust-ndarray/ndarray@refactor`, trimmed to
//! ranks 0..=4 (so `ConstRank<4>::Larger = DynRank`) to keep the mockup small.

use core::fmt::Debug;

pub trait Rank: Copy + Eq + Debug + Send + Sync + 'static
{
    const N: Option<usize>;
    type Smaller: Rank;
    type Larger: Rank;
}

pub trait RAdd<R>
{
    type Output: Rank;
}

pub trait RMax<R>
{
    type Output: Rank;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstRank<const N: usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynRank;


pub type R0 = ConstRank<0>;
pub type R1 = ConstRank<1>;
pub type R2 = ConstRank<2>;
pub type R3 = ConstRank<3>;
pub type R4 = ConstRank<4>;

impl Rank for ConstRank<0>
{
    const N: Option<usize> = Some(0);
    type Smaller = ConstRank<0>;
    type Larger = ConstRank<1>;
}

impl Rank for ConstRank<1>
{
    const N: Option<usize> = Some(1);
    type Smaller = ConstRank<0>;
    type Larger = ConstRank<2>;
}

impl Rank for ConstRank<2>
{
    const N: Option<usize> = Some(2);
    type Smaller = ConstRank<1>;
    type Larger = ConstRank<3>;
}

impl Rank for ConstRank<3>
{
    const N: Option<usize> = Some(3);
    type Smaller = ConstRank<2>;
    type Larger = ConstRank<4>;
}

impl Rank for ConstRank<4>
{
    const N: Option<usize> = Some(4);
    type Smaller = ConstRank<3>;
    type Larger = DynRank;
}

impl Rank for DynRank
{
    const N: Option<usize> = None;
    type Smaller = Self;
    type Larger = Self;
}

impl RAdd<ConstRank<0>> for ConstRank<0> { type Output = ConstRank<0>; }
impl RMax<ConstRank<0>> for ConstRank<0> { type Output = ConstRank<0>; }
impl RAdd<ConstRank<1>> for ConstRank<0> { type Output = ConstRank<1>; }
impl RMax<ConstRank<1>> for ConstRank<0> { type Output = ConstRank<1>; }
impl RAdd<ConstRank<2>> for ConstRank<0> { type Output = ConstRank<2>; }
impl RMax<ConstRank<2>> for ConstRank<0> { type Output = ConstRank<2>; }
impl RAdd<ConstRank<3>> for ConstRank<0> { type Output = ConstRank<3>; }
impl RMax<ConstRank<3>> for ConstRank<0> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<4>> for ConstRank<0> { type Output = ConstRank<4>; }
impl RMax<ConstRank<4>> for ConstRank<0> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<0>> for ConstRank<1> { type Output = ConstRank<1>; }
impl RMax<ConstRank<0>> for ConstRank<1> { type Output = ConstRank<1>; }
impl RAdd<ConstRank<1>> for ConstRank<1> { type Output = ConstRank<2>; }
impl RMax<ConstRank<1>> for ConstRank<1> { type Output = ConstRank<1>; }
impl RAdd<ConstRank<2>> for ConstRank<1> { type Output = ConstRank<3>; }
impl RMax<ConstRank<2>> for ConstRank<1> { type Output = ConstRank<2>; }
impl RAdd<ConstRank<3>> for ConstRank<1> { type Output = ConstRank<4>; }
impl RMax<ConstRank<3>> for ConstRank<1> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<4>> for ConstRank<1> { type Output = DynRank; }
impl RMax<ConstRank<4>> for ConstRank<1> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<0>> for ConstRank<2> { type Output = ConstRank<2>; }
impl RMax<ConstRank<0>> for ConstRank<2> { type Output = ConstRank<2>; }
impl RAdd<ConstRank<1>> for ConstRank<2> { type Output = ConstRank<3>; }
impl RMax<ConstRank<1>> for ConstRank<2> { type Output = ConstRank<2>; }
impl RAdd<ConstRank<2>> for ConstRank<2> { type Output = ConstRank<4>; }
impl RMax<ConstRank<2>> for ConstRank<2> { type Output = ConstRank<2>; }
impl RAdd<ConstRank<3>> for ConstRank<2> { type Output = DynRank; }
impl RMax<ConstRank<3>> for ConstRank<2> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<4>> for ConstRank<2> { type Output = DynRank; }
impl RMax<ConstRank<4>> for ConstRank<2> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<0>> for ConstRank<3> { type Output = ConstRank<3>; }
impl RMax<ConstRank<0>> for ConstRank<3> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<1>> for ConstRank<3> { type Output = ConstRank<4>; }
impl RMax<ConstRank<1>> for ConstRank<3> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<2>> for ConstRank<3> { type Output = DynRank; }
impl RMax<ConstRank<2>> for ConstRank<3> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<3>> for ConstRank<3> { type Output = DynRank; }
impl RMax<ConstRank<3>> for ConstRank<3> { type Output = ConstRank<3>; }
impl RAdd<ConstRank<4>> for ConstRank<3> { type Output = DynRank; }
impl RMax<ConstRank<4>> for ConstRank<3> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<0>> for ConstRank<4> { type Output = ConstRank<4>; }
impl RMax<ConstRank<0>> for ConstRank<4> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<1>> for ConstRank<4> { type Output = DynRank; }
impl RMax<ConstRank<1>> for ConstRank<4> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<2>> for ConstRank<4> { type Output = DynRank; }
impl RMax<ConstRank<2>> for ConstRank<4> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<3>> for ConstRank<4> { type Output = DynRank; }
impl RMax<ConstRank<3>> for ConstRank<4> { type Output = ConstRank<4>; }
impl RAdd<ConstRank<4>> for ConstRank<4> { type Output = DynRank; }
impl RMax<ConstRank<4>> for ConstRank<4> { type Output = ConstRank<4>; }

impl<const N: usize> RAdd<DynRank> for ConstRank<N> { type Output = DynRank; }
impl<const N: usize> RAdd<ConstRank<N>> for DynRank { type Output = DynRank; }
impl RAdd<DynRank> for DynRank { type Output = DynRank; }
impl<const N: usize> RMax<DynRank> for ConstRank<N> { type Output = DynRank; }
impl<const N: usize> RMax<ConstRank<N>> for DynRank { type Output = DynRank; }
impl RMax<DynRank> for DynRank { type Output = DynRank; }

/// Ties a type to its compile-time rank and runtime `ndim`.
pub trait Ranked
{
    type NDim: Rank;
    fn ndim(&self) -> usize;
}
