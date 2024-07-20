use super::{AddAxis, Layout, Patterned, RemoveAxis};

pub struct Dense<const N: usize>(pub(super) [usize; N]);

impl<const N: usize> Layout for Dense<N> {
    const NDIM: Option<usize> = Some(N);

    fn ndim(&self) -> usize {
        N
    }

    fn slice(&self) -> &[usize] {
        &self.0
    }

    fn slice_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

macro_rules! impl_patterned {
    ($name:ty, $pattern:ty) => {
        impl Patterned for $name {
            type Pattern = $pattern;

            fn as_pattern(&self) -> Self::Pattern {
                self.0.into()
            }
        }

        impl From<$pattern> for $name {
            fn from(value: $pattern) -> Self {
                Dense { 0: value.into() }
            }
        }
    };
}

impl Patterned for Dense<0> {
    type Pattern = ();

    fn as_pattern(&self) -> Self::Pattern {
        ()
    }
}

impl From<()> for Dense<0> {
    fn from(_value: ()) -> Self {
        Dense { 0: [] }
    }
}

impl RemoveAxis for Dense<1> {
    type Smaller = Dense<0>;

    fn remove_axis(&self, _axis: usize) -> Self::Smaller {
        Dense { 0: [] }
    }
}

impl RemoveAxis for Dense<2> {
    type Smaller = Dense<1>;

    fn remove_axis(&self, axis: usize) -> Self::Smaller {
        match axis {
            0 => Dense { 0: [self.0[1]] },
            1.. => Dense { 0: [self.0[0]] },
        }
    }
}

impl_patterned!(Dense<1>, (usize,));
impl_patterned!(Dense<2>, (usize, usize));

impl AddAxis for Dense<0> {
    type Larger = Dense<1>;

    fn add_axis(&self, _axis: usize, length: usize) -> Self::Larger {
        Dense { 0: [length] }
    }
}

impl AddAxis for Dense<1> {
    type Larger = Dense<2>;

    fn add_axis(&self, axis: usize, length: usize) -> Self::Larger {
        match axis {
            0 => Dense {
                0: [length, self.0[0]],
            },
            1.. => Dense {
                0: [self.0[0], length],
            },
        }
    }
}
