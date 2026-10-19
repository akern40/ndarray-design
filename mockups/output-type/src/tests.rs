//! The same generic test functions written against each mockup. What matters is
//! (1) whether the generic signature can be written at all and how many bounds it needs,
//! (2) what concrete type comes out for `NShape<2>`, `DShape`, `NShape<4>` (the escape to
//! dynamic rank) and `ConstShape2<3>` (erasure).

use crate::rank::*;
use crate::shape::*;

/// Type assertion helper: `is::<Expected>(&value)` only compiles if the types match.
fn is<T>(_: &T) {}

// ---------------------------------------------------------------- Mockup A
mod a
{
    use super::*;
    use crate::a_assoc::*;

    fn index_axis<S: RankOps>(s: &S, axis: usize) -> S::Smaller
    {
        s.remove_axis(axis)
    }

    /// What an `axis(k)` handle needs: iterate sub-shapes of rank `Smaller`.
    fn axis_iter<S: RankOps>(s: &S, axis: usize) -> impl Iterator<Item = S::Smaller> + '_
    {
        (0..s.axis_len(axis)).map(move |_| s.remove_axis(axis))
    }

    fn stack<S: RankOps>(parts: &[S], axis: usize) -> S::Larger
    {
        parts[0].insert_axis(axis, parts.len())
    }

    fn broadcast<S1, S2>(a: &S1, b: &S2) -> Option<S1::Output>
    where
        S1: BroadcastWith<S2>,
        S2: RankOps,
        S1::NDim: RMax<S2::NDim>,
    {
        a.broadcast(b)
    }

    /// Two rank changes in a row: `Smaller::Smaller` chains without extra bounds.
    fn index_twice<S: RankOps>(s: &S) -> <S::Smaller as RankOps>::Smaller
    {
        s.remove_axis(0).remove_axis(0)
    }

    #[test]
    fn output_types()
    {
        let s2 = NShape([2, 3]);
        let s4 = NShape([1, 2, 3, 4]);
        let d = DShape(vec![2, 3]);
        let c = ConstShape2::<2>(3);

        is::<NShape<1>>(&index_axis(&s2, 0));
        is::<NShape<3>>(&stack(&[s2, s2], 0));
        is::<NShape<2>>(&s2.slice_axis(1, 1));
        is::<DShape>(&s2.into_dyn());
        is::<NShape<0>>(&index_twice(&s2));
        assert_eq!(axis_iter(&s2, 0).count(), 2);

        // Escape at the top const rank.
        is::<DShape>(&stack(&[s4, s4], 0));
        is::<DShape>(&index_axis(&d, 0));

        // Broadcasting.
        is::<NShape<2>>(&broadcast(&s2, &NShape([3])).unwrap());
        is::<DShape>(&broadcast(&s2, &d).unwrap());
        is::<DShape>(&broadcast(&d, &s2).unwrap());

        // Erasure: every runtime-axis op on the const shape lands in the plain family.
        is::<NShape<1>>(&index_axis(&c, 1));
        is::<NShape<2>>(&c.slice_axis(1, 2));
        is::<NShape<3>>(&stack(&[c, c], 0));
        is::<NShape<2>>(&broadcast(&c, &s2).unwrap());
        assert_eq!(c.slice_axis(0, 1), NShape([1, 3]));
    }
}

// ---------------------------------------------------------------- Mockup B
mod b
{
    use super::*;
    use crate::b_gat::*;

    type Ix<const N: usize> = Std<ConstRank<N>>;
    type IxDyn = Std<DynRank>;

    fn index_axis<S: Reranked>(s: &S, axis: usize) -> S::WithRank<<S::NDim as Rank>::Smaller>
    where <S::NDim as Rank>::Smaller: Storage
    {
        s.remove_axis(axis)
    }

    fn axis_iter<S: Reranked>(s: &S, axis: usize) -> impl Iterator<Item = S::WithRank<<S::NDim as Rank>::Smaller>> + '_
    where <S::NDim as Rank>::Smaller: Storage
    {
        (0..s.axis_len(axis)).map(move |_| s.remove_axis(axis))
    }

    fn stack<S: Reranked>(parts: &[S], axis: usize) -> S::WithRank<<S::NDim as Rank>::Larger>
    where <S::NDim as Rank>::Larger: Storage
    {
        parts[0].insert_axis(axis, parts.len())
    }

    // `broadcast` is the free function from `b_gat` (no pairwise trait).

    /// Two rank changes in a row. Finding: the GAT does **not** compose by itself —
    /// rustc does not know `S::WithRank<A>::WithRank<B> == S::WithRank<B>`, so a chained
    /// result must either be spelled as the chain (below) or the caller must add the bound
    /// `S: Reranked<WithRank<X> = S>`-style equalities. Mockup A chains for free.
    fn index_twice<S: Reranked>(
        s: &S,
    ) -> <S::WithRank<<S::NDim as Rank>::Smaller> as Reranked>::WithRank<<<S::NDim as Rank>::Smaller as Rank>::Smaller>
    where
        <S::NDim as Rank>::Smaller: Storage,
        <<S::NDim as Rank>::Smaller as Rank>::Smaller: Storage,
    {
        s.remove_axis(0).remove_axis(0)
    }

    /// Generic code that wants "slicing gives me back my own type" can say so.
    fn slice_same<S>(s: &S) -> S
    where
        S: Reranked<WithRank<<S as Ranked>::NDim> = S>,
        S::NDim: Storage,
    {
        s.slice_axis(0, 1)
    }

    #[test]
    fn output_types()
    {
        let s2: Ix<2> = Std(Arr([2, 3]));
        let s4: Ix<4> = Std(Arr([1, 2, 3, 4]));
        let d: IxDyn = Std(vec![2, 3]);
        let c = ConstShape2::<2>(3);

        is::<Ix<1>>(&index_axis(&s2, 0));
        is::<Ix<3>>(&stack(&[s2.clone(), s2.clone()], 0));
        is::<Ix<2>>(&s2.slice_axis(1, 1));
        is::<Ix<2>>(&slice_same(&s2));
        is::<IxDyn>(&s2.into_dyn());
        is::<Ix<0>>(&index_twice(&s2));
        assert_eq!(axis_iter(&s2, 0).count(), 2);

        is::<IxDyn>(&stack(&[s4.clone(), s4], 0));
        is::<IxDyn>(&index_axis(&d, 0));

        is::<Ix<2>>(&broadcast(&s2, &Std::<ConstRank<1>>(Arr([3]))).unwrap());
        is::<IxDyn>(&broadcast(&s2, &d).unwrap());
        is::<IxDyn>(&broadcast(&d, &s2).unwrap());

        is::<Ix<1>>(&index_axis(&c, 1));
        is::<Ix<2>>(&c.slice_axis(1, 2));
        is::<Ix<3>>(&stack(&[c, c], 0));
        is::<Ix<2>>(&broadcast(&c, &s2).unwrap());
        assert_eq!(c.slice_axis(0, 1), Std(Arr([1, 3])));
    }
}
