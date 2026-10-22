use crate::accessor::{Accessor, Dense, Sparse};
use crate::array::{Array, ArrayRef, ShapeError, SparseArray};
use crate::layout::{Layout, Strided};

fn dense() -> Array<f64, Strided<2>>
{
    // [[0, 1, 2],
    //  [3, 4, 5]]
    Array::from_shape_vec(Strided::c_order([2, 3]), (0..6).map(f64::from).collect()).unwrap()
}

fn sparse() -> SparseArray<f64, Strided<2>>
{
    // [[0, 1, 0],
    //  [0, 0, 5]]
    SparseArray::from_entries(Strided::c_order([2, 3]), 0.0, vec![([0, 1], 1.0), ([1, 2], 5.0)]).unwrap()
}

/// One generic function, both families: the point of §1c.
fn trace<Acc: Accessor<Elem = f64>>(a: &ArrayRef<Acc, Strided<2>>) -> f64
{
    (0..a.shape()[0].min(a.shape()[1])).map(|i| *a.get([i, i]).unwrap()).sum()
}

#[test]
fn offset_range_invariant_is_checked_at_construction()
{
    assert_eq!(
        Array::<f64, _>::from_shape_vec(Strided::c_order([2, 3]), vec![0.0; 5]).err(),
        Some(ShapeError::OffsetRangeExceedsStorage)
    );
    // A strided layout over a larger buffer is fine as long as its range fits.
    let l = Strided::from_shape_strides([2, 2], [3, 1]);
    assert!(Array::<f64, _>::from_shape_vec(l, vec![0.0; 6]).is_ok());
}

#[test]
fn dense_and_sparse_share_the_generic_surface()
{
    let d = dense();
    let s = sparse();
    assert_eq!(trace(&d), 4.0);
    assert_eq!(trace(&s), 0.0);
    assert_eq!(d.sum(), 15.0);
    assert_eq!(s.sum(), 6.0);
    assert_eq!(s.get([0, 0]), Some(&0.0)); // a hole reads as the fill
    assert_eq!(s.get([1, 2]), Some(&5.0));
    assert_eq!(s.get([2, 0]), None);
}

#[test]
fn transpose_and_slice_on_sparse_are_layout_only()
{
    let s = sparse();
    let addr = s.handle_addr();

    let t = s.t();
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.layout().strides(), &[1, 3]); // CSR read as CSC, same buffer
    assert_eq!(t.handle_addr(), addr);
    assert_eq!(t.get([1, 0]), Some(&1.0));
    assert_eq!(t.get([2, 1]), Some(&5.0));
    let mut stored: Vec<_> = t.stored_iter().map(|(i, a)| (i, *a)).collect();
    stored.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(stored, vec![([1, 0], 1.0), ([2, 1], 5.0)]);

    // Column slice [:, 1..3]: only the entries inside the window survive, re-indexed.
    let c = s.slice_axis(1, 1, 2);
    assert_eq!(c.shape(), &[2, 2]);
    let stored: Vec<_> = c.stored_iter().map(|(i, a)| (i, *a)).collect();
    assert_eq!(stored, vec![([0, 0], 1.0), ([1, 1], 5.0)]);
    assert_eq!(c.sum(), 6.0);
    // Row slice [1..2, :] drops the entry in row 0.
    let r = s.slice_axis(0, 1, 1);
    let stored: Vec<_> = r.stored_iter().map(|(i, a)| (i, *a)).collect();
    assert_eq!(stored, vec![([0, 2], 5.0)]);
}

#[test]
fn dense_views_slice_through_the_handle()
{
    let d = dense();
    let c = d.slice_axis(1, 1, 2);
    assert_eq!(c.get([0, 0]), Some(&1.0));
    assert_eq!(c.get([1, 1]), Some(&5.0));
    let all: Vec<_> = c.t().stored_iter().map(|(_, a)| *a).collect();
    assert_eq!(all, vec![1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn insertion_goes_through_mut_ref_and_keeps_the_handle()
{
    let mut s = sparse();
    let addr = s.handle_addr();
    assert_eq!(s.nnz(), 2);

    // Hole -> stored, directly on the owned array.
    *s.get_mut([1, 0]).unwrap() = 7.0;
    assert_eq!(s.nnz(), 3);
    assert_eq!(s.get([1, 0]), Some(&7.0));

    // Through a sliced mutable view: index is view-relative, lands at [0, 2].
    {
        let mut v = s.slice_axis_mut(1, 1, 2);
        *v.get_mut([0, 1]).unwrap() = 9.0;
    }
    assert_eq!(s.get([0, 2]), Some(&9.0));
    assert_eq!(s.nnz(), 4);

    // Enough insertions to force the entries Vec to reallocate; handle unchanged.
    let big_layout = Strided::c_order([64, 64]);
    let mut b = SparseArray::from_entries(big_layout, 0.0, vec![]).unwrap();
    let b_addr = b.handle_addr();
    for i in 0..64 {
        *b.get_mut([i, i]).unwrap() = 1.0;
    }
    assert_eq!(b.nnz(), 64);
    assert_eq!(b.handle_addr(), b_addr);
    assert_eq!(s.handle_addr(), addr);
    assert_eq!(b.sum(), 64.0);
}

#[test]
fn copy_on_write_refreshes_the_handle()
{
    let a = dense().into_shared();
    let mut b = a.clone();
    *b.get_mut([0, 0]).unwrap() = 100.0; // DerefMut: ensure_unique clones and re-derives the handle
    assert_eq!(a.get([0, 0]), Some(&0.0));
    assert_eq!(b.get([0, 0]), Some(&100.0));
    assert_eq!(b.get([1, 2]), Some(&5.0));
}

#[test]
fn broadcast_is_non_unique_and_read_only()
{
    let d = Array::from_shape_vec(Strided::c_order([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let b = d.broadcast_axis(0, 4).unwrap();
    assert!(!b.is_unique());
    assert_eq!(b.get([3, 2]), Some(&3.0));
    assert_eq!(b.sum(), 24.0);
    // No inverse on a non-unique layout: storage-driven iteration is not available.
    assert_eq!(b.layout().index_of(2), None);
    // `b` is an ArrayView: there is no `get_mut` path to it by construction.
}

#[test]
fn index_of_inverts_offset_on_derived_views()
{
    let base = Strided::c_order([4, 5]);
    let (l, shift) = crate::layout::AxisOps::slice_axis(&base, 1, 1, 3);
    let l = crate::layout::AxisOps::reversed_axes(&l);
    assert_eq!(shift, 1);
    for i in l.indices() {
        assert_eq!(l.index_of(l.offset(i)), Some(i));
    }
    // An offset that belongs to a column outside the slice is rejected.
    assert_eq!(l.index_of(3), None);
}

#[test]
fn sparse_times_dense_is_a_stored_walk()
{
    // Σ s[i] * d[i]: walk the sparse one's stored entries, look up the dense one.
    let s = sparse();
    let d = dense();
    let fill = *s.fill_value().unwrap();
    let dense_sum = d.sum();
    let (mut acc, mut hit) = (0.0, 0.0);
    for (i, a) in s.stored_iter() {
        let b = *d.get(i).unwrap();
        acc += a * b;
        hit += b;
    }
    let result = acc + fill * (dense_sum - hit);
    assert_eq!(result, 1.0 * 1.0 + 5.0 * 5.0);
    // Dense<f64> and Sparse<f64> are different accessor types, same element type.
    fn same_elem<X: Accessor<Elem = f64>, Y: Accessor<Elem = f64>>() {}
    same_elem::<Dense<f64>, Sparse<f64>>();
}
