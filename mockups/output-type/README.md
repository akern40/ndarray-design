# Mockup: how does an operation name its output layout type?

Roadmap §3.1. Two compiling alternatives, exercised by identical generic test functions in
`src/tests.rs` (`index_axis`, `axis_iter`, `stack`, `broadcast`, `index_twice`, `into_dyn`,
`slice_axis`) over four shapes: `Ix<2>`, `Ix<4>` (top const rank → escapes to dyn), `IxDyn`,
and `ConstShape2<A>` (rank 2 with one `const` axis, to see erasure).

| | A: per-op associated types (`src/a_assoc.rs`) | B: one GAT `WithRank<R>` (`src/b_gat.rs`) |
|---|---|---|
| Trait surface | `Smaller`, `Larger`, `Dyn`, `Erased` + pairwise `BroadcastWith<O>` | `WithRank<R: Storage>` + `build<R>(lens)` |
| Impl count | one per concrete shape type (`Ix<0>`..`Ix<4>`, `IxDyn`, const) + N² broadcast table | **one** impl for the whole `Std<R>` family; const shape gets one |
| Needs a rank→type map on `Rank`? | no | **yes** (`Storage`), and it leaks as `: Storage` bounds on every rank expression in generic code |
| Chaining (`remove_axis` twice) | free: `<S::Smaller as RankOps>::Smaller` | GAT doesn't compose; must spell the chain or add type-equality bounds |
| Binary ops | trait per op, impl per type pair | free function with `RMax` bound; a *method* form hits rustc's "stricter requirements" error on `Self::NDim: RMax<..>` |
| "Slicing returns my own type" in generic code | can't be stated (`Erased` is opaque) | `S: Reranked<WithRank<S::NDim> = S>` |
| Adding a new op kind (e.g. `Erased`) | touches the trait and every impl | nothing: it's another `WithRank<...>` |

Both agree on the const-shape consequence: every runtime-axis operation (`remove`, `insert`,
`slice`, `broadcast`) lands in the plain family; the const length survives only ops that
don't take an axis argument.

Run with `cargo test` from this directory (standalone crate; not a workspace member).
