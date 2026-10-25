# ndarray 1.0a requirements

A plain list of what 1.0a has to do. No design details here; those live in `roadmap.md`.

## Tracked

- Dense arrays at least as fast as today's.
- Arrays with lengths known at compile time, fully or partly.
- Tiled (blocked) arrays.
- Sparse arrays.
- No sealed traits: other crates can add their own kinds of storage and layout.
- Works without the standard library.
- Named axes are possible as a small add-on crate, with no changes to core.

## Noted, wording still open

- Arrays whose element type is only known at runtime (what Python, Arrow and file formats hand you).
- Arrays of packed elements smaller than a byte (4-bit and 8-bit number formats, bit-packed booleans).
- An array can keep memory that another library allocated alive, so it can be returned, stored or sent to another thread, not just borrowed.

## Proposed, not yet discussed

- Exchange arrays without copying with Python (NumPy, PyTorch, JAX), Arrow and GPU libraries.
- Convert array references to and from the other Rust array crates (faer, mdarray, nalgebra) without copying.
- A small, stable core that other crates can depend on for years.
- Axes can carry labels or coordinates, not just names (time stamps, latitudes), as an add-on.
- Work over arrays too big for memory, one chunk at a time, with the chunks supplied by another crate.
- Arrays that keep growing along one axis (streams, sensors, audio), including fixed-size rolling windows.
- Arrays that know where they sit inside a bigger array (halo cells, a piece of a distributed array, a chunk of a file).
- View one field of an array of records, or a fixed-size array element, without copying.
- Element-wise operations whose result has a different element type (units of measure, mixed precision).
- Split a mutable array into non-overlapping pieces, tiles included, and hand them to threads safely.
- Views and layouts usable without an allocator, so the same types can run on embedded targets and in GPU kernels written in Rust.
- Arrays with many axes (dozens) stay practical.
- Ragged arrays (rows of different lengths): in scope or not?
