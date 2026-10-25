# ndarray 1.0a requirements

A plain list of what 1.0a has to do. No design details here; those live in `roadmap.md`.

## Tracked

- Dense arrays at least as fast as today's.
- Arrays with lengths known at compile time, fully or partly.
- Tiled (blocked) arrays.
- Sparse arrays.
- No sealed traits: other crates can add their own kinds of storage and layout.
- Works without the standard library.
- The core crate's safety is provable, not just tested.
- Parallel work: split arrays (tiles included) into independent pieces for CPU or GPU threads, safely.
- Work over arrays too big for memory, one chunk at a time.
- GPU memory: arrays whose elements the CPU can't read still get every shape operation (slicing, transposing, reshaping, splitting into tiles).
- GPU kernels: the same view and layout types work inside GPU kernels written in Rust.
- GPU interop: convert to and from the GPU array types of other Rust crates without copying.

## Noted, wording still open

- Arrays whose element type is only known at runtime (what Python, Arrow and file formats hand you).
- Arrays of packed elements smaller than a byte (4-bit and 8-bit number formats, bit-packed booleans).
- An array can keep memory that another library allocated alive, so it can be returned, stored or sent to another thread, not just borrowed.

## Low priority: don't rule out

- Named axes, typed axes and typed indices, as add-on crates. Core only has to avoid blocking them.
- Axes that carry labels or coordinates (time stamps, latitudes), as an add-on.

## Unsure

- Streams: arrays that keep growing along one axis, and rolling windows. It's not yet clear what "support" would mean.

## Out of scope

- Index origins other than zero.
- Ragged arrays. A padded ragged array can be built with a custom layout.

## Proposed, not yet discussed

- Exchange arrays without copying with Python (NumPy, PyTorch, JAX), Arrow and GPU libraries.
- Convert array references to and from the other Rust array crates (faer, mdarray, nalgebra) without copying.
- A small, stable core that other crates can depend on for years.
- Arrays that know where they sit inside a bigger array (halo cells, a piece of a distributed array, a chunk of a file).
- View one field of an array of records, or a fixed-size array element, without copying.
- Element-wise operations whose result has a different element type (units of measure, mixed precision).
- Views and layouts usable without an allocator, so the same types can run on embedded targets.
- Arrays with many axes (dozens) stay practical.
