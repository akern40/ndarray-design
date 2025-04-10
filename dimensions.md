# Shapes and Strides and Dimensions, Oh My!

So, `ndarray` has a rather complex landscape for the concepts of shapes, strides, dimensions, indices, and more.
Now that a reference type is well on its way to completion (see [this PR on `ndarray`](https://github.com/rust-ndarray/ndarray/pull/1440)), our next task is to see if we can create a cleaner API for these ideas.

## Terminology

Before we hop in to any code, let's just get some terminology out there.
I think our guiding light here should be the [`NumPy` reference on array objects](https://numpy.org/doc/stable/reference/arrays.ndarray.html).
There's no doubt that `NumPy` is _the_ standard for multidimensional arrays, and I think _prima facie_ that we should use the same terminology.

### Shape

First up: ["... **shape**, which is a tuple of N non-negative integers that specify the sizes of each dimension](https://numpy.org/doc/stable/reference/arrays.ndarray.html#:~:text=N%20non%2Dnegative%20integers%20that%20specify%20thesizes%20of%20each%20dimension).
That one seems simple enough.

### Dimension and Axis

From NumPy: ["a dimension of an array is sometimes referred to as an "axis""](https://numpy.org/doc/2.0/user/absolute_beginners.html#:~:text=a%20dimension%20of%20an%20array%20is%20sometimes%20referred%20to%20as%20an%20“axis”).
Ok, so NumPy conceives of "dimension" as referring to one axis in the N-dimensional space constructed by an N-dimensional array.

### Dimensionality

In the very next sentences: "This terminology may be useful to disambiguate between the dimensionality of an array and the dimensionality of the data represented by the array.
For instance, the array `a` could represent three points, each lying within a four-dimensional space, but `a` has only two "axes"."
Alright, so dimensionality of the array is the number of axes that it has.

### Strides

Similarly easy - a set of N (signed) integers that indicate the memory spacing between consecutive elements of an array along a given axis.
Interesting note, here: NumPy measures these spacings in bytes, but Rust has a very, very strong preference towards measuring spacings in terms of number of elements.
That's what `ndarray` does, as well.

## `ndarray` Design, as of v0.16.0

Alright, so let's see how `ndarray` does things currently.
The first thing to understand is that `ndarray`'s core types (like `ArrayBase`) have a `shape` field and a `strides` field.
So `ndarray` is really an n-dimensional _strided_ array library.
That has a few implications:

1. The array can't have any `const` elements in its shape
2. The array must carry its shape and strides, even if they're known or can be calculated
3. The array is only really suitable for dense arrays
4. The array can't use funky element orderings such as Z-ordering or Hilbert curves

The second thing to understand is that both of those fields are the same type - a generic `D`.
Since the shape is necessarily a set of unsigned numbers and the stride is necessarily a set of signed numbers, this means that one of them must compromise.
Right now, the strides are stored as unsigned integers and reinterpreted as signed integers whenever they're used.

The advantage of this setup is its simplicity, in that the basic array types are only parameterized by two generics (element type and shape/stride type).

### `Dimension` trait

This is probably the most prominent part of the `ndarray` API for shapes/strides/layout.
It is the trait bound on the second generic argument to `ArrayBase` required for most behavior in `ndarray`.
The top-line documentation for this type says "Array shape and index trait", but as we covered, this is also implemented for strides.

# Links

-   https://github.com/rust-ndarray/ndarray/issues/367
-   https://github.com/rust-ndarray/ndarray/issues/519
