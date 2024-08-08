# `ndarrray` Design, from the Ground Up

Disclaimer: This README is not official for `ndarray`.
Its aim is to articulate, as clearly as possible, some design principles and goals that might be achieved through a redesign of `ndarray`'s core.

Second Disclaimer: This design document tries to build an understanding of `ndarray`'s (potential) design from first principles, as a sort of pedagogical exercise.
But the ideas in this document are the culmination of years of design, dialogue, and development, in particular by [bluss](https://github.com/bluss), [Jim Turner](https://github.com/jturner314), [Vincent Barrielle](https://github.com/vbarrielle), and [Luca Palmieri](https://github.com/LukeMathWalker)[^0].

## `ndarray` as a Multi-Dimensional Vec and Slice
Fundamentally, we might think of `ndarray`'s core aspiration as providing data structures that are multi-dimensional generalizations of Rust's standard `Vec<T>` and `&[T]`: a block of data with some information describing its "shape" (rather than just its length).
So this design document will start there.
Before we go any further, it's worth emphasizing precisely what those types are, and the slice type in particular, because understanding those details will be important for understanding the art of the possible with `ndarray`.
If you're already comfortable with Dynamically Sized Types and fat pointers, you can skip to [the next section](#lets-just-copy-vec-and-slice).

### A Totally Inadequate Primer on Rust's Slices
Ok, so what's going on with slices?
We know them, we love them, and I absolutely did not understand them until I started thinking about `ndarray`'s internals.
The slice is a [*dynamically sized type*](https://doc.rust-lang.org/reference/dynamically-sized-types.html), and a reference to a slice is known as a "fat pointer".
And [what is a "fat pointer"](https://stackoverflow.com/questions/57754901/what-is-a-fat-pointer)?
It's a pointer, plus some additional information about that pointer, that still looks and acts like a pointer.

... Ok, so that sentence is all well and good, but to be honest it didn't really click with me when I read it for the first time.
Or the second time.
Or really ever.
So I embarked on a hunt of Rust's own code to learn about the birds and the bees of slices: how are they born?
I found a method that claimed to create one from thin air ([`std::slice::from_raw_parts`](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html)) and went from there:
```rust
pub const unsafe fn from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
    unsafe {
        // ...
        // Some safety stuff we don't care about right now...
        // ...
        &*ptr::slice_from_raw_parts(data, len)
    }
}
```
Ok, so far so good: we want a slice, we know where the data lies (`data`) and the amount of data we have (`len`) and we call a function that gives us a slice that we dereference and then borrow.
So what's that [`slice_from_raw_parts`](https://doc.rust-lang.org/stable/std/ptr/fn.slice_from_raw_parts.html) doing?
```rust
pub const fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
    from_raw_parts(data, len)
}
```
It's... a wrapper.
That returns a pointer to a slice.
Kinda disappointing, huh?
But what's this `from_raw_parts`, then?
How did we **make** that slice?
We all know we "cannot return value referencing local variable" [[1](https://stackoverflow.com/questions/32682876/is-there-any-way-to-return-a-reference-to-a-variable-created-in-a-function), [2](https://stackoverflow.com/questions/43079077/proper-way-to-return-a-new-string-in-rust), [3](https://stackoverflow.com/questions/29428227/return-local-string-as-a-slice-str), [4](https://stackoverflow.com/questions/32300132/why-cant-i-store-a-value-and-a-reference-to-that-value-in-the-same-struct)] because the Rust compiler has yelled at us so many times about it.
So how on God's Green Earth did the Rust language itself just return a pointer to a slice that was created in a local function?
Is it cheating?
Divine intervention?
I had to know.

So let's look at [`from_raw_parts`](https://doc.rust-lang.org/std/ptr/fn.from_raw_parts.html)[^1]:
```rust
pub const fn from_raw_parts<T: ?Sized>(
    data_pointer: *const impl Thin,
    metadata: <T as Pointee>::Metadata,
) -> *const T { // <-- Shouldn't this be a *const [T]??
    // Intrinsics magic
}
```
When I saw this definition my head spun.
This `from_raw_parts` function returns... a pointer to `T`?
A plain old pointer?
But then how does `slice_from_raw_parts`, without any additional magic, return a pointer to a *slice*?
Where did the *slice* come from?

And that's when it clicked: a slice *is a pointer*.
A *fat* pointer, which "magically" contains the length of the data within the pointer type itself.
Rust performs a perfectly legal type cast when it calls that final function, switching the return type on you, but `&[T]` is synctactic sugar for a `*const T` with some metadata packed in there.

### Let's Just Copy Vec and Slice
Alright, so we know that a slice is a DST, a reference to a slice is a fat pointer, and `Vec`s own and manage the memory that slices point to.
Life is great, because our path forward is clear: we'll make our own DST and fat pointer types, shove some multidimensional information in place of just a length, write a `Vec`-like owning structure for managing our data, and voilà[^2] we've got ourselves `ndarray`.

Except Rust [has no custom DSTs](https://doc.rust-lang.org/nomicon/exotic-sizes.html) or fat pointers on stable[^3].

This means we can't mirror Rust's `Vec<T>` and `[T]` precisely.
So maybe we can build up something that *almosts* acts like that pairing.
The idea is to have one kind of type that acts as the owning data structure, and another type that acts as a referencing data structure.

As we get a solid idea of some fundamentals for `ndarray`'s design, I'll call them out like this:
> [!TIP]
> `ndarray` should have some concept of an "owning" structure and some concept of a "referencing" structure. The referencing structure should carry information about the lifetime and mutability of the data.

There's another idea we want to borrow from the `Vec`/`[T]` duo: [`Deref` implementations](https://doc.rust-lang.org/beta/book/ch15-02-deref.html).
The [Array Reference Type RFC](https://github.com/rust-ndarray/ndarray/issues/879) on GitHub[^4] by Jim Turner makes several good arguments that `ndarray`'s owning data type should implement `Deref` with the referencing type as its target.
The RFC has a lot of rich detail, but I'd like to pull out what I think are the three best arguments for this design:
1. Integration into the Rust ecosystem by analogy to other smart pointer types
2. Expression of function arguments (although we'll get back to that later)
3. In-place arithmetic operators

If we're using the referencing type as a function argument, that means we'd like to have most capability implemented on the reference type, rather than the owning type.
So we'll jot that down as our second design idea:
> [!TIP]
> The owning data structure must dereference to a referencing data structure.
> Whenever possible, functionality should be implemented on the reference type rather than the owning type.

> [!WARNING]
> Notice the word *must* in the design tip above.
> That little choice will have some major implications later in this design document, especially when it comes to discussing traits.
> But I want to stress that it's just that: a choice.
> This document is a proposal, of sorts, but I am 100% positive that at least one of the choices this document makes will be wrong.
> So, dear reader, please provide feedback!

What else can we steal from our dynamic duo?
Well, we just went into depth about how slices are just pointers, i.e., *\~raw pointers\~*[^5].
This is important because raw pointers let us do some important tricks that can be otherwise difficult to accomplish, like messing with lifetimes and getting aliasing pointers.
This turns out to be pretty beneficial for implementing functions like splitting arrays in half.
Unfortunately, without custom DSTs, we can't get actual raw pointers.
So what to do?
Here's where I'll introduce our last Alert convention: design questions that I consider particularly open.
> [!IMPORTANT]
> Should there be a third data structure that represents a "raw pointer array"?
> Stripped of lifetime information, this would act more like a `*const T` or `*mut T` than a `&[T]` or `&mut [T]`.
> I think this is most open because it's not 100% clear to me precisely what functionality this enables over just the regular reference type.
> Please provide feedback here!

[^0]: And that's not an exhaustive list! Names pulled from contributions and discussions on the `ndarray` GitHub page.
[^1]: An experimental API, so stable Rust is actually cheating here.
[^2]: Cue the [Zelda cooking sound](https://www.youtube.com/watch?v=-Bl6xL2it4w).
[^3]: The `ptr_metadata` experimental API starts to get there, but is still on nightly.
[^4]: Which got me into this whole design thinking in the first place.
[^5]: Cue the [Zelda blood moon music](https://www.youtube.com/watch?v=uAxD8-_6_rs). I've been playing a lot of Breath of the Wild. I know I'm seven years late.
