# `ndarrray` Design, from the Ground Up

Disclaimer: This README is not official for `ndarray`.
Its aim is to articulate, as clearly as possible, some design principles and goals that might be achieved through a redesign of `ndarray`'s core.

Second Disclaimer: This design document tries to build an understanding of `ndarray`'s (potential) design from first principles, as a sort of pedagogical exercise.
But the ideas in this document are the culmination of years of design, dialogue, and development, in particular by [bluss](https://github.com/bluss), [Jim Turner](https://github.com/jturner314), [Vincent Barrielle](https://github.com/vbarrielle), and [Luca Palmieri](https://github.com/LukeMathWalker)[^0].

## `ndarray` as a Multi-Dimensional Vec and Slice
Fundamentally, we might think of `ndarray`'s core aspiration as providing data structures that are multi-dimensional generalizations of Rust's standard `Vec<T>` and `&[T]`: a block of data with some information describing its "shape" (rather than just its length).
So this design document will start there.
Before you read further, make sure to familiarize yourself with the details of slices, especially their roles as Dynamically Sized Types and the concepts of fat pointers.
I've left code walkthrough that helped me think about this at the [bottom of the document](#a-totally-inadequate-primer-on-rusts-slices), but feel free to just keep going from here.

### Let's Just Copy Vec and Slice
Alright, so we know that a slice is a DST, a reference to a slice is a fat pointer, and `Vec`s own and manage the memory that slices point to.
Life is great, because our path forward is clear: we'll make our own DST and fat pointer types, shove some multidimensional information in place of just a length, write a `Vec`-like owning structure for managing our data, and voilà[^2] we've got ourselves `ndarray`.

Except Rust [has no custom DSTs](https://doc.rust-lang.org/nomicon/exotic-sizes.html) or fat pointers on stable[^3].

This means we can't mirror Rust's `Vec<T>` and `[T]` precisely.
So maybe we can build up something that *almosts* acts like that pairing.
The idea is to have one kind of type that acts as the owning data structure, and another type that acts as a referencing data structure.

As we get a solid idea of some fundamentals for `ndarray`'s design, I'll call them out like this:
> [!TIP]
> `ndarray` should have some concept of an "owning" structure and some concept of a "referencing" structure.

There's another idea we want to borrow from the `Vec`/`[T]` duo: [`Deref` implementations](https://doc.rust-lang.org/beta/book/ch15-02-deref.html).
Jim Turner's [Array Reference Type RFC](https://github.com/rust-ndarray/ndarray/issues/879)[^4] makes several good arguments for `ndarray`'s owning data type implementing `Deref` with the referencing type as its target.
The RFC has a lot of rich detail, but I'd like to pull out what I think are the three best arguments for this design:
1. Integration into the Rust ecosystem by analogy to other smart pointer types
2. Expression of function arguments (this will come up later)
3. In-place arithmetic operators

If we're using the referencing type as a function argument, that means we'd like to have most capability implemented on the reference type, rather than the owning type.
So we'll jot that down as our second design idea:
> [!TIP]
> The owning data structure must dereference to a referencing data structure.
> Whenever possible, functionality should be implemented on the reference type rather than the owning type.

<!-- > [!WARNING]
> Notice the word *must* in the design tip above.
> That little choice will have some major implications later in this design document, especially when it comes to discussing traits.
> But I want to stress that it's just that: a choice.
> This document is a proposal, of sorts, but I am 100% positive that at least one of the choices this document makes will be wrong.
> So, dear reader, please provide feedback! -->

What else can we steal from our dynamic duo?
Well, we slices are just pointers.
So what about *\~raw pointers\~*[^5].
Raw pointers let us do some important tricks that can be otherwise difficult to accomplish, like messing with lifetimes and getting aliasing pointers.
This turns out to be pretty beneficial for implementing functions like splitting arrays in half.
Unfortunately, without custom DSTs, we can't get actual raw pointers.
So what to do?
Here's where I'll introduce another formatting convention: design questions that I consider particularly open.
> [!IMPORTANT]
> Should there be a third data structure that represents a "raw pointer array"?
> Stripped of lifetime information, this would act more like a `*const T` or `*mut T` than a `&[T]` or `&mut [T]`.
> I think this is most open because it's not 100% clear to me precisely what functionality this enables over just the regular reference type.
> Please provide feedback here!

### References, Views, and C++'s `mdspan`
For those familiar with `ndarray`'s existing codebase / design, you'll know that "views" play a large role in both the library and its use.
These views represent non-owning looks into a multidimensional array.

Which sounds suspiciously like the reference type described above.

So should `ndarray` get rid of views and just use the reference type?
This design has a certain cleanliness to it: having both a reference type and a view can be a confusing API - users now have to learn the difference between two types of non-ownership - and may only be worth keeping if there are critical differences between the two.

As this section progresses, it's worth noting that C++ just had its first multi-dimensional array construct accepted into `std` in C++23.
The new [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) takes the stance that there should be just one type of non-owning multi-dimensional array.
We'll be revisiting parts of its design, which I personally think is very well thought out, as we continue to build up our `ndarray` internals.

To understand the design trade-offs with combining the concept of a "view" and a "reference", we have to look at fat pointers, `Deref`, and lifetimes:

#### Lack of Fat Pointers
Fat pointers have a number of advantages in a design of this kind, but one of them is the critical fact that Rust lets us pass around pointers (both thin and fat) as values without lifetimes.
This means that a function can build a fat pointer in its body, and return it as a reference to the constructed type with the proper lifetime attached.
(This is what happens with the call chain of `from_raw_parts` in the slice primer below, just in multiple steps.)

For `ndarray`, this limitation has a major consequence: you can't write a function that returns a *reference* to a non-owning type *that doesn't have the same shape and offset as the owning type*.
In other words, this signature:
```rust
fn deref<'a>(owner: &'a OwningType) -> &'a ReferenceType;
```
is only possible when `OwningType` and `ReferenceType` view the exact same data.
This is exactly what we want for `Deref`, but it's a big problem when we want to return a non-owning array that has a different shape, stride, or offset, say from the result of slicing.
For that behavior, we'd have to have a signature like
```rust
fn slice<'a>(owner: &'a OwningType) -> ReferenceType;
```
... but hold on.
Where'd the lifetime `'a` go?

#### Lifetimes
Unlike in C++, Rust must explicitly carry around the lifetimes of its values.
Going back to our `Vec<T>` / `&[T]` example, this is accomplished via the lifetime of the borrow: a `Vec` with lifetime `'a` will produce a slice with type `&'a [T]`.
The same happens with a `deref` to the reference type, above.

But once we start insisting on only one non-owning type, it becomes clear that (unlike for slices), we're going to have non-reference values of that type!
To make it concrete: `[T]` is exceedingly rare to see, with `&[T]` being the norm, but `ReferenceType` would have to be part of everyday use.
This is why `ndarray`'s existing `ArrayView` is generic on the lifetime: it acts _as if_ it's carrying a reference to the array into which it's peering.

So that's one option: change `ReferenceType` to be generic on lifetime (it's already going to need several other generics; we'll get to that later):
```rust
fn deref<'a>(owner: &'a OwningType) -> &'a ReferenceType<'a>;
fn slice<'a>(owner: &'a OwningType) -> ReferenceType<'a>;
```
... except that `deref` function signature is impossible: the `Deref` implementation would have to be
```rust
impl<'a> Deref for OwningType {
    type Target = ReferenceType<'a>;

    fn deref(owner: &OwningType) -> &ReferenceType<'a> {
        todo!();
    }
}
```
which leads to `'a` being an unconstrained lifetime parameter, which the Rust compiler will inform you is not allowed.
So a lifetime-generic reference type fails to meet one of our first criteria: being the `Deref` target for the owning type.

So, until custom DSTs and fat pointers are possible, `ndarray` must continue to have both a reference type (for `deref`) and a separate view type (for non-owning but non-reference arrays).

> [!TIP]
> `ndarray` must define a non-owning view type which can represent a look into a subset of an owning type, reference type, or another view type.
The view type must carry the appropriate lifetime and mutability information of the array from which is received its data.
Like the owning type, the view type should `deref` to the reference type.

In practice, since carrying mutability information around is a different type, `ndarray` does (and likely will continue to) have both an `ArrayView` and an `ArrayViewMut`.

### Mutability
The final topic in our `Vec` / slice analogy is mutability.
In the following signature:
```rust
fn mutate(slice: &mut [f64]);
```
the `mut` indicates that the function can (and probably will) muck around with the underlying data contained in the slice.
The function cannot, however, change the number of elements the slice refers to.
Maybe this is by design, but maybe not: the slice's status as a fat pointer means that the length is part of pointer metadata.
I'm not even sure what the rules are on the mutability of pointer metadata.

So how does this analogy extend to `ndarray`?
It seems that the most important question is whether
```rust
fn mutate(arr: &mut ReferenceType);
```
can mutate *both* the array data *and* the array shape, or *only* the array data.
The analogy to `&mut [T]` would imply the latter, but I think there's a very good argument for the former.
Up until now, we have established three types: owning, view, and reference types.
And we've established that both the owning and view types should `deref` to the reference type.
One of the major goals of that design is allowing users to write functions that operate on arrays using just the reference type as the input argument.
So what happens if users want to write a function that alters the shape of an array, in place?
Well, if the reference type only allows for data mutability, then users will have to write two functions:
```rust
fn mutate_owned(arr: &mut OwningType);
fn mutate_view(arr: &mut ViewType);
```
In the current `ndarray` design (0.16.1), this isn't a problem, since the owning type and the view type are both generic instantiations of `ArrayBase`, so only one function is needed.
However, I'd argue that keeping this kind of API would be even more confusing to users.
For reasons I'll discuss later, we already expect the owning, view, and reference types to have three generic parameters.
Keeping an `ArrayBase` concept that encompasses both the owning type and the view type would require a *fourth* generic, and the function signature would look like
```rust
fn mutate_base<O, S, D, A>(arr: &mut ArrayBase<O, S, D, A>);
```
where `O` is the "ownership" of the array, `S` is the storage (different from `ndarray`'s current storage concept, see below), `D` is the dimensionality, and `A` is the element type.
And god forbid they should want to write such a function that has more than one argument.

Instead, I'd suggest that the reference type allows for mutability of both data and shape.
If the reference type is `deref`'d from an owning type, then the function will change the owning type in-place.
If the reference type is `deref`'d from a view type, then the function will change the view's shape without changing the underlying owning type behind it.
It creates simple rules for function writers to follow: take in `&ReferenceType` to read the array, and `&mut ReferenceType` to write the array.
It also create simple rules for function consumers: if you call a mutable function on an owning type, expect that owning type to be changed.
If you call a mutable function on a view type, expect that view type to be changed.

> [!IMPORTANT]
> Should the mutability of array views and array references be different?
> In particular, should they share the same rules for mutability of both data and shape?
> What considerations have I missed here?

## Generic Parameters and C++23's `mdspan`
Ok, so we've gone through the parallels between `Vec`s and arrays and seen what the consequences are for our designs.
However, there's one place where the parallels totally break down: arrays are not just generic in type, but also in dimensionality (at least).
So we've already got two generics, `A`, and `D`.

`ndarray` currently (0.16.1) has a third generic parameter, `S`, for "storage".
At the moment, that storage type serves to pack three concepts into `ArrayBase`: "raw" array types, view types, and shared array data via `Arc` and `Cow`.
I'd like to argue that while the storage generic has an important place, the current implementation packs in too many ideas.
In particular, the fact that an array is a view should not be considered part of storage, and should instead be handled by a separate struct.

The storage type would then be free to implement something closer to `mdspan`'s `AccessorPolicy`.
In that design, the `LayoutPolicy` (which would likely be the `D` generic type here) is responsible for turning an index into a linear offset from the "origin" or "start" of the array (or array view).
If `D` played that role in `ndarray`, then the storage type could act like `AccessorPolicy`, responsible for turning that linear offset into a reference for the correct position in the array.

I should be clear that a complete copy of `mdspan` is neither possible nor desirable; it's a standard designed for C++, and will not translate to idiomatic Rust on a 1:1 basis.
But I do think that the separation of "index to offset" and "offset to reference" is a smart design.

> [!IMPORTANT]
> Is following `mdspan`'s design ill-advised?
> If it is a good idea, what should the specifics of the traits be that enable / enforce that design?

# A Totally Inadequate Primer on Rust's Slices
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
But what's this `ptr::from_raw_parts`, then?
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


[^0]: And that's not an exhaustive list! Names pulled from contributions and discussions on the `ndarray` GitHub page.
[^1]: An experimental API, so stable Rust is actually cheating here.
[^2]: Cue the [Zelda cooking sound](https://www.youtube.com/watch?v=-Bl6xL2it4w).
[^3]: The `ptr_metadata` experimental API starts to get there, but is still on nightly.
[^4]: Which got me into this whole design thinking in the first place.
[^5]: Cue the [Zelda blood moon music](https://www.youtube.com/watch?v=uAxD8-_6_rs). I've been playing a lot of Breath of the Wild. I know I'm seven years late.
