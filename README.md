# `ndarrray` Design, from the Ground Up

Disclaimer: This README is not official for `ndarray`.
Its aim is to articulate, as clearly as possible, some design principles and goals that might be achieved through a redesign of `ndarray`'s core.

Second Disclaimer: This design document tries to build an understanding of `ndarray`'s (potential) design from first principles, as a sort of pedagogical exercise.
But the ideas in this document are the culmination of years of design, dialogue, and development, in particular by [bluss](https://github.com/bluss), [Jim Turner](https://github.com/jturner314), [Vincent Barrielle](https://github.com/vbarrielle), and [Luca Palmieri](https://github.com/LukeMathWalker)[^0].

## `ndarray` as a Multi-Dimensional Vec and Slice
Fundamentally, we might think of `ndarray`'s core aspiration as providing data structures that are multi-dimensional generalizations of Rust's standard `Vec<T>` and `&[T]`: a block of data with some information describing its "shape" (rather than just its length).
So this design document will start there.
(But will go further, discussing how to design types that support other aspirations, like stack-allocated arrays.)
Before you read further, make sure to familiarize yourself with the details of slices, especially their roles as Dynamically Sized Types and the concepts of fat pointers.
I've left a code walkthrough that helped me think about this at the [bottom of the document](#a-totally-inadequate-primer-on-rusts-slices), but feel free to just keep going from here.

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
2. Expression of function arguments (this is critical)
3. In-place arithmetic operators (allowing the `*val = ...` syntax)

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

### References, Views, and C++'s `mdspan`
For those familiar with `ndarray`'s existing codebase / design, you'll know that "views" play a large role in both the library and its use.
These views represent non-owning looks into a multidimensional array.

Which sound suspiciously like the reference type described above.

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

For `ndarray`, the "no fat pointers" limitation has a major consequence: you can't write a function that returns a *reference* to a non-owning type *that doesn't have the same shape and offset as the owning type*.
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

But once we start insisting on only one non-owning type, it becomes clear that (unlike for slices), we won't just have references to that type!
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

So, until custom DSTs and fat pointers are possible, `ndarray` must continue to have both a reference type (for `deref`) and a separate view type (for non-owning arrays with a different shape from their owned parent).

> [!TIP]
> `ndarray` must define a non-owning view type which can represent a look into a subset of an owning type, reference type, or another view type.
The view type must carry the appropriate lifetime and mutability information of the array from which is received its data.
Like the owning type, the view type should `deref` to the reference type.

In practice, since carrying mutability information around is a different type, `ndarray` does (and likely will continue to) have both an `ArrayView` and an `ArrayViewMut`.

### Mutability
The next topic in our `Vec` / slice analogy is mutability.
In the following signature:
```rust
fn mutate(slice: &mut [f64]);
```
the `mut` indicates that the function can (and probably will) muck around with the underlying data contained in the slice.
The function cannot, however, change the number of elements the slice refers to.
(Maybe this is by design, but maybe not: the slice's status as a fat pointer means that the length is part of pointer metadata.
I'm not even sure what the rules are on the mutability of pointer metadata.)

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
But as we progress, we'll realize that we'd prefer that the owning type and the view type *not* be the same (generic) type.

Instead, I'd suggest that the reference type allows for mutability of both data and shape.
That means the following:
- Almost all functionality can be implemented on the reference type.
- It is transparent to function callers what is being changed: the value they pass in will be altered.
- It is transparent to function writers what they are changing: whatever the reference type refers to.
- It creates simple rules for function writers to follow: take in `&ReferenceType` to read the array, and `&mut ReferenceType` to write the array.
This all culminates in the following:
```rust
mutate(&owned);             // Mutates the owned value
mutate(&owned.view_mut());  // Mutates the shape of the view, but the data of the owned
```

> [!TIP]
> `ndarray`'s reference type(s) should allow for mutability of both shape and data; this mutability will pass through to the owning or view from which the reference type is derived.

### Raw Pointers
The final topic in our analogy is *\~raw pointers\~*[^5].
We know slices are just pointers.
So what about raw pointers for our arrays?
Raw pointers let us do some important tricks that can be otherwise difficult to accomplish, like messing with lifetimes and getting aliasing pointers.
This turns out to be pretty beneficial for implementing functions like splitting arrays in half or dealing with uninitialized data.
Unfortunately, without custom DSTs, we can't get actual raw pointers.
So what to do?
The current convention (0.16.1) is to introduce raw *views*, which just act *as if* they contain a raw pointer.
This lets you (unsafely) do what you need to do.

However, we can't have our raw views `deref` to our standard reference type; since most capability is implemented on the reference type, we'll end up with raw views having a ton of unsafe capability.
Instead, we'll introduce the convention of a raw reference type, and have raw views `deref` to that.
We're starting to collect types quickly, but we can help users abstract away from this by also having our standard reference type `deref` to the raw reference type.
This allows us to have a layered hierarchy of capability.

This design has also been created to avoid the "data traits" that `ndarray` currently has.
After doing quite a bit of design, I had concluded that the data traits introduce complexity whose job is to deal with the fact that all arrays are generic variants of the same type.
I think the trade of hierarchical trait bounds for different types is worth it, but I could be wrong.

We'll now set in stone our need for raw views:
> [!TIP]
> `ndarray` must define a raw reference type.
Whenever possible, functionality that doesn't require array elements to safely dereference should be implemented on the raw reference type.
The regular reference type must `deref` to the raw reference type.

> [!TIP]
> `ndarray` must define a raw view type, which can represent a look into a subset of an owning type, reference type, or another view type.
This raw view type must *not* carry the lifetime of the array from which it received its data, although it must continue to carry its mutability.

### The Landscape So Far
Just to review, I've included a diagram that shows the current type landscape that we've created:
```
                  DerefMut  ┌──────────────┐               
              ┌─────────────│ Raw View Mut ◄─┐             
              │             └──────────────┘ │             
┌─────────────▼─┐  Deref    ┌──────────┐     │.raw_view_mut
│ Raw Reference ◄───────────│ Raw View ◄─────┤             
└───────▲───────┘           └──────────┘     │             
        │                                    │.raw_view    
        │         DerefMut  ┌──────────┐     │             
DerefMut│     ┌─────────────│ ViewMut  ◄─────┤             
        │     │             └──────────┘     │             
        │     │                              │.view_mut    
┌───────┴─────▼─┐  Deref    ┌──────────┐     │             
│ Reference     ◄───────────│ View     ◄─────┤             
└───────▲───────┘           └──────────┘     │             
        │        ┌────────┐                  │.view        
        └────────┤ Owning ┼──────────────────┘             
        DerefMut └────────┘                                
```

## Generic Parameters, C++23's `mdspan`, and Backends
Ok, so we've gone through the parallels between `Vec`s and arrays and seen what the consequences are for our designs.
However, there's one place where the parallels totally break down: arrays are not just generic in type, but also in dimensionality (at least).
So we've already got two generics, `A`, and `D`.

`ndarray` currently (0.16.1) has a third generic parameter, `S`, for "storage".
At the moment, that storage type serves to pack three concepts into `ArrayBase`: raw views, views, and shared array data via `Arc` and `Cow`.
I'd like to argue that while the storage generic has an important place, the current implementation packs in too many ideas.
In particular, the fact that an array is a view should not be considered part of storage, and should instead be handled by a separate struct, as above.

There has also been plenty of conversation around giving `ndarray` additional capabilities: GPU support is a common one, as are stack-allocated arrays or constant dimensions.
That last case can already be handled by the dimensionality generic, but the first two require specialization that `ndarray` isn't built for right now.
That specialization essentially needs to happen at two levels: the reference level and the ownership level.
Together, these two specializations constitute a sort of "backend" to `ndarray`.

`mdspan` does something a little similar, although it is only concerned with the reference type (the owning type `mdarray` did not make it into the C++ standard, yet).
It has a `LayoutPolicy` which is responsible for turning an index into a linear offset from the "origin" or "start" of the array; we could conceivable fold that into the dimensionality concept that `ndarray` already has.
It also has an `AccessorPolicy`, which is responsible for turning that linear offset into a reference for the correct position in the array.

While we could copy this design directly, I think we'd rather take advantage of Rust's type system to create an even more powerful concept: a `Backend` trait, with two associated types.
One for ownership, and one for referencing.
We'll guarantee that any owning array carries around an instance of the ownership associated type, and any referencing array carries around an instance of the referencing associated type.
The trait implementation would act as "glue" between these two, governing how they interact.
Then the reference, view, and owning types of `ndarray` could be generic on the `Backend`, allowing for a highly-customizable foundation while still allowing users to write their code generically.
We'll also include the element type in the backend, allowing implementors to specialize for specific element types.

As an example, the current `ndarray` design for the `Array` type has a backend that roughly looks like the following:
```rust
impl<A> Backend for (OwnedRepr<A>, NonNull<A>) {
    type Owned: OwnedRepr<A>;
    type Ref: NonNull<A>;
    type Elem: A;
    
    // Glue code demanded by `Backend`
    // ...
    fn ensure_unique<D>(arr: &mut ArrayBase<D, Self>)
    where
        Self: Sized,
        L: Layout,
    {
        // Empty, this is always uniquely held.
    }
}
```
(Although we'd probably have a unit struct, rather than that tuple type, for simplicity).

### `Arc`s, `Cow`s, and Shared Ownership (Oh My!)
`ndarray` currently supports a special kind of shared ownership / copy-on-write by way of `ArcArray` and `CowArray`.
These incorporate their eponymous smart pointer types (`Arc` and `Cow`, respectively) by only holding the *data* as shared, not the shape of the array.
This has several advantages: firstly, shapes tend to be relatively inexpensive allocations as compared to the array data itself, so you'd like to be able to change them easily without having to copy the underlying data.
Second, `ndarray` can handle the copy-on-write internally, allowing for an ergonomic copy-on-write rather than doing something like `Arc::make_mut(...)`.
(This is a challenge, however, as it currently requires developers to ensure the uniqueness of the underlying data whenever they go to mutate it).

So how would this shared ownership model work with the above backend design?
`Cow` has a relatively simple option: with the introduction of the reference type, the owning type (and the viewing type) can implement `Borrow<ReferenceType>`, allowing the reference type to implement `ToOwned`, allowing users to use `Cow<ReferenceType>`.
The difference from `CowArray` would be that the `Cow` wrapper would cover both the data and the shape, so changes to the shape would cause a data copy.
Either way, those traits should be implemented, so `Cow<ReferenceType>` will be an option.

The other option (coexisting with the `Cow` option above) is to embed `Cow` and `Arc` (and possibly others) within the backend, similar to how the current (0.16.1) design works.
The key to `ArcRepr` and `CowRepr` is, essentially, in their customization of `Clone` and in their logic for ensuring uniqueness via an in-place copy of their data.
So an `Arc` backend would look as follows:
```rust
impl<A> Backend for (Arc<OwnedRepr<A>>, NonNull<A>) {
    type Owned: Arc<OwnedRepr<A>>;
    type Ref: NonNull<A>;
    type Elem: A;
    
    // Glue code demanded by `Backend`
    // ...
    fn ensure_unique<D>(arr: &mut ArrayBase<D, Self>)
    where
        Self: Sized,
        L: Layout,
    {
        // The current logic for `ArcRepr`'s `ensure_unique`
        // ...
    }
}
```

There is one catch to this implementation: because `ensure_unique` operates at the ownership level, the `DerefMut` call from owned to reference would be the place to put a call to `ensure_unique`.
This has one major advantage: implementors of mutable functionality on the reference types can be ensured that they're dealing with uniquely held data, and don't have to remember calls to `ensure_unique` or guarantee that it's in the call stack; a tricky detail to miss.
However, we said above that we strongly believe the reference type's shape can be changed.
So we'll end up ensuring uniqueness even for just mutating the shape.

One way to alleviate this problem is simply re-implementing shape-changing (but data preserving) functions on the owning and view types.
It's a bit of a bother, but I think that extra work is worth the cleanliness of the APIs in all other cases.

## Generics Overview
So, looking at the diagram from [above](#the-landscape-so-far) and the `Backend` trait as suggested, what do the types actually look like?
I'd say the following (where I've chosen to use `L: Layout` rather than `D: Dimension`, to make it clear that we're not married to the particular `Dimension` trait implementation right now):
```rust
// Raw Reference
struct RawArrayRefBase<L, B: Backend> {
    layout: L,
    storage: B::Ref,
}

// Reference
struct ArrayRefBase<L, B: Backend>(RawArrayRefBase<L, B>);

// Owning
struct ArrayBase<L, B: Backend> {
    aref: ArrayRefBase<L, B>,
    own: B::Owned,
}

/// Views
// A view of an existing array.
struct ArrayViewBase<'a, L, B: Backend> {
    aref: ArrayRefBase<L, B>,
    life: PhantomData<&'a B::Elem>,
}

// A mutable view of an existing array
pub struct ArrayViewBaseMut<'a, L, B: Backend> {
    aref: ArrayRefBase<L, B>,
    life: PhantomData<&'a mut B::Elem>,
}

// A view of an array without a lifetime, and whose elements are not safe to dereference.
pub struct RawArrayViewBase<L, B: Backend> {
    aref: RawArrayRefBase<L, B>,
    life: PhantomData<*const B::Elem>,
}

// A mutable view of an array without a lifetime, and whose elements are not safe to dereference.
pub struct RawArrayViewBaseMut<L, B: Backend> {
    aref: RawArrayRefBase<L, B>,
    life: PhantomData<*mut B::Elem>,
}
```
With the `Deref` implementations described above, that constitutes the suggested `ndarray` new internals.

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
