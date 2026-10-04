# Migrating from 0.10

The changes most kernels written for 0.10 need. Everything else is either unchanged or reported by
the compiler with a direct suggestion.

## Buffers are slices

A kernel takes its buffers as slices rather than `Array`, and launches them with `BufferArg`:

```rust,ignore
// 0.10
#[cube(launch)]
fn scale(input: &Array<f32>, output: &mut Array<f32>) {
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
}
// ArrayArg::from_raw_parts(handle, len)

// 0.11
#[cube(launch)]
fn scale(input: &[f32], output: &mut [f32]) {
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
}
// BufferArg::from_raw_parts(handle, len)
```

A struct that owns a buffer holds a `Box<[T]>`. `Array` remains for arrays declared inside a kernel,
with `Array::new(len)`.

## Shared memory

`SharedMemory::<T>::new(len)` is now `Shared::<[T]>::new_slice(len)`, or
`Shared::<[T]>::new_aligned_slice(len, alignment)`. The result is used like any other slice.

## The client

`ComputeClient<R>` is now `Client`, which is not generic over the runtime. A `cubecl::Device` gives
one with `device.client()`, and `R::client(&device)` still does.

## Structs

A `CubeType` struct is borrowed, reborrowed and copied as in Rust: see
[Copies and references](./language-support/struct.md#copies-and-references). Assigning a whole struct
needs `#[derive(CubeTypeMut)]`.

## Functions

A `#[cube]` function is now a device function, traced once and called, wherever its signature
allows; `#[cube(inline)]` restores tracing it at every call. See
[Functions](./language-support/functions.md). A type that implements `CubeType` by hand needs its
expand type to implement `CallArg`, usually as the empty `impl CallArg for MyTypeExpand {}`.
