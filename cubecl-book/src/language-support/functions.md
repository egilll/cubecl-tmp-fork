# Functions

A `#[cube]` function called from a kernel becomes a device function: it is traced once per
specialization and called from every call site, instead of being traced again and pasted into the
kernel at each call. A helper called from 64 places is compiled once, which can make the driver's
compilation many times faster.

Whether a call survives to the generated code is decided per target. CubeCL inlines a call when it
must (the target can't express it, or the function uses something only the kernel's own body can,
such as a barrier, a plane operation or an atomic), or when it's free (a handful of operations, or a
single call site). Otherwise the call is kept, and the
target's own compiler decides whether to inline it.

## Specialization

A function is traced again for every distinct combination of:

- its generic types and const generics;
- its comptime arguments, which must be `Hash` to be told apart;
- the types of its runtime arguments, and the values of the constant ones (a literal argument is
  part of the specialization, not a parameter).

## When a call is traced inline

Some calls are traced inline, as before, without any error:

- a function taking `&mut` arguments other than slices, `impl Trait` arguments, or returning a
  reference or anything borrowing its arguments;
- an argument that isn't a scalar, a vector, a slice, or a struct of them, such as an array, shared
  memory or a tensor;
- a comptime argument whose type isn't known to be `Hash` (for example an unbounded generic);
- a function that returns a struct, terminates the kernel, or uses a local of its caller.

## Controlling it

The `inline` argument works like Rust's `#[inline]`:

```rust
#[cube(inline)] // Trace at every call, as `inline(always)`.
fn add_one(x: f32) -> f32 {
    x + 1.0
}

#[cube(inline(never))] // Always a call; a signature that can't be one is an error.
fn heavy(x: f32, y: f32) -> f32 {
    f32::exp(x) * y + f32::powf(x, y)
}
```

On a `#[cube] impl` block it applies to every method, and a method's own `#[cube(inline...)]`
overrides it. `CUBECL_INLINE=all` (or `inline = "all"` in the `[compilation]` configuration)
inlines every call on every target, which gives the code CubeCL generated before device functions.

## Types

Every expand type implements `CallArg`, which says whether and how a value is passed to a function.
`#[derive(CubeType)]` implements it, passing a struct as its runtime fields. A type implementing
`CubeType` by hand needs an implementation too; the empty one traces any call taking it inline:

```rust
impl CallArg for MyTypeExpand {}
```
