# Struct Support

CubeCL provides robust support for Rust structs, allowing you to organize and modularize your kernel
code with zero-cost abstractions. Structs can be used as kernel arguments, returned from kernels, or
as intermediate types within your GPU code. This enables you to write idiomatic, maintainable Rust
code that maps efficiently to GPU kernels.

## Defining structs

To use a struct in a CubeCL kernel, simply derive the required traits on the struct that you want to
use:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub struct Pair<T: LaunchArg> {
    pub left: T,
    pub right: T,
}
```

- `CubeType` enables the struct to be used as a CubeCL type in a kernel.
- `CubeLaunch` allows the struct to be used as a kernel argument or return type.

Structs can contain other structs, arrays, or generic parameters, as long as all fields implement
the required CubeCL traits. Generics are also supported, allowing you to create reusable types that
can be instantiated with different types.

## Using structs in kernels

Structs can be passed as kernel arguments if annotated with `CubeLaunch`, returned from kernels, or
used as local variables:

```rust,ignore
# use cubecl::prelude::*;
#
# #[derive(CubeType, CubeLaunch)]
# pub struct Pair<T: LaunchArg> {
#     pub left: T,
#     pub right: T,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_struct_example(pair: Pair<Box<[f32]>>, output: &mut [f32]) {
    let pos = UNIT_POS as usize;
    output[pos] = pair.left[pos] + pair.right[pos];
}
#
# pub fn launch(device: &cubecl::Device) {
#     let client = device.client();
#
#     let left = client.create_from_slice(f32::as_bytes(&[1.0]));
#     let right = client.create_from_slice(f32::as_bytes(&[1.0]));
#     let output = client.empty(core::mem::size_of::<f32>());
#
#     unsafe {
#         kernel_struct_example::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(1),
#             PairLaunch::new(
#                 BufferArg::from_raw_parts(left, 1),
#                 BufferArg::from_raw_parts(right, 1),
#             ),
#             BufferArg::from_raw_parts(output.clone(), 1),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_example with runtime {:?} => {:?}",
#         client.name(),
#         f32::from_bytes(&client.read_one(output).unwrap())
#     );
# }
#
# fn main() {
#     launch(&Default::default());
# }
```

You can also mutate struct fields if the struct is passed as a mutable reference:

```rust,ignore
# use cubecl::prelude::*;
#
# #[derive(CubeType, CubeLaunch)]
# pub struct Pair<T: CubeType> {
#     pub left: T,
#     pub right: T,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_struct_mut(output: Pair<&mut [f32]>) {
    let pos = UNIT_POS as usize;
    output.left[pos] = 42.0;
    output.right[pos] = 3.14;
}
#
# pub fn launch(device: &cubecl::Device) {
#     let client = device.client();
#
#     let left = client.create_from_slice(f32::as_bytes(&[1.0]));
#     let right = client.create_from_slice(f32::as_bytes(&[1.0]));
#
#     unsafe {
#         kernel_struct_mut::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(1),
#             PairLaunch::new(
#                 BufferArg::from_raw_parts(left.clone(), 1),
#                 BufferArg::from_raw_parts(right.clone(), 1),
#             ),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_mut with runtime {:?} => ({:?}, {:?})",
#         client.name(),
#         f32::from_bytes(&client.read_one(left).unwrap()),
#         f32::from_bytes(&client.read_one(right).unwrap()),
#     );
# }
#
# fn main() {
#     launch(&Default::default());
# }
```

## Comptime fields

You can mark struct fields as comptime, which means their values are known at kernel compilation
time and can be used for specialization:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub struct TaggedSlice {
    pub array: Box<[f32]>,
    #[cube(comptime)]
    pub tag: String,
}

#[cube(launch_unchecked)]
pub fn kernel_with_tag(output: &mut TaggedSlice) {
    if UNIT_POS == 0 {
        if comptime! {&output.tag == "zero"} {
            output.array[0] = 0.0;
        } else {
            output.array[0] = 1.0;
        }
    }
}
#
# pub fn launch(device: &cubecl::Device) {
#     let client = device.client();
#
#     let output = client.empty(core::mem::size_of::<f32>());
#
#     unsafe {
#         kernel_with_tag::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(1),
#             TaggedSliceLaunch::new(
#                 BufferArg::from_raw_parts(output.clone(), 1),
#                 "not_zero".to_string(),
#             ),
#         )
#     };
#
#     println!(
#         "Executed kernel_with_tag with runtime {:?} => {:?}",
#         client.name(),
#         f32::from_bytes(&client.read_one(output).unwrap())
#     );
# }
#
# fn main() {
#     launch(&Default::default());
# }
```

## Copies and references

Structs are borrowed, reborrowed and copied as in Rust. A `&mut` reference can be passed where a `&`
is expected, and `*x` copies a `Copy` struct into fresh variables, so writing the copy leaves the
original alone:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, Clone, Copy)]
pub struct Memory {
    pub stability: f32,
    pub difficulty: f32,
}

#[cube]
fn retention(memory: &Memory) -> f32 {
    memory.stability / memory.difficulty
}

#[cube]
fn review(memory: &mut Memory) -> f32 {
    let before = *memory;
    memory.stability *= 2.0;
    retention(memory) - retention(&before)
}
#
# fn main() {}
```

Assigning a whole struct, as in `*memory = other`, also needs `#[derive(CubeTypeMut)]`.

## Adding methods to struct

You can implement methods for structs using the `#[cube]` attribute. Please note that the `#[cube]`
attribute must be on the impl block. Here's an example:

```rust,ignore
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub struct Pair<T: LaunchArg> {
    pub left: T,
    pub right: T,
}

#[cube]
impl Pair<Box<[f32]>> {
    pub fn sum(&self, index: usize) -> f32 {
        self.left[index] + self.right[index]
    }
}

#[cube(launch_unchecked)]
pub fn kernel_struct_example(pair: &Pair<Box<[f32]>>, output: &mut [f32]) {
    output[UNIT_POS as usize] = pair.sum(UNIT_POS as usize);
}
#
# pub fn launch(device: &cubecl::Device) {
#     let client = device.client();
#
#     let left = client.create_from_slice(f32::as_bytes(&[1.0]));
#     let right = client.create_from_slice(f32::as_bytes(&[1.0]));
#     let output = client.empty(core::mem::size_of::<f32>());
#
#     unsafe {
#         kernel_struct_example::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(1),
#             PairLaunch::new(
#                 BufferArg::from_raw_parts(left, 1),
#                 BufferArg::from_raw_parts(right, 1),
#             ),
#             BufferArg::from_raw_parts(output.clone(), 1),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_example with runtime {:?} => {:?}",
#         client.name(),
#         f32::from_bytes(&client.read_one(output).unwrap())
#     );
# }
#
# fn main() {
#     launch(&Default::default());
# }
```
