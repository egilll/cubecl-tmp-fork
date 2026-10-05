use crate::{
    frontend::container::{buffer_len::expand_buffer_length_native, slice},
    prelude::*,
};
use alloc::boxed::Box;
use cubecl_zspace::Tiling;
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct BufferCompilationArg {
    pub inplace: Option<usize>,
}

/// Buffer representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct BufferBinding {
    pub handle: cubecl_runtime::server::BufferBinding,
    pub(crate) length: [usize; 1],
    /// The element type the buffer was made for, when it is known; a
    /// kernel slice of another element type refuses it.
    pub(crate) elem: Option<ElemType>,
}

pub enum BufferArg {
    /// The buffer is passed with a buffer handle.
    Handle {
        /// The buffer handle.
        handle: BufferBinding,
    },
    /// The buffer is aliasing another input buffer.
    Alias {
        /// The position of the input buffer.
        input_pos: usize,
        /// The length of the underlying handle
        length: [usize; 1],
    },
}

impl BufferArg {
    /// Create a new buffer argument.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts(handle, length),
            }
        }
    }
    /// Create a new buffer argument from a binding.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        binding: cubecl_runtime::server::BufferBinding,
        length: usize,
    ) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts_binding(binding, length),
            }
        }
    }

    /// This argument, declared to hold elements of `elem`: binding it to a
    /// kernel slice of another element type panics at launch.
    pub fn typed(self, elem: ElemType) -> Self {
        match self {
            BufferArg::Handle { mut handle } => {
                handle.elem = Some(elem);
                BufferArg::Handle { handle }
            }
            alias => alias,
        }
    }

    /// The buffer's length in elements, as it was declared.
    pub fn len(&self) -> usize {
        match self {
            BufferArg::Handle { handle } => handle.length[0],
            BufferArg::Alias { length, .. } => length[0],
        }
    }

    /// Whether the buffer was declared empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alias(input_pos: usize, length: usize) -> Self {
        Self::Alias {
            input_pos,
            length: [length],
        }
    }

    pub fn size(&self) -> usize {
        match self {
            BufferArg::Handle { handle } => handle.length[0],
            BufferArg::Alias { length, .. } => length[0],
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            BufferArg::Handle { handle } => &handle.length,
            BufferArg::Alias { length, .. } => length,
        }
    }
}

impl BufferBinding {
    /// Create a new buffer handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), length) }
    }

    /// Create a new buffer handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length or size, may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        handle: cubecl_runtime::server::BufferBinding,
        length: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
            elem: None,
        }
    }

    /// Return the handle as a tensor instead of a buffer.
    pub fn into_tensor(self) -> TensorBinding {
        let shape = self.length.into();

        TensorBinding {
            handle: self.handle,
            strides: [1].into(),
            shape,
            tiling: Tiling::UNTILED,
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Box<[C]> {
    type RuntimeArg = BufferArg;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        <[C]>::register(arg, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<Box<[C]>> {
        <[C]>::expand(arg, builder).expand.into()
    }
}

impl<C: CubePrimitive> LaunchArg for [C] {
    type RuntimeArg = BufferArg;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let (elem_size, expected) = launcher.with_scope(|scope| {
            (
                C::__expand_size(scope),
                scalar_elem(scope.ctx(), C::__expand_as_type(scope)),
            )
        });
        let inplace = match &arg {
            BufferArg::Handle { handle } => {
                if let (Some(elem), Some(expected)) = (handle.elem, expected) {
                    assert!(
                        elem == expected,
                        "a buffer of {elem} is bound to a kernel slice of {expected}"
                    );
                }
                None
            }
            BufferArg::Alias { input_pos, .. } => Some(*input_pos),
        };
        launcher.register_buffer(arg, elem_size);

        BufferCompilationArg { inplace }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<[C]> {
        let buffer = match arg.inplace {
            Some(id) => builder.inplace(id),
            None => builder.buffer(C::__expand_as_type(&builder.scope)),
        };
        let scope = &builder.scope;
        let len = expand_buffer_length_native(scope, buffer);
        let slice_var =
            slice::from_raw_parts::<C>(scope, buffer, 0usize.into_expand(scope), len.into());
        slice_var.expand.into()
    }
}

/// The scalar element under `ty`'s vector, atomic and pointer layers, when
/// it has one.
fn scalar_elem(ctx: &pliron::context::Context, ty: pliron::r#type::TypeHandle) -> Option<ElemType> {
    use cubecl_ir::interfaces::{HasElementType, ScalarType, ScalarizableType};
    use pliron::r#type::type_cast;
    let mut ty = ty;
    for _ in 0..8 {
        let deref = ty.deref(ctx);
        if let Some(scalar) = type_cast::<dyn ScalarType>(&*deref) {
            return Some(scalar.elem_type(ctx));
        }
        let next = if let Some(scalarizable) = type_cast::<dyn ScalarizableType>(&*deref) {
            Some(scalarizable.scalar_type(ctx))
        } else {
            type_cast::<dyn HasElementType>(&*deref).and_then(|inner| inner.element_type(ctx))
        };
        match next {
            Some(next) if next != ty => ty = next,
            _ => return None,
        }
    }
    None
}
