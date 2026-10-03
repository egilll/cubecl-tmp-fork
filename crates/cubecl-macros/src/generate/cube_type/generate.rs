use proc_macro2::TokenStream;
use quote::quote;
use syn::{Generics, Ident, parse_quote};

use crate::{parse::cube_type::CubeType, paths::prelude_type};

impl CubeType {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        match self {
            CubeType::Enum(data) => data.generate(with_launch),
            CubeType::Struct(data) => data.generate(with_launch),
        }
    }
}

/// Lets an expand type be dereferenced and reborrowed like the type it expands: `*x` copies it
/// into fresh variables, and `&mut x` can be passed where `&x` is expected.
pub(crate) fn reference_impls(name_expand: &Ident, generics: &Generics) -> TokenStream {
    let scope = prelude_type("Scope");
    let deref = prelude_type("DerefExpand");
    let clone = prelude_type("ExpandTypeClone");
    let into_mut = prelude_type("IntoMut");

    let (impl_generics, generic_names, where_clause) = generics.split_for_impl();
    let mut ref_generics = generics.clone();
    ref_generics.params.insert(0, parse_quote!['__a]);
    let (ref_impl_generics, _, _) = ref_generics.split_for_impl();

    quote! {
        impl #impl_generics #deref for #name_expand #generic_names #where_clause {
            type Target = Self;

            fn __expand_deref_method(&self, scope: &#scope) -> Self {
                #into_mut::into_mut(#clone::clone_unchecked(self), scope)
            }
        }

        // Every argument is passed through `Into`, which Rust's reborrow does not reach.
        impl #ref_impl_generics ::core::convert::From<&'__a mut #name_expand #generic_names>
            for &'__a #name_expand #generic_names #where_clause
        {
            fn from(value: &'__a mut #name_expand #generic_names) -> Self {
                value
            }
        }
    }
}
