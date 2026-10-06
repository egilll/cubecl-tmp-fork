use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_quote};

use crate::{parse::cube_type::CubeTypeStruct, paths::prelude_type};

pub fn generate(input: &DeriveInput) -> syn::Result<TokenStream> {
    let mut transparent = false;
    for attr in &input.attrs {
        if attr.path().is_ident("repr") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("transparent") {
                    transparent = true;
                }
                Ok(())
            })?;
        }
    }
    if !transparent {
        return Err(syn::Error::new_spanned(
            input,
            "DeviceRepr requires #[repr(transparent)]",
        ));
    }
    let parsed = CubeTypeStruct::from_derive_input(input)?;
    let runtime: Vec<_> = parsed
        .fields
        .iter()
        .filter(|field| !field.comptime.is_present())
        .collect();
    if runtime.len() != 1
        || parsed
            .fields
            .iter()
            .any(|field| field.comptime.is_present() && !field.is_marker())
    {
        return Err(syn::Error::new_spanned(
            input,
            "DeviceRepr requires one native field and only comptime PhantomData markers",
        ));
    }
    let field = runtime[0];
    let member = field.member();
    let repr = &field.ty;
    let markers: Vec<_> = parsed
        .fields
        .iter()
        .filter(|field| field.is_marker())
        .map(|field| field.member())
        .collect();
    let name = &parsed.ident;
    let expanded = parsed.name_expand.as_ref().unwrap();
    let device_repr = prelude_type("DeviceRepr");
    let cube = prelude_type("CubeType");
    let primitive = prelude_type("CubePrimitive");
    let (_, type_generics, _) = parsed.generics.split_for_impl();
    let mut generics = parsed.generics.clone();
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#repr: #primitive));
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(Self: #cube<ExpandType = #expanded #type_generics> + 'static));
    let (impl_generics, ty_generics, clause) = generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics #device_repr for #name #ty_generics #clause {
            type Repr = #repr;

            fn from_repr(value: Self::Repr) -> Self {
                Self { #member: value, #(#markers: ::core::marker::PhantomData,)* }
            }

            fn into_repr(self) -> Self::Repr { self.#member }

            fn expand_from_repr(value: <Self::Repr as #cube>::ExpandType) -> Self::ExpandType {
                #expanded { #member: value, #(#markers: ::core::marker::PhantomData,)* }
            }

            fn expand_into_repr(value: Self::ExpandType) -> <Self::Repr as #cube>::ExpandType {
                value.#member
            }
        }
    })
}
