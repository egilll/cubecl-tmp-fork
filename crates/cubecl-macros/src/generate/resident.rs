//! `#[derive(Resident)]`: the owned host mirror of a launchable struct.
//!
//! For `struct S` with `#[derive(CubeType, CubeLaunch, Resident)]` it
//! generates `SResident`, one owned value per launched field: a
//! `StorageBuffer<Q>` for a `Table`, `Column` or `Storage` of `Q`, a
//! `ResidentGrid<Q>` for a `Grid`, `GridTable` or `GridColumn` of `Q`, the
//! field's own `Resident` for a field marked `#[resident(nested)]`,
//! `Host<E>` for a field marked `#[resident(with = Host)]` whose type's
//! first argument is `E` (`Host` provides `arg` and `bytes`), and the value
//! itself for anything else (scalars). `SResident::arg` is the kernel
//! argument, `SResident::bytes` the device memory held, and `SResident`
//! binds as `S` through `Bind`. Construction is a struct literal of
//! uploaded buffers.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, GenericArgument, PathArguments, Type};

enum Kind {
    Storage(Type),
    Grid(Type),
    Nested(Type),
    With(syn::Path, Type),
    Scalar(Type),
}

fn first_argument(segment: &syn::PathSegment) -> Option<Type> {
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    arguments.args.iter().find_map(|argument| match argument {
        GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    })
}

fn kind(field: &syn::Field) -> Option<Kind> {
    let comptime = field.attrs.iter().any(|attribute| {
        attribute.path().is_ident("cube")
            && attribute
                .parse_args::<syn::Ident>()
                .is_ok_and(|ident| ident == "comptime")
    });
    if comptime {
        return None;
    }
    let nested = field.attrs.iter().any(|attribute| {
        attribute.path().is_ident("resident")
            && attribute
                .parse_args::<syn::Ident>()
                .is_ok_and(|ident| ident == "nested")
    });
    let ty = field.ty.clone();
    if nested {
        return Some(Kind::Nested(ty));
    }
    let with = field.attrs.iter().find_map(|attribute| {
        if !attribute.path().is_ident("resident") {
            return None;
        }
        let mut host = None;
        attribute
            .parse_nested_meta(|meta| {
                if meta.path.is_ident("with") {
                    host = Some(meta.value()?.parse::<syn::Path>()?);
                }
                Ok(())
            })
            .ok()?;
        host
    });
    if let Some(host) = with {
        let element = match &ty {
            Type::Path(path) => path.path.segments.last().and_then(first_argument),
            _ => None,
        };
        return element.map(|element| Kind::With(host, element));
    }
    if let Type::Path(path) = &ty
        && let Some(segment) = path.path.segments.last()
    {
        let name = segment.ident.to_string();
        if let Some(element) = first_argument(segment) {
            match name.as_str() {
                "Table" | "Column" | "Storage" => return Some(Kind::Storage(element)),
                "Grid" | "GridTable" | "GridColumn" => return Some(Kind::Grid(element)),
                _ => {}
            }
        }
    }
    Some(Kind::Scalar(ty))
}

fn resident_of(ty: &Type) -> Type {
    let mut ty = ty.clone();
    if let Type::Path(path) = &mut ty
        && let Some(segment) = path.path.segments.last_mut()
    {
        segment.ident = format_ident!("{}Resident", segment.ident);
    }
    ty
}

fn launch_of(ty: &Type) -> Type {
    let mut ty = ty.clone();
    if let Type::Path(path) = &mut ty
        && let Some(segment) = path.path.segments.last_mut()
    {
        segment.ident = format_ident!("{}Launch", segment.ident);
    }
    ty
}

/// The owned host mirror of a launchable struct.
pub fn generate(input: &DeriveInput) -> syn::Result<TokenStream> {
    let input = input.clone();
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input, "Resident derives only for structs"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(&input, "Resident needs named fields"));
    };
    let prelude = crate::paths::prelude_path();
    let vis = &input.vis;
    let name = &input.ident;
    let resident = format_ident!("{name}Resident");
    let launch = launch_of(&syn::parse_quote!(#name));
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let mut declarations = Vec::new();
    let mut arguments = Vec::new();
    let mut bytes = Vec::new();
    let mut bounds = Vec::new();
    for field in &fields.named {
        let Some(kind) = kind(field) else { continue };
        let ident = field.ident.as_ref().expect("named");
        let field_vis = &field.vis;
        let docs = field.attrs.iter().filter(|a| a.path().is_ident("doc"));
        match kind {
            Kind::Storage(element) => {
                bounds.push(quote! { #prelude::StorageElement<#element>: #prelude::CubeElement });
                declarations.push(
                    quote! { #(#docs)* #field_vis #ident: #prelude::StorageBuffer<#element> },
                );
                arguments.push(quote! { (&self.#ident).into() });
                bytes.push(quote! { #prelude::ResidentBytes::bytes(&self.#ident) });
            }
            Kind::Grid(element) => {
                bounds.push(quote! { #prelude::StorageElement<#element>: #prelude::CubeElement });
                declarations.push(quote! { #(#docs)* #field_vis #ident: #prelude::ResidentGrid<#element> });
                arguments.push(quote! { self.#ident.arg() });
                bytes.push(quote! { #prelude::ResidentBytes::bytes(&self.#ident.values) });
            }
            Kind::With(host, element) => {
                declarations.push(quote! { #(#docs)* #field_vis #ident: #host<#element> });
                arguments.push(quote! { self.#ident.arg() });
                bytes.push(quote! { self.#ident.bytes() });
            }
            Kind::Nested(ty) => {
                let nested = resident_of(&ty);
                declarations.push(quote! { #(#docs)* #field_vis #ident: #nested });
                arguments.push(quote! { self.#ident.arg() });
                bytes.push(quote! { self.#ident.bytes() });
            }
            Kind::Scalar(ty) => {
                declarations.push(quote! { #(#docs)* #field_vis #ident: #ty });
                arguments.push(quote! { self.#ident.clone().into() });
            }
        }
    }
    let launch_type = quote! { #launch #type_generics };
    let declared = &input.generics.params;
    let predicates = where_clause
        .map(|clause| clause.predicates.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    let where_all = quote! { where #(#predicates,)* #(#bounds,)* };
    Ok(quote! {
        /// The owned host mirror of [`#name`].
        #[derive(Clone)]
        #vis struct #resident <#declared> #where_all {
            #(#declarations,)*
        }

        impl #impl_generics #resident #type_generics #where_all {
            /// As a kernel argument.
            #[must_use]
            #vis fn arg(&self) -> #launch_type {
                #launch::new(#(#arguments),*)
            }

            /// Device memory held, in bytes.
            #[must_use]
            #vis fn bytes(&self) -> usize {
                0 #(+ #bytes)*
            }
        }

        impl #impl_generics #prelude::Bind for #resident #type_generics #where_all {
            type Device = #name #type_generics;

            fn arg(&self) -> #launch_type {
                #resident::arg(self)
            }
        }
    })
}
