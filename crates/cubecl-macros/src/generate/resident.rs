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
//! binds as `S` through `Bind`; `S` implements `Residency`.
//!
//! Without `with` fields it also generates `SShape`: each buffer's element
//! count (a `GridShape` for grids), each nested field's shape and each
//! scalar's value. `SResident` implements `Shaped`: its `sizes` lists the
//! buffers in carving order and `bytes` sums them, so a value's footprint
//! is known before it exists, and `carve` makes every buffer from a
//! `Carve` in that order. Construction is a struct literal
//! of uploaded buffers, or a carve followed by writes.

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
        return Err(syn::Error::new_spanned(
            &input,
            "Resident derives only for structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &input,
            "Resident needs named fields",
        ));
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
                declarations
                    .push(quote! { #(#docs)* #field_vis #ident: #prelude::ResidentGrid<#element> });
                arguments.push(quote! { self.#ident.arg() });
                bytes.push(quote! { #prelude::ResidentBytes::bytes(&self.#ident.values) });
            }
            Kind::With(host, element) => {
                declarations.push(quote! { #(#docs)* #field_vis #ident: #host<#element> });
                arguments.push(quote! { self.#ident.arg() });
                bytes.push(quote! { self.#ident.bytes() });
            }
            Kind::Nested(ty) => {
                bounds.push(quote! { #ty: #prelude::Residency });
                bounds.push(quote! { <#ty as #prelude::Residency>::Resident: Clone });
                let nested = quote! { <#ty as #prelude::Residency>::Resident };
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
    let shape = shape(&input, &fields.named, &prelude)?;
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

        impl #impl_generics #prelude::Residency for #name #type_generics #where_all {
            type Resident = #resident #type_generics;
        }

        #shape
    })
}

/// Whether `tokens` name `ident` anywhere.
fn mentions(tokens: TokenStream, ident: &syn::Ident) -> bool {
    tokens.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(name) => name == *ident,
        proc_macro2::TokenTree::Group(group) => mentions(group.stream(), ident),
        _ => false,
    })
}

/// `SShape` and `Shaped for SResident`, unless a field is held `with` a
/// host type of its own. The shape keeps only the generic parameters its
/// nested and scalar fields name: a buffer's shape is a count.
fn shape(
    input: &DeriveInput,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    prelude: &syn::Path,
) -> syn::Result<TokenStream> {
    let vis = &input.vis;
    let name = &input.ident;
    let resident = format_ident!("{name}Resident");
    let shape = format_ident!("{name}Shape");
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let mut declarations = Vec::new();
    let mut sizes = Vec::new();
    let mut carved = Vec::new();
    let mut cloned = Vec::new();
    let mut bounds = Vec::new();
    let mut shape_bounds = Vec::new();
    let mut named = Vec::new();
    for field in fields {
        let Some(kind) = kind(field) else { continue };
        let ident = field.ident.as_ref().expect("named");
        let label = ident.to_string();
        let field_vis = &field.vis;
        let docs = field.attrs.iter().filter(|a| a.path().is_ident("doc"));
        cloned.push(quote! { #ident: ::core::clone::Clone::clone(&self.#ident) });
        match kind {
            Kind::Storage(element) => {
                bounds.push(quote! { #prelude::StorageElement<#element>: #prelude::CubeElement });
                declarations.push(quote! { #(#docs)* #field_vis #ident: usize });
                sizes.push(quote! {
                    visit(#prelude::BufferSize::of::<#element>(#label, shape.#ident));
                });
                carved.push(quote! {
                    #ident: carver.carve::<#element>(
                        #prelude::BufferSize::of::<#element>(#label, shape.#ident),
                    )
                });
            }
            Kind::Grid(element) => {
                bounds.push(quote! { #prelude::StorageElement<#element>: #prelude::CubeElement });
                declarations.push(quote! { #(#docs)* #field_vis #ident: #prelude::GridShape });
                sizes.push(quote! {
                    visit(#prelude::BufferSize::of::<#element>(
                        #label,
                        shape.#ident.rows * shape.#ident.width as usize,
                    ));
                });
                carved.push(quote! {
                    #ident: #prelude::ResidentGrid {
                        values: carver.carve::<#element>(#prelude::BufferSize::of::<#element>(
                            #label,
                            shape.#ident.rows * shape.#ident.width as usize,
                        )),
                        width: shape.#ident.width,
                    }
                });
            }
            Kind::Nested(ty) => {
                let bound = quote! {
                    <#ty as #prelude::Residency>::Resident: #prelude::Shaped + Clone
                };
                bounds.push(quote! { #ty: #prelude::Residency });
                bounds.push(bound.clone());
                shape_bounds.push(quote! { #ty: #prelude::Residency });
                shape_bounds.push(bound);
                named.push(quote! { #ty });
                let nested = quote! {
                    <<#ty as #prelude::Residency>::Resident as #prelude::Shaped>
                };
                declarations.push(quote! { #(#docs)* #field_vis #ident: #nested::Shape });
                sizes.push(quote! { #nested::sizes(&shape.#ident, visit); });
                carved.push(quote! { #ident: #nested::carve(&shape.#ident, carver) });
            }
            Kind::Scalar(ty) => {
                shape_bounds.push(quote! { #ty: Clone });
                named.push(quote! { #ty });
                declarations.push(quote! { #(#docs)* #field_vis #ident: #ty });
                carved.push(quote! { #ident: shape.#ident.clone() });
            }
            Kind::With(..) => return Ok(TokenStream::new()),
        }
    }
    // The parameters the shape's own fields name, with their bounds.
    let named: TokenStream = named.into_iter().collect();
    let used: Vec<&syn::GenericParam> = input
        .generics
        .params
        .iter()
        .filter(|param| match param {
            syn::GenericParam::Type(ty) => mentions(named.clone(), &ty.ident),
            syn::GenericParam::Const(constant) => mentions(named.clone(), &constant.ident),
            syn::GenericParam::Lifetime(_) => true,
        })
        .collect();
    let used_idents: Vec<TokenStream> = used
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(ty) => {
                let ident = &ty.ident;
                quote! { #ident }
            }
            syn::GenericParam::Const(constant) => {
                let ident = &constant.ident;
                quote! { #ident }
            }
            syn::GenericParam::Lifetime(lifetime) => {
                let lifetime = &lifetime.lifetime;
                quote! { #lifetime }
            }
        })
        .collect();
    let declared: Vec<TokenStream> = used
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(ty) => {
                let (ident, bounds) = (&ty.ident, &ty.bounds);
                if bounds.is_empty() {
                    quote! { #ident }
                } else {
                    quote! { #ident: #bounds }
                }
            }
            other => quote! { #other },
        })
        .collect();
    let unused: Vec<&syn::Ident> = input
        .generics
        .type_params()
        .map(|ty| &ty.ident)
        .filter(|ident| !mentions(named.clone(), ident))
        .collect();
    let predicates: Vec<_> = where_clause
        .map(|clause| {
            clause
                .predicates
                .iter()
                .filter(|predicate| {
                    let tokens = quote! { #predicate };
                    !unused.iter().any(|ident| mentions(tokens.clone(), ident))
                })
                .collect()
        })
        .unwrap_or_default();
    let shape_type = quote! { #shape <#(#used_idents),*> };
    let shape_where = quote! { where #(#predicates,)* #(#shape_bounds,)* };
    let all_predicates = where_clause
        .map(|clause| clause.predicates.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    let where_all = quote! { where #(#all_predicates,)* #(#bounds,)* };
    Ok(quote! {
        /// The buffer sizes and scalars of a [`#resident`], before it exists.
        #vis struct #shape <#(#declared),*> #shape_where {
            #(#declarations,)*
        }

        impl <#(#declared),*> ::core::fmt::Debug for #shape_type #shape_where {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.debug_struct(stringify!(#shape)).finish_non_exhaustive()
            }
        }

        impl <#(#declared),*> ::core::clone::Clone for #shape_type #shape_where {
            fn clone(&self) -> Self {
                Self {
                    #(#cloned,)*
                }
            }
        }

        impl #impl_generics #prelude::Shaped for #resident #type_generics #where_all {
            type Shape = #shape_type;

            fn sizes(shape: &Self::Shape, visit: &mut dyn FnMut(#prelude::BufferSize)) {
                #(#sizes)*
            }

            fn carve(shape: &Self::Shape, carver: &mut impl #prelude::Carve) -> Self {
                Self {
                    #(#carved,)*
                }
            }
        }
    })
}
