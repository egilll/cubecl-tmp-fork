use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
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
    let options = Options::parse(input)?;
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
    let transparent = prelude_type("Transparent");
    let (_, type_generics, _) = parsed.generics.split_for_impl();
    let mut generics = parsed.generics.clone();
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#repr: #primitive));
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#name #type_generics: #cube<ExpandType = #expanded #type_generics> + 'static));
    let (impl_generics, ty_generics, clause) = generics.split_for_impl();
    let with_repr = with_repr(name, repr, &parsed.generics, &generics);
    let forwarded = options.forward(&Forwarded {
        name,
        expanded,
        member: &member,
        repr,
        generics: &generics,
    });
    Ok(quote! {
        #forwarded
        #with_repr

        impl #impl_generics #device_repr for #name #ty_generics #clause {
            type Repr = #repr;
            // SAFETY: `#[repr(transparent)]` makes `Repr` the only field
            // with a size, and `DeviceRepr` requires every `Repr` be valid.
            const TRANSPARENT: ::core::option::Option<#transparent<Self>> =
                ::core::option::Option::Some(unsafe { #transparent::new() });

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

/// `WithRepr` for a wrapper generic over its representation, rebranding by
/// substituting that parameter.
fn with_repr(
    name: &syn::Ident,
    repr: &syn::Type,
    declared: &syn::Generics,
    generics: &syn::Generics,
) -> TokenStream {
    let syn::Type::Path(path) = repr else {
        return TokenStream::new();
    };
    let Some(param) = path.path.get_ident() else {
        return TokenStream::new();
    };
    if !declared.type_params().any(|p| &p.ident == param) {
        return TokenStream::new();
    }
    let prelude = crate::paths::prelude_path();
    let args = declared.params.iter().map(|p| match p {
        syn::GenericParam::Type(t) if &t.ident == param => quote![__Repr],
        syn::GenericParam::Type(t) => t.ident.to_token_stream(),
        syn::GenericParam::Lifetime(l) => l.lifetime.to_token_stream(),
        syn::GenericParam::Const(c) => c.ident.to_token_stream(),
    });
    let rebranded = quote![#name<#(#args),*>];
    let mut generics = generics.clone();
    generics.params.push(parse_quote!(__Repr: #prelude::CubePrimitive));
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#rebranded: #prelude::DeviceRepr<Repr = __Repr>));
    let (impl_generics, _, clause) = generics.split_for_impl();
    let (_, ty_generics, _) = declared.split_for_impl();
    quote! {
        impl #impl_generics #prelude::WithRepr<__Repr> for #name #ty_generics #clause {
            type Output = #rebranded;
        }
    }
}

/// Same-brand traits a `#[device_repr(..)]` attribute forwards to the native
/// representation, without bounds on phantom markers. Arithmetic stays explicit.
#[derive(Default)]
struct Options {
    copy: bool,
    eq: bool,
    key: bool,
    ord: bool,
}

struct Forwarded<'a> {
    name: &'a syn::Ident,
    expanded: &'a syn::Ident,
    member: &'a syn::Member,
    repr: &'a syn::Type,
    generics: &'a syn::Generics,
}

impl Options {
    fn parse(input: &DeriveInput) -> syn::Result<Self> {
        let mut options = Self::default();
        for attr in input.attrs.iter().filter(|attr| attr.path().is_ident("device_repr")) {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("copy") {
                    options.copy = true;
                } else if meta.path.is_ident("eq") {
                    options.eq = true;
                } else if meta.path.is_ident("key") {
                    options.key = true;
                } else if meta.path.is_ident("ord") {
                    options.ord = true;
                } else {
                    return Err(meta.error("expected `copy`, `eq`, `key` or `ord`"));
                }
                Ok(())
            })?;
        }
        options.eq |= options.ord;
        Ok(options)
    }

    fn forward(&self, value: &Forwarded) -> TokenStream {
        let Forwarded {
            name,
            expanded,
            member,
            repr,
            generics,
        } = value;
        let prelude = crate::paths::prelude_path();
        let bounded = |predicates: &[syn::WherePredicate]| {
            let mut generics = (*generics).clone();
            generics.make_where_clause().predicates.extend(predicates.iter().cloned());
            generics
        };
        let mut tokens = TokenStream::new();
        if self.copy {
            let (impl_generics, ty_generics, clause) = generics.split_for_impl();
            let expand_generics = bounded(&[
                parse_quote!(#repr: ::core::marker::Copy),
                parse_quote!(<#repr as #prelude::CubeType>::ExpandType: ::core::marker::Copy),
            ]);
            let expand_clause = &expand_generics.where_clause;
            tokens.extend(quote! {
                impl #impl_generics ::core::clone::Clone for #name #ty_generics #clause {
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl #impl_generics ::core::marker::Copy for #name #ty_generics #clause {}

                impl #impl_generics ::core::clone::Clone for #expanded #ty_generics #expand_clause {
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl #impl_generics ::core::marker::Copy for #expanded #ty_generics #expand_clause {}
            });
        }
        if self.eq {
            let generics = bounded(&[
                parse_quote!(#repr: ::core::cmp::PartialEq),
                parse_quote!(<#repr as #prelude::CubeType>::ExpandType: #prelude::PartialEqExpand),
            ]);
            let (impl_generics, ty_generics, clause) = generics.split_for_impl();
            let methods = ["eq", "ne"].map(|op| {
                let method = format_ident!("__expand_{op}_method");
                quote! {
                    fn #method(&self, scope: &#prelude::Scope, rhs: &Self) -> #prelude::NativeExpand<bool> {
                        #prelude::PartialEqExpand::#method(&self.#member, scope, &rhs.#member)
                    }
                }
            });
            tokens.extend(quote! {
                impl #impl_generics ::core::cmp::PartialEq for #name #ty_generics #clause {
                    fn eq(&self, other: &Self) -> bool {
                        self.#member == other.#member
                    }
                }

                impl #impl_generics #prelude::PartialEqExpand for #expanded #ty_generics #clause {
                    #(#methods)*
                }
            });
        }
        if self.ord {
            let generics = bounded(&[
                parse_quote!(#repr: #prelude::CubePartialOrd),
                parse_quote!(<#repr as #prelude::CubeType>::ExpandType: #prelude::PartialOrdExpand),
            ]);
            let (impl_generics, ty_generics, clause) = generics.split_for_impl();
            let ordering = quote![::core::cmp::Ordering];
            let methods = ["lt", "le", "gt", "ge"].map(|op| {
                let method = format_ident!("__expand_{op}_method");
                quote! {
                    fn #method(&self, scope: &#prelude::Scope, rhs: &Self) -> #prelude::NativeExpand<bool> {
                        #prelude::PartialOrdExpand::#method(&self.#member, scope, &rhs.#member)
                    }
                }
            });
            tokens.extend(quote! {
                impl #impl_generics ::core::cmp::PartialOrd for #name #ty_generics #clause {
                    fn partial_cmp(&self, other: &Self) -> ::core::option::Option<#ordering> {
                        self.#member.partial_cmp(&other.#member)
                    }
                }

                impl #impl_generics #prelude::PartialOrdExpand for #expanded #ty_generics #clause {
                    fn __expand_partial_cmp_method(
                        &self,
                        scope: &#prelude::Scope,
                        rhs: &Self,
                    ) -> #prelude::OptionExpand<#ordering> {
                        #prelude::PartialOrdExpand::__expand_partial_cmp_method(
                            &self.#member,
                            scope,
                            &rhs.#member,
                        )
                    }
                    #(#methods)*
                }

                impl #impl_generics #prelude::Ordered for #name #ty_generics #clause {}

                impl #impl_generics #prelude::OrderedExpand for #expanded #ty_generics #clause {
                    type Value = #name #ty_generics;
                }
            });
        }
        if self.key {
            let generics = bounded(&[parse_quote!(#repr: #prelude::StorageKey)]);
            let (impl_generics, ty_generics, clause) = generics.split_for_impl();
            tokens.extend(quote! {
                impl #impl_generics #prelude::StorageKey for #name #ty_generics #clause {
                    fn __expand_position(
                        scope: &#prelude::Scope,
                        key: <Self as #prelude::CubeType>::ExpandType,
                    ) -> #prelude::NativeExpand<usize> {
                        <#repr as #prelude::StorageKey>::__expand_position(scope, key.#member)
                    }
                }
            });
        }
        tokens
    }
}
