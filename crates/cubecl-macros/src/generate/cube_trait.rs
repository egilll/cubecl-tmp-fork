use crate::{
    parse::cube_trait::{CubeTrait, CubeTraitImpl, CubeTraitImplItem, CubeTraitItem},
    paths::{frontend_path, prelude_type},
};
use proc_macro2::TokenStream;
use quote::quote;
use quote::{ToTokens, format_ident};
use syn::{
    ConstParam, GenericArgument, Token, Type, TypeParam, TypePath, parse_quote, spanned::Spanned,
};

impl ToTokens for CubeTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let original_body = &self.original_trait.items;
        let mut colon = self.original_trait.colon_token;
        let mut base_traits = self.original_trait.supertraits.clone();
        let mut algebra_generics = self.original_trait.generics.clone();
        crate::parse::algebra::add_expand_bounds(&mut algebra_generics);
        let where_clause = algebra_generics.where_clause.clone();
        let cube_type = prelude_type("CubeType");
        for bound in &self.original_trait.supertraits {
            for expanded in
                crate::parse::algebra::expanded_bounds(bound, &parse_quote!(Self), false)
            {
                base_traits.push(parse_quote!(#cube_type<ExpandType: #expanded>));
                colon = Some(Token![:](tokens.span()));
            }
        }
        let attrs = &self.attrs;
        let vis = &self.vis;
        let unsafety = &self.unsafety;
        let name = &self.name;
        let generics = &self.generics;
        let assoc_fns = self
            .items
            .iter()
            .filter_map(CubeTraitItem::func)
            .map(|sig| {
                if let Some(default) = self
                    .defaults
                    .iter()
                    .find(|function| function.sig.name == sig.name)
                {
                    default.clone().to_tokens_mut()
                } else {
                    quote![#sig;]
                }
            });
        let assoc_methods = self
            .items
            .iter()
            .filter_map(CubeTraitItem::associated_method);

        let has_expand = self.default_methods
            || self
                .items
                .iter()
                .any(|it| matches!(it, CubeTraitItem::Method(_)));

        if has_expand {
            let cube_type = prelude_type("CubeType");
            let expand_name = format_ident!("{}Expand", self.name);
            let associated_bounds = self
                .items
                .iter()
                .filter_map(CubeTraitItem::other_ident)
                .map(|it| parse_quote![#it = Self::#it])
                .collect::<Vec<GenericArgument>>();

            let mut generic_args = quote![];

            if !generics.params.is_empty() || !associated_bounds.is_empty() {
                let generics = generics.params.iter().map(|it| match it {
                    syn::GenericParam::Lifetime(lifetime_param) => {
                        GenericArgument::Lifetime(lifetime_param.lifetime.clone())
                    }
                    syn::GenericParam::Type(TypeParam { ident, .. })
                    | syn::GenericParam::Const(ConstParam { ident, .. }) => {
                        GenericArgument::Type(parse_quote!(#ident))
                    }
                });
                let args = generics.chain(associated_bounds);
                generic_args = quote![<#(#args),*>];
            }

            base_traits.push(parse_quote!(#cube_type<ExpandType: #expand_name #generic_args>));
            colon = Some(Token![:](tokens.span()));
        }

        let out = quote! {
            #(#attrs)*
            #[allow(clippy::too_many_arguments)]
            #vis #unsafety trait #name #generics #colon #base_traits #where_clause {
                #(#original_body)*

                #(
                    #[allow(clippy::too_many_arguments, unused_braces)]
                    #assoc_fns
                )*

                #(
                    #[allow(clippy::too_many_arguments)]
                    #assoc_methods
                )*
            }
        };
        tokens.extend(out);

        if has_expand {
            tokens.extend(self.generate_expand());
        }
    }
}

impl CubeTrait {
    fn generate_expand(&self) -> TokenStream {
        let attrs = &self.attrs;
        let vis = &self.vis;
        let unsafety = &self.unsafety;
        let name = format_ident!("{}Expand", self.name);
        let generics = crate::parse::algebra::expanded_generics(&self.generics);
        let where_clause = &generics.where_clause;
        let others = self.items.iter().filter_map(CubeTraitItem::other);
        let methods = self
            .items
            .iter()
            .filter_map(CubeTraitItem::method)
            .cloned()
            .map(|mut method| {
                method.plain_self();
                if let Some(default) = self
                    .defaults
                    .iter()
                    .find(|function| function.sig.name == method.name)
                {
                    default.clone().to_tokens_mut()
                } else {
                    quote![#method;]
                }
            });
        let mut supertraits = self.expand_supertraits.clone();
        if !self.defaults.is_empty() {
            let into_expand = prelude_type("IntoExpand");
            supertraits.push(parse_quote!(Sized));
            supertraits.push(parse_quote!(#into_expand<Expand = Self>));
        }
        for bound in &self.original_trait.supertraits {
            for expanded in crate::parse::algebra::expanded_bounds(bound, &parse_quote!(Self), true)
            {
                supertraits.push(expanded);
            }
        }
        let colon = (!supertraits.is_empty()).then(|| quote![:]);

        quote! {
            #(#attrs)*
            #[allow(clippy::too_many_arguments)]
            #vis #unsafety trait #name #generics #colon #supertraits #where_clause {
                #(#others)*

                #(
                    #[allow(clippy::too_many_arguments)]
                    #methods
                )*
            }
        }
    }
}

impl CubeTraitImpl {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        if let Some(operator) = self.operator {
            return self.operator_tokens(operator);
        }
        let has_expand = self.default_methods
            || self
                .items
                .iter()
                .any(|it| matches!(it, CubeTraitImplItem::Method(_)));

        let expand = if has_expand {
            self.generate_expand()
        } else {
            quote![]
        };

        let unsafety = &self.unsafety;
        let items = &self.original_items;
        let fns = &self
            .items
            .iter_mut()
            .filter_map(CubeTraitImplItem::func)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let struct_name = &self.struct_name;
        let trait_name = &self.trait_name;
        let (generics, _, impl_where) = self.generics.split_for_impl();

        quote! {
            #unsafety impl #generics #trait_name for #struct_name #impl_where {
                #(#items)*
                #(
                    #[allow(unused, clippy::all)]
                    #fns
                )*
            }
            #expand
        }
    }
}

impl CubeTraitImpl {
    /// A `core::ops` operator: the impl as written, and its kernel
    /// implementation, the frontend's `Cube{operator}` for the same host type
    /// and operands.
    fn operator_tokens(&mut self, operator: &str) -> TokenStream {
        let unsafety = &self.unsafety;
        let items = &self.original_items;
        let struct_name = &self.struct_name;
        let trait_name = &self.trait_name;
        let (generics, _, impl_where) = self.generics.split_for_impl();

        let mut cube_trait = frontend_path();
        let mut segment = trait_name.segments.last().unwrap().clone();
        segment.ident = format_ident!("Cube{operator}", span = segment.ident.span());
        cube_trait.segments.push(segment);

        let fns = self
            .items
            .iter_mut()
            .filter_map(CubeTraitImplItem::func)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();

        quote! {
            #unsafety impl #generics #trait_name for #struct_name #impl_where {
                #(#items)*
            }

            impl #generics #cube_trait for #struct_name #impl_where {
                #(
                    #[allow(unused, clippy::all)]
                    #fns
                )*
            }
        }
    }

    fn generate_expand(&mut self) -> TokenStream {
        let others = self
            .items
            .iter()
            .filter_map(CubeTraitImplItem::other)
            .cloned()
            .collect::<Vec<_>>();
        let methods = self
            .items
            .iter_mut()
            .filter_map(CubeTraitImplItem::method)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let unsafety = &self.unsafety;

        let struct_name = match expand_type_of(&self.struct_name) {
            Ok(name) => name,
            Err(err) => return err.into_compile_error(),
        };

        let mut trait_name = self.trait_name.clone();
        let trait_ident = trait_name.segments.last_mut().unwrap();
        trait_ident.ident = format_ident!("{}Expand", trait_ident.ident);

        let (generics, _, impl_where) = self.generics.split_for_impl();

        quote! {
            #unsafety impl #generics #trait_name for #struct_name #impl_where {
                #(#others)*
                #(
                    #[allow(unused, clippy::all)]
                    #methods
                )*
            }
        }
    }
}

/// The expand type of an impl's self type `ty`: primitives and vectors expand
/// to `NativeExpand`, other structs to their derived `{Name}Expand`.
fn expand_type_of(ty: &Type) -> syn::Result<TokenStream> {
    const NATIVE: &[&str] = &[
        "f16", "bf16", "f32", "f64", "bool", "u8", "u16", "u32", "u64", "usize", "i8", "i16",
        "i32", "i64", "isize", "Vector",
    ];
    let mut path = path_of_type(ty)?;
    let name = path.path.segments.last().unwrap().ident.clone();
    if NATIVE.contains(&name.to_string().as_str()) {
        let native = prelude_type("NativeExpand");
        return Ok(quote![#native<#path>]);
    }
    path.path.segments.last_mut().unwrap().ident = format_ident!("{name}Expand");
    Ok(quote![#path])
}

fn path_of_type(ty: &Type) -> Result<TypePath, syn::Error> {
    match ty {
        Type::Array(type_array) => path_of_type(&type_array.elem),
        Type::Group(type_group) => path_of_type(&type_group.elem),
        Type::Paren(type_paren) => path_of_type(&type_paren.elem),
        Type::Path(type_path) => Ok(type_path.clone()),
        Type::Ptr(type_ptr) => path_of_type(&type_ptr.elem),
        Type::Reference(type_reference) => path_of_type(&type_reference.elem),
        Type::Slice(type_slice) => path_of_type(&type_slice.elem),
        other => Err(syn::Error::new(
            ty.span(),
            format!("Tried to get path of unsupported type: {other:?}"),
        )),
    }
}
