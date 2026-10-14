use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::Type;

use crate::{
    parse::{kernel::expand_kernel_ty, signature::*},
    paths::{frontend_type, prelude_type},
};

impl ToTokens for KernelSignature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let scope = prelude_type("Scope");

        let name = &self.name;
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let scope_lifetime = &self.scope_lifetime;

        let return_type = match &self.returns {
            KernelReturns::ExpandType(ty) if primitive_generic(ty, generics) => {
                let native = frontend_type("NativeExpand");
                quote![#native<#ty>]
            }
            KernelReturns::ExpandType(ty) => {
                let normalized_ty = expand_kernel_ty(ty.clone(), false)
                    .unwrap_or_else(|err| Type::Verbatim(err.into_compile_error()));
                quote![#normalized_ty]
            }
            KernelReturns::Plain(ty) => quote![#ty],
        };
        let out = if let Some(receiver) = &self.receiver_arg {
            let args = self.parameters.iter().skip(1);

            quote! {
                fn #name #generics(
                    #receiver,
                    scope: &#scope_lifetime #scope,
                    #(#args),*
                ) -> #return_type #where_clause
            }
        } else {
            let args = &self.parameters;
            quote! {
                fn #name #generics(
                    scope: &#scope_lifetime #scope,
                    #(#args),*
                ) -> #return_type #where_clause
            }
        };

        tokens.extend(out);
    }
}

impl ToTokens for KernelParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let mut_ = &self.mutability;
        if self.expand_arg {
            let expand_arg = frontend_type("ExpandArg");
            let ty = &self.ty;
            tokens.extend(quote![#mut_ #name: #expand_arg<#ty>]);
        } else {
            let ty = &self.normalized_ty;
            tokens.extend(quote![#mut_ #name: #ty]);
        }
    }
}

/// Whether `ty` is a type parameter bounded as a primitive, whose expand type
/// is `NativeExpand<T>`. A result typed so names `T` itself rather than a
/// projection through it, so a caller's untyped literal argument, as in
/// `twice(2.0) + x`, still takes its type from what the result meets.
fn primitive_generic(ty: &Type, generics: &syn::Generics) -> bool {
    const PRIMITIVES: [&str; 5] = ["CubePrimitive", "Scalar", "Numeric", "Float", "Int"];
    let Type::Path(path) = ty else {
        return false;
    };
    let Some(ident) = path
        .qself
        .is_none()
        .then(|| path.path.get_ident())
        .flatten()
    else {
        return false;
    };
    let primitive = |bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>| {
        bounds.iter().any(|bound| match bound {
            syn::TypeParamBound::Trait(bound) => bound
                .path
                .segments
                .last()
                .is_some_and(|last| PRIMITIVES.iter().any(|name| last.ident == name)),
            _ => false,
        })
    };
    let declared = generics
        .type_params()
        .any(|param| &param.ident == ident && primitive(&param.bounds));
    let required = generics
        .where_clause
        .iter()
        .flat_map(|clause| &clause.predicates)
        .any(|predicate| match predicate {
            syn::WherePredicate::Type(predicate) => {
                matches!(&predicate.bounded_ty, Type::Path(bounded) if bounded.path.is_ident(ident))
                    && primitive(&predicate.bounds)
            }
            _ => false,
        });
    generics.type_params().any(|param| &param.ident == ident) && (declared || required)
}
