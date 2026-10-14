use darling::usage::{GenericsExt, Purpose, UsesTypeParams};
use proc_macro2::TokenStream;
use syn::{
    Generics, Token, Type, WhereClause, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

pub mod asm;
pub mod assign;
pub mod autotune;
pub mod cube_impl;
pub mod cube_trait;
pub mod cube_type;
pub mod expression;
pub mod into_runtime;
pub mod kernel;
pub mod launch;
pub mod representation;
pub mod resident;
pub mod signature;
pub mod statement;

pub trait RuntimeField: Clone + UsesTypeParams {
    fn ty(self) -> Type;
}

impl RuntimeField for syn::Field {
    fn ty(self) -> Type {
        self.ty
    }
}

pub(crate) fn bounded_where_clause<T: RuntimeField>(
    generics: &Generics,
    runtime_fields: impl IntoIterator<Item = T>,
    bound: impl Fn(&Type) -> TokenStream,
) -> Option<WhereClause> {
    let params = generics.declared_type_params();
    let opts = Purpose::BoundImpl.into();

    let fields_params = runtime_fields
        .into_iter()
        .filter(|it| !it.clone().uses_type_params(&opts, &params).is_empty())
        .map(|it| it.ty())
        .collect::<Vec<_>>();

    if fields_params.is_empty() {
        return generics.where_clause.clone();
    }

    let span = generics.span();
    let mut where_clause = generics
        .where_clause
        .clone()
        .unwrap_or_else(|| WhereClause {
            where_token: Token![where](span),
            predicates: Punctuated::new(),
        });

    for field_ty in fields_params {
        let bound = bound(&field_ty);
        where_clause.predicates.push(parse_quote!(#bound));
    }

    Some(where_clause)
}

/// Whether `ty` is a type parameter bounded as a primitive, whose expand type
/// is `NativeExpand<T>`. A result typed so names `T` itself rather than a
/// projection through it, so a caller's untyped literal argument, as in
/// `twice(2.0) + x`, still takes its type from what the result meets.
pub(crate) fn primitive_generic(ty: &Type, generics: &syn::Generics) -> bool {
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
