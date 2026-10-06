use quote::format_ident;
use syn::{
    GenericArgument, Generics, PathArguments, Type, TypeParamBound, WherePredicate, parse_quote,
};

use crate::paths::{frontend_type, prelude_type};

fn operator(path: &syn::Path) -> Option<String> {
    let names: Vec<_> = path
        .segments
        .iter()
        .map(|it| it.ident.to_string())
        .collect();
    let name = names.last()?;
    let recognized = matches!(
        name.as_str(),
        "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Rem"
            | "BitAnd"
            | "BitOr"
            | "BitXor"
            | "Shl"
            | "Shr"
            | "Neg"
            | "Not"
            | "AddAssign"
            | "SubAssign"
            | "MulAssign"
            | "DivAssign"
            | "RemAssign"
            | "PartialEq"
            | "PartialOrd"
    );
    let standard = names.len() == 1
        || (names.len() == 3
            && matches!(names[0].as_str(), "core" | "std")
            && matches!(names[1].as_str(), "ops" | "cmp"));
    (recognized && standard).then(|| name.clone())
}

pub fn expanded_bound(
    bound: &TypeParamBound,
    lhs: &Type,
    expanded_self: bool,
) -> Option<TypeParamBound> {
    let TypeParamBound::Trait(bound) = bound else {
        return None;
    };
    if bound.lifetimes.is_some() {
        return None;
    }
    let name = operator(&bound.path)?;
    let expand = frontend_type(&format_ident!("{name}Expand").to_string());
    let cube = prelude_type("CubeType");
    let expand_ty = |ty: &Type| -> Type {
        if expanded_self && *ty == parse_quote!(Self) {
            parse_quote!(Self)
        } else {
            parse_quote!(<#ty as #cube>::ExpandType)
        }
    };
    let mut rhs = lhs.clone();
    let mut output = None;
    if let PathArguments::AngleBracketed(args) = &bound.path.segments.last()?.arguments {
        for arg in &args.args {
            match arg {
                GenericArgument::Type(ty) => rhs = ty.clone(),
                GenericArgument::AssocType(assoc) if assoc.ident == "Output" => {
                    output = Some(expand_ty(&assoc.ty))
                }
                _ => return None,
            }
        }
    }
    if output.is_none()
        && !expanded_self
        && !name.ends_with("Assign")
        && !matches!(name.as_str(), "Neg" | "Not" | "PartialEq" | "PartialOrd")
    {
        let path = &bound.path;
        output = Some(expand_ty(&parse_quote!(<#lhs as #path>::Output)));
    }
    if matches!(name.as_str(), "Neg" | "Not") {
        Some(parse_quote!(#expand))
    } else {
        let rhs = expand_ty(&rhs);
        match output {
            Some(output) => Some(parse_quote!(#expand<#rhs, Output = #output>)),
            None => Some(parse_quote!(#expand<#rhs>)),
        }
    }
}

pub fn expanded_generics(generics: &Generics) -> Generics {
    let mut expanded = generics.clone();
    if let Some(clause) = &mut expanded.where_clause {
        clause.predicates = clause
            .predicates
            .iter()
            .filter_map(|predicate| {
                if let WherePredicate::Type(predicate) = predicate
                    && predicate.bounded_ty == parse_quote!(Self)
                {
                    let bounds: syn::punctuated::Punctuated<_, syn::Token![+]> = predicate
                        .bounds
                        .iter()
                        .flat_map(|bound| expanded_bounds(bound, &parse_quote!(Self), true))
                        .collect();
                    return (!bounds.is_empty()).then(|| parse_quote!(Self: #bounds));
                }
                Some(predicate.clone())
            })
            .collect();
    }
    add_expand_bounds(&mut expanded);
    expanded
}

pub fn expanded_bounds(
    bound: &TypeParamBound,
    lhs: &Type,
    expanded_self: bool,
) -> Vec<TypeParamBound> {
    let mut expanded: Vec<_> = expanded_bound(bound, lhs, expanded_self)
        .into_iter()
        .collect();
    if let TypeParamBound::Trait(bound) = bound
        && operator(&bound.path).as_deref() == Some("PartialOrd")
    {
        let mut equality = bound.clone();
        equality.path.segments.last_mut().unwrap().ident = format_ident!("PartialEq");
        expanded.extend(expanded_bound(
            &TypeParamBound::Trait(equality),
            lhs,
            expanded_self,
        ));
    }
    expanded
}

pub fn add_expand_bounds(generics: &mut Generics) {
    let mut predicates: Vec<WherePredicate> = Vec::new();
    let cube = prelude_type("CubeType");
    let mut collect =
        |ty: Type, bounds: &syn::punctuated::Punctuated<TypeParamBound, syn::Token![+]>| {
            for bound in bounds {
                for expanded in expanded_bounds(bound, &ty, false) {
                    predicates.push(parse_quote!(#ty: #cube<ExpandType: #expanded>));
                }
            }
        };
    for param in generics.type_params() {
        let ident = &param.ident;
        collect(parse_quote!(#ident), &param.bounds);
    }
    if let Some(clause) = &generics.where_clause {
        for predicate in &clause.predicates {
            if let WherePredicate::Type(predicate) = predicate {
                collect(predicate.bounded_ty.clone(), &predicate.bounds);
            }
        }
    }
    if !predicates.is_empty() {
        generics.make_where_clause().predicates.extend(predicates);
    }
}
