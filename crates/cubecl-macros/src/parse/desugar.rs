use std::mem::take;

use quote::{format_ident, quote, quote_spanned};
use syn::{
    Index, Local, LocalInit, Pat, PatIdent, PatSlice, PatStruct, PatTuple, PatTupleStruct, Stmt,
    parse_quote, parse_quote_spanned,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
};

pub struct Desugar;
impl VisitMut for Desugar {
    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        let mut next_id = 0;
        let stmts = desugar_pats(take(&mut i.stmts), &mut next_id);

        i.stmts = stmts;
        visit_mut::visit_block_mut(self, i)
    }
}

/// Desugars `?` in the bodies of every function of an item.
pub struct DesugarTry;
impl VisitMut for DesugarTry {
    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        desugar_try(&mut i.block, &i.sig.output);
    }

    fn visit_impl_item_fn_mut(&mut self, i: &mut syn::ImplItemFn) {
        desugar_try(&mut i.block, &i.sig.output);
    }

    fn visit_trait_item_fn_mut(&mut self, i: &mut syn::TraitItemFn) {
        if let Some(block) = &mut i.default {
            desugar_try(block, &i.sig.output);
        }
    }
}

/// `iter.fold(init, |acc, item| body)` becomes
/// `{ let mut acc = init; for item in iter { acc = body; } acc }`, so a fold
/// runs over anything a kernel loop runs over.
pub struct DesugarFold;
impl VisitMut for DesugarFold {
    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        visit_mut::visit_expr_mut(self, i);
        let syn::Expr::MethodCall(call) = i else {
            return;
        };
        if call.method != "fold" || call.args.len() != 2 {
            return;
        }
        let syn::Expr::Closure(closure) = &call.args[1] else {
            return;
        };
        if closure.inputs.len() != 2 {
            return;
        }
        let (accumulator, item) = (&closure.inputs[0], &closure.inputs[1]);
        let Pat::Ident(PatIdent { ident, .. }) = accumulator else {
            return;
        };
        let (receiver, init, body) = (&call.receiver, &call.args[0], &closure.body);
        *i = parse_quote_spanned! {call.span()=> {
            let mut #ident = #init;
            for #item in #receiver {
                #ident = #body;
            }
            #ident
        }};
    }
}

/// Runtime `?` in the body's statements: `let p = e?; rest` becomes
/// `let t = e.split(); if t.0 { let p = t.1; rest } else { absent }`,
/// so the remaining body runs only on a present value. Functions returning
/// a host `Result` keep comptime `?`.
fn desugar_try(block: &mut syn::Block, returns: &syn::ReturnType) {
    let syn::ReturnType::Type(_, returns) = returns else {
        return;
    };
    let host_result = matches!(&**returns, syn::Type::Path(path)
        if path.path.segments.last().is_some_and(|segment| segment.ident == "Result"));
    if !host_result {
        let mut next_id = 0;
        block.stmts = desugar_try_stmts(take(&mut block.stmts), returns, &mut next_id);
    }
}

fn desugar_try_stmts(stmts: Vec<Stmt>, returns: &syn::Type, next_id: &mut usize) -> Vec<Stmt> {
    let mut output = Vec::new();
    let mut stmts = stmts.into_iter();
    while let Some(stmt) = stmts.next() {
        let (pat, inner) = match stmt {
            Stmt::Local(Local {
                pat,
                init:
                    Some(LocalInit {
                        expr,
                        diverge: None,
                        ..
                    }),
                ..
            }) if matches!(*expr, syn::Expr::Try(_)) => {
                let syn::Expr::Try(inner) = *expr else {
                    unreachable!()
                };
                (Some(pat), inner)
            }
            Stmt::Expr(syn::Expr::Try(inner), Some(_)) => (None, inner),
            stmt => {
                output.push(stmt);
                continue;
            }
        };
        let id = format_ident!("__try_{}", *next_id);
        *next_id += 1;
        let rest = desugar_try_stmts(stmts.collect(), returns, next_id);
        let value = inner.expr;
        let fallible = crate::paths::prelude_type("Fallible");
        let bind = pat.map(|pat| quote_spanned![pat.span()=> let #pat = #id.1;]);
        output.extend::<Vec<Stmt>>(parse_quote! {
            let #id = (#value).split();
            if #id.0 {
                #bind
                #(#rest)*
            } else {
                <#returns as #fallible>::absent()
            }
        });
        break;
    }
    output
}

fn desugar_pats(stmts: Vec<Stmt>, next_id: &mut usize) -> Vec<Stmt> {
    let mut output = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::Local(Local {
                pat: Pat::Struct(pat),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_struct_destructure(pat, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat:
                    Pat::Tuple(PatTuple { elems, .. }) | Pat::TupleStruct(PatTupleStruct { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_tuple_destructure(elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat: Pat::Slice(PatSlice { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let elems = elems.into_iter().collect::<Vec<_>>();
                let stmts = desugar_slice_destructure(&elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            stmt => output.push(stmt),
        }
    }
    output
}

fn desugar_struct_destructure(pat: PatStruct, init: LocalInit, id: usize) -> Vec<Stmt> {
    let init_ident = format_ident!("__struct_destructure_init_{id}");
    let fields = pat.fields.into_iter().map(|field| {
        let attrs = field.attrs;
        let pat = field.pat;
        let member = field.member;
        quote_spanned! {pat.span()=>
            #(#attrs)* let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_tuple_destructure(
    fields: impl IntoIterator<Item = Pat>,
    init: LocalInit,
    id: usize,
) -> Vec<Stmt> {
    // A literal tuple binds each pattern to its own element, so a mutable
    // binding of a constant element becomes a runtime variable as
    // `let mut x = 0.0;` does.
    let fields = fields.into_iter().collect::<Vec<_>>();
    if let syn::Expr::Tuple(tuple) = &*init.expr
        && tuple.elems.len() == fields.len()
    {
        let bindings = fields.iter().zip(&tuple.elems).map(|(pat, elem)| {
            quote_spanned! {pat.span()=>
                let #pat = #elem;
            }
        });
        return parse_quote! {
            #(#bindings)*
        };
    }
    desugar_tuple_through_init(fields, init, id)
}

fn desugar_tuple_through_init(
    fields: impl IntoIterator<Item = Pat>,
    init: LocalInit,
    id: usize,
) -> Vec<Stmt> {
    let init_ident = format_ident!("__tuple_destructure_init_{id}");
    let fields = fields.into_iter().enumerate().map(|(i, pat)| {
        let member = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_slice_destructure(fields: &[Pat], init: LocalInit, id: usize) -> Vec<Stmt> {
    if let Some(field) = fields.iter().find(|field| {
        matches!(
            field,
            Pat::Ident(PatIdent {
                subpat: Some(_),
                ..
            })
        )
    }) {
        let err = syn::Error::new(field.span(), "@ patterns are not currently supported")
            .to_compile_error();
        return vec![parse_quote!(#err;)];
    }

    // Slice patterns can't have more than one rest pattern, so it can always be cleanly separated
    // into before rest (which start at 0) and after rest (which start from len - n_after_rest).
    let rest_pos = fields
        .iter()
        .position(|field| matches!(field, Pat::Rest(_)))
        .unwrap_or(fields.len());
    let from_start = &fields[..rest_pos];
    let from_end = &fields[(rest_pos + 1).min(fields.len())..];
    let init_ident = format_ident!("__slice_destructure_init_{id}");
    let len_ident = format_ident!("__slice_destructure_len_{id}");

    let from_start_fields = from_start.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#offset];
        }
    });

    let init = init.expr;

    let len_expr = if from_end.is_empty() {
        quote![]
    } else {
        quote_spanned![init.span()=> let #len_ident = #init_ident.len();]
    };
    let from_end_fields = from_end.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(from_end.len() - i);
        // This requires a bit of a hack on `sub::expand` to make it work for `Sequence`
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#len_ident - #offset];
        }
    });

    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #len_expr
        #(#from_start_fields)*
        #(#from_end_fields)*
    }
}

#[cfg(test)]
mod tests {
    use super::Desugar;
    use crate::{expression::Block, scope::Context};
    use syn::{Ident, parse_quote, visit_mut::VisitMut};

    #[test]
    fn nested_tuple_patterns_are_fully_desugared() {
        let mut block: syn::Block = parse_quote!({
            let (a, b, (c, d, (e, f))) = tuple;
        });

        Desugar.visit_block_mut(&mut block);

        let mut context = Context::new(parse_quote!(()), false, false);
        let result = Block::from_block(block, &mut context);

        assert!(result.is_ok(), "{result:?}");

        let bindings: [Ident; 6] = [
            parse_quote!(a),
            parse_quote!(b),
            parse_quote!(c),
            parse_quote!(d),
            parse_quote!(e),
            parse_quote!(f),
        ];
        for binding in bindings {
            assert!(
                context.variable(&binding).is_some(),
                "missing binding `{binding}`"
            );
        }
    }
}
