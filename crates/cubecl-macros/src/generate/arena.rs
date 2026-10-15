//! `#[derive(Arena)]`: a struct of tables bound as one kernel argument, its
//! tables of one element sharing a buffer (see `cubecl_core::compute::arena`).
//!
//! For `struct S` it implements `LaunchArg` for `S` (with `ArenaArg` as the
//! runtime argument) and generates `SArena`, the host side: the packed
//! `Arena` and an `ArenaColumn` per table, built from an `SData` with an
//! `ArenaSource` per table.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DataStruct, DeriveInput, Fields};

use crate::paths::prelude_path;

pub fn generate(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let Data::Struct(DataStruct { fields: Fields::Named(fields), .. }) = &input.data else {
        return Err(syn::Error::new_spanned(input, "an arena is a struct of named tables"));
    };
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "an arena has no generics"));
    }
    let prelude = prelude_path();
    let vis = &input.vis;
    let expanded = format_ident!("{}Expand", name);
    let arena = format_ident!("{}Arena", name);
    let data = format_ident!("{}Data", name);
    let idents: Vec<_> = fields.named.iter().map(|field| field.ident.clone().expect("named")).collect();
    let types: Vec<_> = fields.named.iter().map(|field| field.ty.clone()).collect();
    let docs: Vec<Vec<_>> = fields
        .named
        .iter()
        .map(|field| field.attrs.iter().filter(|attr| attr.path().is_ident("doc")).cloned().collect())
        .collect();
    let indices = 0..idents.len();
    let doc = format!("The host side of [`{name}`]: its packed buffers and each table's region.");
    let data_doc = format!("What each table of [`{name}`] initially holds.");

    Ok(quote! {
        impl #prelude::LaunchArg for #name {
            type RuntimeArg = #prelude::ArenaArg;
            type CompilationArg = #prelude::ArenaCompilationArg;

            fn register(arg: #prelude::ArenaArg, launcher: &mut #prelude::KernelLauncher) -> Self::CompilationArg {
                arg.register(launcher)
            }

            fn expand(arg: &Self::CompilationArg, builder: &mut #prelude::KernelBuilder) -> #expanded {
                let mut tables = #prelude::ArenaExpansion::new(arg);
                #expanded {
                    #(#idents: tables.table::<#types>(builder),)*
                }
            }
        }

        #[doc = #doc]
        #[derive(Clone)]
        #vis struct #arena {
            pub arena: #prelude::Arena,
            #(
                #(#docs)*
                pub #idents: #prelude::ArenaColumn<<#types as #prelude::ArenaTable>::Host>,
            )*
        }

        #[doc = #data_doc]
        #vis struct #data<'a> {
            #(
                #(#docs)*
                pub #idents: #prelude::ArenaSource<'a, <#types as #prelude::ArenaTable>::Host>,
            )*
        }

        impl #arena {
            /// Packs every table and uploads them.
            pub fn new(client: &#prelude::Client, data: #data<'_>) -> Self {
                let mut builder = #prelude::ArenaBuilder::default();
                #(builder.table::<#types>(data.#idents);)*
                let arena = builder.finish(client);
                Self { #(#idents: arena.column(#indices),)* arena }
            }

            /// The arena as a kernel argument.
            pub fn arg(&self) -> #prelude::ArenaArg {
                self.arena.arg()
            }
        }
    })
}
