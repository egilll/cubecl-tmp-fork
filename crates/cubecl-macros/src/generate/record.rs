//! `#[derive(Record)]`: a struct stored as the words of its fields, each a
//! record itself, in declaration order.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DataStruct, DeriveInput, Fields};

use crate::paths::prelude_path;

pub fn generate(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let Data::Struct(DataStruct { fields: Fields::Named(fields), .. }) = &input.data else {
        return Err(syn::Error::new_spanned(input, "a record is a struct with named fields"));
    };
    let prelude = prelude_path();
    let mut generics = input.generics.clone();
    let field_types: Vec<_> = fields.named.iter().map(|field| field.ty.clone()).collect();
    {
        let clause = generics.make_where_clause();
        for ty in &field_types {
            clause.predicates.push(syn::parse_quote!(#ty: #prelude::Record));
        }
    }
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let literal = if input.generics.type_params().next().is_some() {
        let params = input.generics.type_params().map(|p| &p.ident);
        quote!(#name::<#(#params),*>)
    } else {
        quote!(#name)
    };

    let mut words = vec![quote!(0usize)];
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    let mut loads = Vec::new();
    let mut stores = Vec::new();
    let mut offsets = Vec::new();
    for (index, field) in fields.named.iter().enumerate() {
        let ident = field.ident.as_ref().expect("named field");
        let ty = &field.ty;
        let offset = format_ident!("__at_{}", index);
        let at = quote!(#(#words)+*);
        offsets.push(quote! { let #offset = at + comptime![#at]; });
        writes.push(quote! { #prelude::Words::write(&self.#ident, &mut words[#at..]); });
        reads.push(quote! { #ident: <#ty as #prelude::Words>::read(&words[#at..]) });
        loads.push(quote! { #ident: <#ty as #prelude::Record>::read_at(words, #offset) });
        stores.push(quote! { <#ty as #prelude::Record>::write_at(value.#ident, words, #offset); });
        words.push(quote!(<#ty as #prelude::Words>::WORDS));
    }

    Ok(quote! {
        impl #impl_generics #prelude::Words for #name #type_generics #where_clause {
            const WORDS: usize = #(#words)+*;

            fn write(&self, words: &mut [u32]) {
                #(#writes)*
            }

            fn read(words: &[u32]) -> Self {
                #literal { #(#reads),* }
            }
        }

        #[#prelude::cube]
        impl #impl_generics #prelude::Record for #name #type_generics #where_clause {
            fn read_at(words: &[u32], at: usize) -> #name #type_generics {
                #(#offsets)*
                #literal { #(#loads),* }
            }

            fn write_at(value: #name #type_generics, words: &mut [u32], at: usize) {
                #(#offsets)*
                #(#stores)*
            }
        }
    })
}
