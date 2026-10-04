use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type, Visibility, WhereClause};

use super::generate::reference_impls;
use crate::{
    generate::bounded_where_clause,
    parse::cube_type::{CubeTypeStruct, TypeField},
    paths::{frontend_type, prelude_type},
};

impl CubeTypeStruct {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        if with_launch {
            let launch_ty = self.launch_ty();
            let launch_new = self.launch_new();
            let launch_arg_impl = self.launch_arg_impl();

            quote! {
                #launch_ty
                #launch_new
                #launch_arg_impl
            }
        } else {
            let expand_ty = self.expand_ty();
            let clone_impl = self.clone_expand();
            let cube_type_impl = self.cube_type_impl();
            let expand_type_impl = self.expand_type_impl();
            let call_arg_impl = self.call_arg_impl();

            quote! {
                #expand_ty
                #clone_impl
                #cube_type_impl
                #expand_type_impl
                #call_arg_impl
            }
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let expand_derives = match &self.derive {
            Some(derives) => quote![#[#derives]],
            None => quote![],
        };

        let fields = self.fields.iter().map(TypeField::expand_field);
        let name = &self.name_expand;
        let generics = &self.generics;
        let vis = &self.vis;

        quote! {
            #expand_derives
            #vis struct #name #generics {
                #(#fields),*
            }
        }
    }

    fn clone_expand(&self) -> proc_macro2::TokenStream {
        let clone = prelude_type("ExpandTypeClone");

        let fields = self.fields.iter().map(TypeField::clone_field);
        let name = &self.name_expand;
        let generics = &self.generics;

        let (generics_impl, generics_use, where_clause) = generics.split_for_impl();

        quote! {
            impl #generics_impl #clone for #name #generics_use #where_clause {
                fn clone_unchecked(&self) -> Self {
                    Self {
                        #(#fields),*
                    }
                }
            }
        }
    }

    fn launch_ty(&self) -> proc_macro2::TokenStream {
        let name = &self.name_launch;
        let fields = self.fields.iter().map(TypeField::launch_field);
        let generics = self.expanded_generics();
        let where_clause = self.launch_arg_where();
        let vis = &self.vis;

        quote! {
            #vis struct #name #generics #where_clause {
                #(#fields),*
            }
        }
    }

    fn launch_new(&self) -> proc_macro2::TokenStream {
        let args = self.fields.iter().map(TypeField::launch_new_arg);
        let fields = self.fields.iter().map(|field| &field.ident);
        let name = &self.name_launch;

        let generics = self.expanded_generics();
        let (generics_impl, generics_use, _) = generics.split_for_impl();
        let where_clause = self.launch_arg_where();
        let vis = &self.vis;

        quote! {
            impl #generics_impl #name #generics_use #where_clause {
                /// New kernel
                #[allow(clippy::too_many_arguments)]
                #vis fn new(#(#args),*) -> Self {
                    Self {
                        #(#fields),*
                    }
                }
            }
        }
    }

    fn register_impl(&self) -> proc_macro2::TokenStream {
        let kernel_launcher = prelude_type("KernelLauncher");
        let launch_arg = prelude_type("LaunchArg");
        let compilation_ty = self.compilation_ty_ident();
        let fields =
            self.fields.iter().map(TypeField::split).map(
                |(_, ident, ty, comptime)| match comptime {
                    true => quote![#ident: arg.#ident],
                    false => quote![#ident: <#ty as #launch_arg>::register(arg.#ident, launcher)],
                },
            );

        quote! {
            fn register(arg: Self::RuntimeArg, launcher: &mut #kernel_launcher) -> Self::CompilationArg {
                #compilation_ty {
                    #(#fields),*
                }
            }
        }
    }

    fn cube_type_impl(&self) -> proc_macro2::TokenStream {
        let cube_type = prelude_type("CubeType");
        let name = &self.ident;
        let name_expand = &self.name_expand;

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        quote! {
            impl #generics #cube_type for #name #generic_names #where_clause {
                type ExpandType = #name_expand #generic_names;
            }
        }
    }

    fn compilation_ty_ident(&self) -> Ident {
        Ident::new(
            format!("{}CompilationArg", self.ident).as_str(),
            self.ident.span(),
        )
    }

    fn compilation_ty(&self, name: &Ident) -> proc_macro2::TokenStream {
        let name_debug = &self.ident;
        let fields = self.fields.iter().map(TypeField::compilation_arg_field);
        let generics = &self.generics;
        let (type_generics_names, impl_generics, _) = self.generics.split_for_impl();
        let vis = &self.vis;
        let where_clause = self.launch_arg_where();

        fn generate<'a, F: Fn(&Ident) -> TokenStream>(
            fields: impl Iterator<Item = &'a TypeField>,
            func: F,
        ) -> Vec<TokenStream> {
            fields
                .map(|field| func(field.ident.as_ref().unwrap()))
                .collect::<Vec<_>>()
        }

        let clone = generate(self.fields.iter(), |name| quote!(#name: self.#name.clone()));
        let hash = generate(self.fields.iter(), |name| quote!(self.#name.hash(state)));
        let partial_eq = generate(
            self.fields.iter(),
            |name| quote!(self.#name.eq(&other.#name)),
        );
        let debug = generate(
            self.fields.iter(),
            |name| quote!(.field(stringify!(#name), &self.#name)),
        );

        quote! {
            #vis struct #name #generics #where_clause {
                #(#fields),*
            }

            impl #type_generics_names Clone for #name #impl_generics #where_clause {
                fn clone(&self) -> Self {
                    Self {
                        #(#clone,)*
                    }
                }
            }

            impl #type_generics_names core::hash::Hash for #name #impl_generics #where_clause {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    #(#hash;)*
                }
            }

            impl #type_generics_names core::cmp::PartialEq for #name #impl_generics #where_clause {
                fn eq(&self, other: &Self) -> bool {
                    #(#partial_eq &&)* true
                }
            }

            impl #type_generics_names core::fmt::Debug for #name #impl_generics #where_clause {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct(stringify!(#name_debug))
                    #(#debug)*
                    .finish()
                }
            }
            impl #type_generics_names core::cmp::Eq for #name #impl_generics #where_clause { }
        }
    }

    fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let cube_type = prelude_type("CubeType");
        let body_input =
            self.fields
                .iter()
                .map(TypeField::split)
                .map(|(_vis, name, ty, is_comptime)| {
                    if is_comptime {
                        quote![#name: arg.#name.clone()]
                    } else {
                        quote![#name: <#ty as #launch_arg>::expand(&arg.#name, builder)]
                    }
                });

        let name = &self.ident;
        let name_launch = &self.name_launch;
        let name_expand = &self.name_expand;

        let (type_generics, type_generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        let register_impl = self.register_impl();

        let (_, compilation_generics, _) = self.generics.split_for_impl();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        let compilation_ident = self.compilation_ty_ident();
        let compilation_arg = self.compilation_ty(&compilation_ident);

        quote! {
            #compilation_arg

            impl #type_generics #launch_arg for #name #type_generic_names #where_clause {
                type RuntimeArg = #name_launch #all_generic_names;
                type CompilationArg = #compilation_ident #compilation_generics;

                #register_impl

                fn expand(
                    arg: &Self::CompilationArg,
                    builder: &mut KernelBuilder,
                ) -> <Self as #cube_type>::ExpandType {
                    #name_expand {
                        #(#body_input),*
                    }
                }
            }
        }
    }

    fn expand_type_impl(&self) -> proc_macro2::TokenStream {
        let into_expand = prelude_type("IntoExpand");
        let into_mut = prelude_type("IntoMut");
        let debug = prelude_type("CubeDebug");
        let as_ref = prelude_type("AsRefExpand");
        let as_mut = prelude_type("AsMutExpand");
        let scope = prelude_type("Scope");

        let name_expand = &self.name_expand;
        let reference_impls = reference_impls(
            name_expand.as_ref().expect("should be set when parsed"),
            &self.generics,
        );
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let body = self
            .fields
            .iter()
            .map(TypeField::split)
            .map(|(_, ident, _, is_comptime)| {
                if is_comptime {
                    quote![#ident: self.#ident]
                } else {
                    quote![#ident: #into_mut::into_mut(self.#ident, scope)]
                }
            });

        quote! {
            impl #generics #into_expand for #name_expand #generic_names #where_clause {
                type Expand = Self;

                fn into_expand(self, _: &#scope) -> Self {
                    self
                }
            }

            impl #generics #into_mut for #name_expand #generic_names #where_clause {
                fn into_mut(self, scope: &#scope) -> Self {
                    Self {
                        #(#body),*
                    }
                }
            }

            impl #generics #debug for #name_expand #generic_names #where_clause {}
            impl #generics #as_ref for #name_expand #generic_names #where_clause {
                fn __expand_ref_method(&self, _: &#scope) -> &Self {
                    self
                }
            }
            impl #generics #as_mut for #name_expand #generic_names #where_clause {
                fn __expand_ref_mut_method(&mut self, _: &#scope) -> &mut Self {
                    self
                }
            }
            #reference_impls
        }
    }

    /// `CallArg` for the expand type, so the struct can be an argument of a
    /// device function: its runtime fields become parameters, its comptime
    /// fields part of the specialization. A field that can't be passed, or a
    /// comptime field whose type isn't known to be `Hash`, makes a call taking
    /// the struct trace inline. A struct without comptime fields is returned
    /// by its runtime fields.
    fn call_arg_impl(&self) -> proc_macro2::TokenStream {
        let call = frontend_type("call");
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        if self.no_call_arg.is_present() {
            return quote! {
                impl #generics #call::CallArg for #name_expand #generic_names #where_clause {}
            };
        }

        let runtime_names: Vec<_> = self
            .fields
            .iter()
            .filter(|it| !it.comptime.is_present())
            .map(|it| it.ident.as_ref().unwrap())
            .collect();
        let comptime_names: Vec<_> = self
            .fields
            .iter()
            .filter(|it| it.comptime.is_present())
            .map(|it| it.ident.as_ref().unwrap())
            .collect();

        // Returned by its runtime fields, in order. A struct with comptime
        // fields can't be rebuilt from results alone, so a call returning one
        // traces inline.
        let returns = match comptime_names.is_empty() {
            true => quote! {
                fn call_returns(
                    &self,
                    scope: &#call::__private::Scope,
                ) -> ::core::option::Option<#call::__private::Vec<#call::__private::Value>> {
                    let mut values = #call::__private::Vec::new();
                    #(values.extend(#call::CallArg::call_returns(&self.#runtime_names, scope)?);)*
                    ::core::option::Option::Some(values)
                }

                fn call_from_results(
                    results: &mut dyn ::core::iter::Iterator<Item = #call::__private::Value>,
                ) -> Self {
                    Self {
                        #(#runtime_names: #call::CallArg::call_from_results(results),)*
                    }
                }
            },
            false => quote![],
        };
        quote! {
            impl #generics #call::CallArg for #name_expand #generic_names #where_clause {
                fn call_slots(
                    &self,
                    scope: &#call::__private::Scope,
                    slots: &mut #call::__private::Vec<#call::CallSlot>,
                ) -> bool {
                    true #(&& #call::CallArg::call_slots(&self.#runtime_names, scope, slots))*
                }

                fn call_key(&self, hasher: &mut dyn ::core::hash::Hasher) -> bool {
                    #[allow(unused_imports)]
                    use #call::{HashComptime as _, NoHashComptime as _};
                    true
                        #(&& #call::CallArg::call_key(&self.#runtime_names, hasher))*
                        #(&& (&#call::HashProbe(&self.#comptime_names)).hash_comptime(hasher))*
                }

                fn call_rebuild(
                    &self,
                    scope: &#call::__private::Scope,
                    params: &mut dyn ::core::iter::Iterator<Item = #call::__private::Value>,
                ) -> Self {
                    Self {
                        #(#runtime_names: #call::CallArg::call_rebuild(
                            &self.#runtime_names,
                            scope,
                            params,
                        ),)*
                        #(#comptime_names: ::core::clone::Clone::clone(&self.#comptime_names),)*
                    }
                }

                #returns
            }
        }
    }

    fn launch_arg_where(&self) -> Option<WhereClause> {
        if self.skip_bounds.is_present() {
            return self.generics.where_clause.clone();
        }
        let launch_arg = prelude_type("LaunchArg");
        let fields = self
            .fields
            .iter()
            .filter(|it| !it.comptime.is_present())
            .cloned();
        bounded_where_clause(&self.generics, fields, |param| quote![#param: #launch_arg])
    }
}

impl TypeField {
    pub fn expand_field(&self) -> TokenStream {
        let cube_type = prelude_type("CubeType");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;
        if self.comptime.is_present() {
            quote![#vis #name: #ty]
        } else {
            quote![#vis #name: <#ty as #cube_type>::ExpandType]
        }
    }

    pub fn clone_field(&self) -> TokenStream {
        let clone = prelude_type("ExpandTypeClone");
        let name = self.ident.as_ref().unwrap();
        let is_comptime = self.comptime.is_present();
        if is_comptime {
            quote![#name: self.#name.clone()]
        } else {
            quote![#name: #clone::clone_unchecked(&self.#name)]
        }
    }

    pub fn launch_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::RuntimeArg]
        } else {
            quote![#vis #name: #ty]
        }
    }

    pub fn launch_new_arg(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#name: <#ty as #launch_arg>::RuntimeArg]
        } else {
            quote![#name: #ty]
        }
    }

    pub fn compilation_arg_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::CompilationArg]
        } else {
            quote![#vis #name: #ty]
        }
    }

    pub fn split(&self) -> (&Visibility, &Ident, &Type, bool) {
        (
            &self.vis,
            self.ident.as_ref().unwrap(),
            &self.ty,
            self.comptime.is_present(),
        )
    }
}
