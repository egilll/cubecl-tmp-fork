use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Member, Type, Visibility, WhereClause};

use super::generate::reference_impls;
use crate::{
    generate::bounded_where_clause,
    parse::{
        cube_type::{CubeTypeStruct, TypeField},
        kernel::names_generic,
    },
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
            let tuple_constructor = self.tuple_constructor();
            let named_constructor = self.named_constructor();

            quote! {
                #expand_ty
                #clone_impl
                #cube_type_impl
                #expand_type_impl
                #call_arg_impl
                #tuple_constructor
                #named_constructor
            }
        }
    }

    /// The expanded tuple constructor, reachable through the struct's own
    /// path, so `Name(a, b)` in a kernel needs no import of the expand type
    /// and infers generics from its arguments as in Rust.
    fn tuple_constructor(&self) -> TokenStream {
        if !self.tuple || self.foreign.is_present() {
            return TokenStream::new();
        }
        let name = &self.ident;
        let expand = self.name_expand.as_ref().unwrap();
        let vis = &self.vis;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        // Phantom markers carry no value, so the constructor takes none for
        // them: a marker-free function named after the type, such as one
        // behind a type alias, expands through it too.
        let mut params = Vec::new();
        let mut types = Vec::new();
        let values = self.fields.iter().enumerate().map(|(index, field)| {
            if field.is_marker() {
                return quote![::core::marker::PhantomData];
            }
            let param = quote::format_ident!("field_{index}");
            let ty = &field.ty;
            params.push(param.clone());
            if field.comptime.is_present() {
                types.push(quote![#ty]);
                return quote![#param];
            }
            if names_generic(ty, &self.generics) {
                let expand_arg = frontend_type("ExpandArg");
                types.push(quote![#expand_arg<#ty>]);
                return quote![#param.into_inner()];
            }
            let cube_type = prelude_type("CubeType");
            types.push(quote![<#ty as #cube_type>::ExpandType]);
            quote![#param]
        });
        let values: Vec<_> = values.collect();
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                #[doc(hidden)]
                #[allow(clippy::too_many_arguments)]
                #vis fn __expand_tuple_constructor(#(#params: #types),*) -> #expand #ty_generics {
                    #expand(#(#values),*)
                }
            }
        }
    }

    /// The expanded constructor of a struct literal, `Name { a, b }` in a
    /// kernel, taking the fields in the order of their names so a literal
    /// can pass them whatever order it writes them in. Like the tuple
    /// constructor, it infers the struct's generics from its arguments.
    fn named_constructor(&self) -> TokenStream {
        if self.tuple || self.fields.is_empty() || self.foreign.is_present() {
            return TokenStream::new();
        }
        let name = &self.ident;
        let expand = self.name_expand.as_ref().unwrap();
        let vis = &self.vis;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        let mut valued: Vec<_> = self
            .fields
            .iter()
            .filter(|field| !field.is_marker())
            .collect();
        valued.sort_by_key(|field| field.ident.as_ref().unwrap().to_string());
        let params = valued.iter().map(|field| {
            let ident = field.ident.as_ref().unwrap();
            let ty = &field.ty;
            if field.comptime.is_present() {
                quote![#ident: #ty]
            } else if names_generic(ty, &self.generics) {
                let expand_arg = frontend_type("ExpandArg");
                quote![#ident: #expand_arg<#ty>]
            } else {
                let cube_type = prelude_type("CubeType");
                quote![#ident: <#ty as #cube_type>::ExpandType]
            }
        });
        let values = self.fields.iter().map(|field| {
            let ident = field.ident.as_ref().unwrap();
            if field.is_marker() {
                quote![#ident: ::core::marker::PhantomData]
            } else if !field.comptime.is_present() && names_generic(&field.ty, &self.generics) {
                quote![#ident: #ident.into_inner()]
            } else {
                quote![#ident]
            }
        });
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                #[doc(hidden)]
                #[allow(clippy::too_many_arguments)]
                #vis fn __expand_named_constructor(#(#params),*) -> #expand #ty_generics {
                    #expand { #(#values),* }
                }
            }
        }
    }

    fn declaration(
        &self,
        name: &Ident,
        fields: Vec<TokenStream>,
        where_clause: Option<&WhereClause>,
    ) -> TokenStream {
        let vis = &self.vis;
        let generics = &self.generics;
        if self.tuple {
            quote![#vis struct #name #generics (#(#fields),*) #where_clause;]
        } else {
            quote![#vis struct #name #generics #where_clause { #(#fields),* }]
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let (expand_derives, field_bounded) = self.expand_derives();

        let fields = self.fields.iter().map(TypeField::expand_field);
        let name = &self.name_expand;
        let generics = &self.generics;

        let declaration = self.declaration(
            name.as_ref().unwrap(),
            fields.collect(),
            generics.where_clause.as_ref(),
        );
        quote! { #expand_derives #declaration #field_bounded }
    }

    /// The expand type's derives. On a generic struct, `Clone` and `Copy` are
    /// implemented bounded on the fields' expand types, which are what the
    /// expand type holds; the standard derives would bound the type
    /// parameters instead, so a struct over `R` would need `R: Copy` rather
    /// than `R::ExpandType: Copy`.
    fn expand_derives(&self) -> (TokenStream, TokenStream) {
        let Some(meta) = &self.derive else {
            return (TokenStream::new(), TokenStream::new());
        };
        let syn::Meta::List(list) = meta else {
            return (quote![#[#meta]], TokenStream::new());
        };
        let Ok(paths) = list.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        ) else {
            return (quote![#[#meta]], TokenStream::new());
        };
        if self.generics.type_params().next().is_none() {
            return (quote![#[#meta]], TokenStream::new());
        }
        let is =
            |path: &syn::Path, name: &str| path.segments.last().is_some_and(|it| it.ident == name);
        let clone = paths.iter().any(|path| is(path, "Clone"));
        let copy = paths.iter().any(|path| is(path, "Copy"));
        let rest: Vec<_> = paths
            .iter()
            .filter(|path| !is(path, "Clone") && !is(path, "Copy"))
            .collect();
        let derives = (!rest.is_empty())
            .then(|| quote![#[derive(#(#rest),*)]])
            .unwrap_or_default();

        let cube_type = prelude_type("CubeType");
        let held = |field: &TypeField| {
            let ty = &field.ty;
            if field.comptime.is_present() {
                quote![#ty]
            } else {
                quote![<#ty as #cube_type>::ExpandType]
            }
        };
        let name = self.name_expand.as_ref().unwrap();
        let (generics_impl, generics_use, where_clause) = self.generics.split_for_impl();
        let predicates = where_clause.map(|clause| {
            let predicates = &clause.predicates;
            quote![#predicates,]
        });
        let bounded = |bound: TokenStream| {
            let fields = self.fields.iter().map(held);
            quote![where #predicates #(#fields: #bound,)*]
        };
        let mut impls = TokenStream::new();
        if clone {
            let where_clause = bounded(quote![::core::clone::Clone]);
            let members = self.fields.iter().map(TypeField::member);
            let body = if self.tuple {
                quote![Self(#(::core::clone::Clone::clone(&self.#members)),*)]
            } else {
                let values = self.fields.iter().map(TypeField::member);
                quote![Self { #(#members: ::core::clone::Clone::clone(&self.#values)),* }]
            };
            impls.extend(quote! {
                #[automatically_derived]
                impl #generics_impl ::core::clone::Clone for #name #generics_use #where_clause {
                    fn clone(&self) -> Self {
                        #body
                    }
                }
            });
        }
        if copy {
            let where_clause = bounded(quote![::core::marker::Copy]);
            impls.extend(quote! {
                #[automatically_derived]
                impl #generics_impl ::core::marker::Copy for #name #generics_use #where_clause {}
            });
        }
        (derives, impls)
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
        let where_clause = self.launch_arg_where();

        self.declaration(
            name.as_ref().unwrap(),
            fields.collect(),
            where_clause.as_ref(),
        )
    }

    fn launch_new(&self) -> proc_macro2::TokenStream {
        // Phantom markers carry no value, so a launch takes no argument for them.
        let args = self
            .fields
            .iter()
            .filter(|field| !field.is_marker())
            .map(TypeField::launch_new_arg);
        let fields = self.fields.iter().map(|field| {
            let member = field.member();
            if field.is_marker() {
                quote![#member: ::core::marker::PhantomData]
            } else if field.ident.is_some() {
                // Shorthand, since the binding takes the field's own name and span.
                quote![#member]
            } else {
                let binding = field.binding();
                quote![#member: #binding]
            }
        });
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
        let expanded = prelude_type("Expanded");
        let cube_type = prelude_type("CubeType");
        let name = &self.ident;
        let name_expand = &self.name_expand;

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        quote! {
            impl #generics #cube_type for #name #generic_names #where_clause {
                type ExpandType = #name_expand #generic_names;
            }
            impl #generics #expanded for #name_expand #generic_names #where_clause {
                type Host = #name #generic_names;
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
        let (type_generics_names, impl_generics, _) = self.generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        fn generate<'a, F: Fn(Member) -> TokenStream>(
            fields: impl Iterator<Item = &'a TypeField>,
            func: F,
        ) -> Vec<TokenStream> {
            fields.map(|field| func(field.member())).collect::<Vec<_>>()
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

        let declaration = self.declaration(name, fields.collect(), where_clause.as_ref());
        quote! {
            #declaration

            #[automatically_derived]
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
            &self.ident,
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
            .map(TypeField::member)
            .collect();
        let comptime_names: Vec<_> = self
            .fields
            .iter()
            .filter(|it| it.comptime.is_present() && !it.is_marker())
            .map(TypeField::member)
            .collect();

        let marker_names: Vec<_> = self
            .fields
            .iter()
            .filter(|it| it.is_marker())
            .map(TypeField::member)
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
                        #(#marker_names: ::core::marker::PhantomData,)*
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

                fn call_key(&self, mut hasher: &mut dyn ::core::hash::Hasher) -> bool {
                    #[allow(unused_imports)]
                    use #call::{HashComptime as _, NoHashComptime as _};
                    ::core::hash::Hash::hash(&::core::any::type_name::<Self>(), &mut hasher);
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
                        #(#marker_names: ::core::marker::PhantomData,)*
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
        let mut generics = self.generics.clone();
        generics.where_clause =
            bounded_where_clause(&generics, fields, |param| quote![#param: #launch_arg]);
        bounded_where_clause(
            &generics,
            self.fields
                .iter()
                .filter(|field| field.is_marker())
                .cloned(),
            |param| quote![#param: Send + Sync + 'static],
        )
    }
}

impl TypeField {
    pub fn expand_field(&self) -> TokenStream {
        let cube_type = prelude_type("CubeType");
        let vis = &self.vis;
        let member = self.member();
        let name = self.ident.as_ref().map(|_| quote![#member:]);
        let ty = &self.ty;
        if self.comptime.is_present() {
            quote![#vis #name #ty]
        } else {
            quote![#vis #name <#ty as #cube_type>::ExpandType]
        }
    }

    pub fn clone_field(&self) -> TokenStream {
        let clone = prelude_type("ExpandTypeClone");
        let name = self.member();
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
        let member = self.member();
        let name = self.ident.as_ref().map(|_| quote![#member:]);
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name <#ty as #launch_arg>::RuntimeArg]
        } else {
            quote![#vis #name #ty]
        }
    }

    pub fn launch_new_arg(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = self.binding();
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
        let member = self.member();
        let name = self.ident.as_ref().map(|_| quote![#member:]);
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name <#ty as #launch_arg>::CompilationArg]
        } else {
            quote![#vis #name #ty]
        }
    }

    pub fn split(&self) -> (&Visibility, Member, &Type, bool) {
        (
            &self.vis,
            self.member(),
            &self.ty,
            self.comptime.is_present(),
        )
    }
}
