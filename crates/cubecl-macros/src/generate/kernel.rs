use std::collections::HashMap;

use darling::usage::{CollectLifetimes as _, CollectTypeParams as _, GenericsExt as _, Purpose};
use inflections::case::to_snake_case;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{Ident, Type, TypeParamBound, parse_quote};

use crate::{
    parse::{
        kernel::{
            DefinedGeneric, ExecutionMode, InlineHint, KernelBody, KernelFn, Launch,
            anon_lifetime_to_static, expand_kernel_ty, map_type_normalized, strip_ref,
        },
        signature::KernelReturns,
    },
    paths::{core_type, frontend_type, prelude_type},
};

impl ToTokens for ExecutionMode {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = prelude_type("ExecutionMode");
        tokens.extend(match self {
            ExecutionMode::Checked => quote![#ty::Checked],
            ExecutionMode::Unchecked => quote![#ty::Unchecked],
        });
    }
}

impl KernelFn {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let attrs = &self.attrs;
        let vis = &self.vis;
        let sig = &self.sig;

        let body = match &self.body {
            KernelBody::Block(block) => match matches!(sig.returns, KernelReturns::ExpandType(_))
                && !self.context.is_intrinsic
            {
                true => &block.to_tokens_runtime_return(&mut self.context),
                false => &block.to_tokens(&mut self.context),
            },
            KernelBody::Verbatim(tokens) => tokens,
        };
        let name = &self.full_name;

        let cfg_debug = cfg!(debug_symbols) && !self.args.no_debug_symbols.is_present();
        let (debug_source, debug_params) = if cfg_debug || self.args.debug_symbols.is_present() {
            let debug_source = frontend_type("debug_source_expand");
            let cube_debug = frontend_type("CubeDebug");
            let src_file = self.args.src_file.as_ref().map(|file| file.value());
            let src_file = src_file.or_else(|| {
                let span: proc_macro::Span = self.span.unwrap();
                let source_path = span.local_file();
                let source_file = source_path.as_ref().and_then(|path| path.file_name());
                source_file.map(|file| file.to_string_lossy().into())
            });
            let source_text = match src_file {
                Some(file) => quote![include_str!(#file)],
                None => quote![""],
            };

            let debug_source = quote_spanned! {self.span=>
                #debug_source(scope, #name, file!(), #source_text, line!(), column!())
            };
            let debug_params = sig
                .runtime_params()
                .map(|it| &it.name)
                .map(|name| {
                    let name_str = name.to_string();
                    quote! [#cube_debug::set_debug_name(&#name, scope, #name_str);]
                })
                .collect();
            (debug_source, debug_params)
        } else {
            (TokenStream::new(), Vec::new())
        };
        let body = self
            .args
            .fast_math
            .as_ref()
            .map(|value| {
                let fast_math = frontend_type("fast_math_expand");
                quote![#fast_math(scope, #value, |scope| {#body})]
            })
            .unwrap_or_else(|| quote![#body]);
        // Only the item holding the function's own body becomes a device
        // function: a method's other items forward to it.
        let body = match self.body {
            KernelBody::Block(_) => self.device_call_body(body),
            KernelBody::Verbatim(_) => body,
        };
        let imports = trait_imports();
        let mappings = self.sig.define_mappings();
        let registers = self
            .analysis
            .register_types(mappings, quote![scope], false, false);

        let out = quote! {
            #[allow(unused_mut)]
            #(#attrs)*
            #vis #sig {
                #debug_source;
                #(#debug_params)*
                #imports;
                #registers

                #body
            }
        };

        out
    }
}

impl KernelFn {
    /// Wrap the expansion of a function so its calls are device function
    /// calls: traced once per specialization into a device function of the
    /// kernel, and called. See `cubecl_core::frontend::call`.
    ///
    /// The body becomes one closure taking the runtime arguments. The call
    /// tries a device function first, and calls the closure inline when that
    /// can't be: a signature the call can't express is decided here, an
    /// argument or result that can't be passed while tracing.
    fn device_call_body(&self, body: TokenStream) -> TokenStream {
        let never = match self.args.inline {
            Some(InlineHint::Always) => return body,
            Some(InlineHint::Never) => true,
            None => false,
        };
        let refuse = |span: proc_macro2::Span, reason: &str| match never {
            true => syn::Error::new(span, format!("`#[cube(inline(never))]`: {reason}"))
                .into_compile_error(),
            false => body.clone(),
        };
        if self.args.is_launch() {
            return refuse(self.span, "a kernel entry point is never called");
        }
        if self.context.is_intrinsic {
            return refuse(self.span, "an intrinsic is expanded at its call");
        }

        let returns = match &self.sig.returns {
            KernelReturns::ExpandType(ty) => match expand_kernel_ty(ty.clone(), false) {
                Ok(ty) => ty,
                Err(err) => return err.into_compile_error(),
            },
            KernelReturns::Plain(ty) if is_self(ty) => ty.clone(),
            KernelReturns::Plain(ty) => {
                return refuse(
                    syn::spanned::Spanned::span(ty),
                    "a device function can't return a comptime value",
                );
            }
        };
        if contains_impl_trait(&returns) {
            return refuse(self.span, "a device function can't return `impl Trait`");
        }
        if contains_borrow(&returns) {
            return refuse(
                self.span,
                "a device function can't return a borrow of its arguments",
            );
        }

        // A mutable slice writes its buffer in place, so it passes as a slice;
        // any other `&mut` would need its referent copied back.
        let is_mut_ref = |ty: &Type| matches!(ty, Type::Reference(r) if r.mutability.is_some());
        let is_mut_slice = |ty: &Type| matches!(ty, Type::Reference(r) if r.mutability.is_some() && matches!(*r.elem, Type::Slice(_)));
        for param in self.sig.runtime_params() {
            if is_mut_ref(&param.ty) && !is_mut_slice(&param.ty) {
                return refuse(
                    param.name.span(),
                    "a device function can't take `&mut` arguments other than slices yet",
                );
            }
            if contains_impl_trait(&param.normalized_ty) {
                return refuse(
                    param.name.span(),
                    "a device function can't take `impl Trait` arguments",
                );
            }
        }

        let call = frontend_type("call");
        let scope_ty = prelude_type("Scope");
        let name = self.full_name.as_str();
        let self_ident = format_ident!("__call_self");
        let has_self = self.sig.runtime_params().any(|param| param.name == "self");
        let in_impl = has_self || self.full_name.contains("::");

        let runtime: Vec<_> = self.sig.runtime_params().collect();
        // The names inside the body closure: `self` can't name a closure
        // parameter.
        let bindings: Vec<_> = runtime
            .iter()
            .map(|param| match param.name == "self" {
                true => self_ident.clone(),
                false => param.name.clone(),
            })
            .collect();
        let closure_params: Vec<_> = runtime
            .iter()
            .zip(bindings.iter())
            .map(|(param, binding)| {
                // A receiver keeps its own type: `Self` is already the type
                // the method's `self` has.
                let ty = match param.name == "self" {
                    true => &param.ty,
                    false => &param.normalized_ty,
                };
                let mut_ = &param.mutability;
                quote![#mut_ #binding: #ty]
            })
            .collect();
        let names: Vec<_> = runtime.iter().map(|param| &param.name).collect();
        // Each argument as the value a call passes: a reference passes what
        // it refers to.
        let referents: Vec<_> = runtime
            .iter()
            .map(|param| {
                let name = &param.name;
                match &param.ty {
                    Type::Reference(_) => quote![&*#name],
                    _ => quote![&#name],
                }
            })
            .collect();
        let rebuilt: Vec<_> = (0..runtime.len())
            .map(|i| format_ident!("__call_arg_{i}"))
            .collect();
        let passed: Vec<_> = runtime
            .iter()
            .zip(rebuilt.iter())
            .map(|(param, rebuilt)| match &param.ty {
                Type::Reference(r) if r.mutability.is_some() => quote![&mut #rebuilt],
                Type::Reference(_) => quote![&#rebuilt],
                _ => quote![#rebuilt],
            })
            .collect();

        let comptime: Vec<_> = self.sig.comptime_params().map(|it| &it.name).collect();
        let mut type_names: Vec<TokenStream> = self
            .sig
            .generics
            .type_params()
            .map(|it| {
                let ident = &it.ident;
                quote![::core::any::type_name::<#ident>()]
            })
            .collect();
        if in_impl {
            type_names.push(quote![::core::any::type_name::<Self>()]);
        }
        let const_params: Vec<_> = self
            .sig
            .generics
            .const_params()
            .map(|it| &it.ident)
            .collect();
        let body = match has_self {
            true => rename_self(body, &self_ident),
            false => body,
        };

        quote! {
            #[allow(clippy::redundant_closure_call, unused_mut)]
            let __call_body = |scope: &#scope_ty, #(#closure_params),*| -> #returns {
                #(let #comptime = ::core::clone::Clone::clone(&#comptime);)*
                #body
            };
            let __call_comptime = {
                #[allow(unused_imports)]
                use #call::{HashComptime as _, NoHashComptime as _};
                let mut __hasher = #call::call_hasher();
                let __hashed = true
                    #(&& (&#call::HashProbe(&#comptime)).hash_comptime(&mut __hasher))*;
                #(#call::hash_const(&mut __hasher, &#const_params);)*
                __hashed.then(|| ::core::hash::Hasher::finish(&__hasher))
            };
            let __call_args = __call_comptime.and_then(|_| {
                let mut __slots = #call::__private::Vec::new();
                let mut __hasher = #call::call_hasher();
                let __passable = true
                    #(&& #call::CallArg::call_slots(#referents, scope, &mut __slots)
                        && #call::CallArg::call_key(#referents, &mut __hasher))*;
                __passable.then(|| #call::CallArgs {
                    slots: __slots,
                    key: ::core::hash::Hasher::finish(&__hasher),
                })
            });
            let __call_result = #call::device_call(
                scope,
                ::core::concat!(::core::module_path!(), "::", #name),
                &[#(#type_names),*],
                __call_comptime.unwrap_or_default(),
                __call_args,
                |scope: &#scope_ty,
                 __params: &mut dyn ::core::iter::Iterator<Item = #call::__private::Value>| {
                    #(#[allow(unused_mut)] let mut #rebuilt = #call::CallArg::call_rebuild(#referents, scope, __params);)*
                    __call_body(scope, #(#passed),*)
                },
            );
            match __call_result {
                ::core::option::Option::Some(__result) => __result,
                ::core::option::Option::None => __call_body(scope, #(#names),*),
            }
        }
    }
}

fn is_self(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.is_ident("Self"))
}

/// Whether `ty` mentions a reference or a lifetime: a result borrowing an
/// argument, which a closure's return type can't tie to its parameters.
fn contains_borrow(ty: &Type) -> bool {
    struct Find(bool);
    impl syn::visit_mut::VisitMut for Find {
        fn visit_type_reference_mut(&mut self, _: &mut syn::TypeReference) {
            self.0 = true;
        }
        fn visit_lifetime_mut(&mut self, _: &mut syn::Lifetime) {
            self.0 = true;
        }
    }
    let mut find = Find(false);
    syn::visit_mut::VisitMut::visit_type_mut(&mut find, &mut ty.clone());
    find.0
}

/// Whether `ty` mentions `impl Trait`, which a closure parameter can't name.
fn contains_impl_trait(ty: &Type) -> bool {
    struct Find(bool);
    impl syn::visit_mut::VisitMut for Find {
        fn visit_type_impl_trait_mut(&mut self, _: &mut syn::TypeImplTrait) {
            self.0 = true;
        }
    }
    let mut find = Find(false);
    syn::visit_mut::VisitMut::visit_type_mut(&mut find, &mut ty.clone());
    find.0
}

/// `tokens` with the `self` keyword replaced by `with`, except as the start
/// of a path (`self::`): inside the closure an outlined method's body becomes,
/// `self` can't name the receiver.
fn rename_self(tokens: TokenStream, with: &Ident) -> TokenStream {
    use proc_macro2::{Group, TokenTree};

    let tokens: Vec<TokenTree> = tokens.into_iter().collect();
    let mut out = Vec::with_capacity(tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) if ident == "self" => {
                let is_path = matches!(
                    tokens.get(i + 1),
                    Some(TokenTree::Punct(p)) if p.as_char() == ':'
                );
                match is_path {
                    true => out.push(token.clone()),
                    false => {
                        let mut renamed = with.clone();
                        renamed.set_span(ident.span());
                        out.push(TokenTree::Ident(renamed));
                    }
                }
            }
            TokenTree::Group(group) => {
                let mut renamed = Group::new(group.delimiter(), rename_self(group.stream(), with));
                renamed.set_span(group.span());
                out.push(TokenTree::Group(renamed));
            }
            other => out.push(other.clone()),
        }
    }
    out.into_iter().collect()
}

fn trait_imports() -> TokenStream {
    let into_runtime = prelude_type("IntoRuntime");
    let assign = prelude_type("Assign");
    quote! {
        use #into_runtime as _;
        use #assign as _;
    }
}

impl Launch {
    fn kernel_phantom_data(&self) -> Option<TokenStream> {
        let generics = self.kernel_generics.clone();
        let declared_lifetimes = generics.declared_lifetimes();
        let declared_type_params = generics.declared_type_params();

        let used_lifetimes = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_lifetimes_cloned(&Purpose::Declare.into(), &declared_lifetimes);
        let used_type_params = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_type_params_cloned(&Purpose::Declare.into(), &declared_type_params);
        let lifetimes: Vec<_> = generics
            .lifetimes()
            .map(|param| &param.lifetime)
            .filter(|lifetime| !used_lifetimes.contains(*lifetime))
            .collect();
        let type_params: Vec<_> = generics
            .type_params()
            .map(|param| &param.ident)
            .filter(|ident| !used_type_params.contains(*ident))
            .collect();

        (!lifetimes.is_empty() || !type_params.is_empty()).then(
            || quote![__ty: ::core::marker::PhantomData<(#(&#lifetimes (),)* #(#type_params),*)>],
        )
    }

    pub fn compilation_args_def(&self) -> (Vec<TokenStream>, Vec<Ident>) {
        let launch_arg = prelude_type("LaunchArg");
        let mut tokens = Vec::new();
        let mut args = Vec::new();

        self.runtime_params().for_each(|input| {
            let ty = strip_ref(input.ty.clone());
            let ty = anon_lifetime_to_static(ty);
            let name = &input.name;

            tokens.push(quote! {
                #name: <#ty as #launch_arg>::CompilationArg
            });
            args.push(name.clone());
        });

        (tokens, args)
    }

    pub fn arg_registers(&self) -> (TokenStream, TokenStream) {
        let launch_arg = prelude_type("LaunchArg");
        let io_attr = prelude_type("BufferIOAttr");
        let mut defined = quote! {};
        let mut args = quote! {};

        self.runtime_params().enumerate().for_each(|(i, input)| {
            // What the signature proves about the argument's buffers: `&T`
            // cannot be written, so a launch that fails before running leaves
            // it alone; anything else may be, so it takes the failure. The
            // compiled kernel's visibility analysis overrides this once the
            // kernel compiles — this declaration is the answer that survives
            // compilation failing.
            let declared = match &input.ty {
                syn::Type::Reference(r) if r.mutability.is_none() => {
                    quote![#io_attr::ReadOnly]
                }
                _ => quote![#io_attr::ReadWrite],
            };
            let ty = strip_ref(input.ty.clone());
            let ty = anon_lifetime_to_static(ty);
            let ident = &input.name;
            let var = Ident::new(format!("comp_arg_{i}").as_str(), ident.span());

            args.extend(quote! {#var,});
            defined.extend(quote! {
                launcher.declare_io(#declared);
                let #var = <#ty as #launch_arg>::register(#ident, &mut launcher);
            });
        });

        (
            quote! {
                #defined
            },
            args,
        )
    }

    pub fn io_mappings(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let mut define = quote! {};

        let expand_fn = |ident, ty| {
            let ty = self.func.analysis.process_ty(&ty);
            let ty = strip_ref(ty);
            let ty = anon_lifetime_to_static(ty);

            quote! {
                let mut #ident = <#ty as #launch_arg>::expand(&self.#ident.dynamic_cast(), &mut builder);
            }
        };
        for param in self.runtime_params() {
            define.extend(expand_fn(&param.name, param.ty.clone()));
        }

        quote! {
            #define
        }
    }

    fn define_body(&self) -> TokenStream {
        let kernel_builder = prelude_type("KernelBuilder");
        let kernel_metadata = prelude_type("KernelMetadata");
        let io_map = self.io_mappings();
        let mut mapping = HashMap::new();
        for param in self.func.sig.parameters.iter() {
            for define in param.defines.iter() {
                match define {
                    DefinedGeneric::Single(ident) => {
                        mapping.insert(ident.clone(), (param.name.clone(), None));
                    }
                    DefinedGeneric::Multiple(ident, index) => {
                        mapping.insert(ident.clone(), (param.name.clone(), Some(*index)));
                    }
                }
            }
        }
        let mapping = self.func.sig.define_mappings();
        let register_type =
            self.func
                .analysis
                .register_types(mapping, quote![builder.scope], true, true);
        let args = self.func.sig.parameters.iter().map(|it| {
            let name = &it.name;
            match it.is_const {
                true => quote![self.#name.clone()],
                false => {
                    let mut mapped = it.ty.clone();
                    match map_type_normalized(&mut mapped, &|_| parse_quote![#name]) {
                        Ok(()) => quote![#mapped],
                        Err(err) => err.into_compile_error(),
                    }
                }
            }
        });
        let generics = self
            .func
            .analysis
            .process_generic_names(&self.func.sig.generics);

        quote! {
            let __name = <Self as #kernel_metadata>::id(self).entrypoint_name(&self.__settings.kernel_name);
            let mut builder = #kernel_builder::new(self.__settings.clone().kernel_name(__name));
            builder.runtime_properties(self.__target_properties.as_ref().clone());
            builder.device_properties(&self.__device_properties);

            #register_type
            #io_map
            expand #generics(&mut builder.scope, #(#args,)*);
            builder.build()
        }
    }

    /// Returns the kernel entrypoint name.
    /// Appropriate for usage in source code such as naming the CUDA or WGSL
    /// entrypoint.
    ///
    /// For example a kernel:
    /// ```text
    /// #[cube(launch)]
    /// fn my_kernel(input: &[f32], output: &mut [f32]) {}
    /// ```
    /// would produce the name `my_kernel`.
    ///
    /// If a generic has the `Float` or `Numeric` bound the kernel also has a
    /// suffix with the name of that type in use:
    /// ```text
    /// fn my_kernel<F: Float>(input: &[F], output: &mut [F]) {}
    /// ```
    /// now produces the name `my_kernel_f16` or `my_kernel_f32` etc. depending
    /// on which variant of the kernel is launched by the user.
    ///
    /// If a kernel has several matching bounds they are appended as suffixes in
    /// order.
    fn kernel_entrypoint_name(&self) -> TokenStream {
        // This base name is always used; a suffix might be added
        // based on generics.
        let base_name = self.func.sig.name.to_string();
        let type_name = prelude_type("type_name_short_sanitized");

        let generics = &self.func.sig.generics;
        let suffix_producing_bounds = [
            format_ident!("Float"),
            format_ident!("Numeric"),
            format_ident!("Int"),
            format_ident!("Scalar"),
            format_ident!("Size"),
        ];

        let mut matching_generics = vec![];

        // Inspect all generics for the bounds of interest in order to
        // determine if a suffix should be added
        for ty in generics.type_params() {
            for bound in &ty.bounds {
                let TypeParamBound::Trait(t) = bound else {
                    continue;
                };

                // Using last should account for the bounds such as `Float` but also
                // `some::prefix::Float`
                let Some(generic_trailing) = t.path.segments.last() else {
                    continue;
                };

                // If we find some type parameter with `Float` as a bound,
                // add a suffix based on a shortened version of the
                // type name
                if suffix_producing_bounds.contains(&generic_trailing.ident) {
                    // E.g. the `F` in `F: Float` or `N` in `N: Numeric`
                    matching_generics.push(ty.ident.clone());
                    continue;
                }
            }
        }

        if matching_generics.is_empty() {
            quote! {
                #base_name
            }
        } else {
            let mut defines = self.func.sig.define_mappings();

            let generic_names = matching_generics.iter().map(|ident| {
                let name = match defines.remove(ident) {
                    Some((name, index)) => match index {
                        Some(index) => {
                            // The defined type should be an array or vector that support indexing.
                            quote![#name[#index]]
                        }
                        None => quote![#name],
                    },
                    None => quote![#type_name::<#ident>();],
                };
                let ident_snake = to_snake_case(&ident.to_string());
                quote! {{
                    let type_name = #name;
                    name.push_str(&cubecl::__private::format!("_{}_{type_name}", #ident_snake));
                }}
            });

            quote! (
                {
                    let mut name = cubecl::__private::format!("{}", #base_name);

                    #(#generic_names)*

                    name
                }
            )
        }
    }

    pub fn kernel_definition(&self) -> TokenStream {
        if self.args.is_launch() {
            let kernel_metadata = prelude_type("KernelMetadata");
            let cube_kernel = prelude_type("CubeKernel");
            let kernel_settings = prelude_type("KernelSettings");
            let device_properties = prelude_type("DeviceProperties");
            let target_properties = prelude_type("TargetProperties");
            let private = core_type("__private");
            let kernel_definition: syn::Path = prelude_type("KernelDefinition");
            let kernel_id = prelude_type("KernelId");
            let elem_ty = prelude_type("ElemType");

            let kernel_name = self.kernel_name();
            let define = self.define_body();
            let kernel_doc = format!("{} Kernel", self.func.sig.name);

            let (generics, generic_names, where_clause) = self.kernel_generics.split_for_impl();
            let const_params: Vec<_> = self.comptime_params().collect();
            let param_names = self
                .comptime_params()
                .map(|param| param.name.clone())
                .collect::<Vec<_>>();
            let phantom_data = self.kernel_phantom_data();
            let phantom_data_init = phantom_data
                .as_ref()
                .map(|_| quote![__ty: ::core::marker::PhantomData]);
            let (compilation_args, args) = self.compilation_args_def();
            let info_names = param_names.clone().into_iter().chain(args.clone());

            let info_ty_name = format_ident!("{kernel_name}Info");
            let info_ty = self.info_ty(&info_ty_name);
            let info_generics = generic_names.as_turbofish();

            let kernel_source_name = self.kernel_entrypoint_name();
            let mut settings = quote![settings.kernel_name(#kernel_source_name)];
            let cfg_debug = cfg!(debug_symbols) && !self.args.no_debug_symbols.is_present();
            if cfg_debug || self.args.debug_symbols.is_present() {
                settings.extend(quote![.debug_symbols()]);
            }
            if let Some(cluster_dim) = &self.args.cluster_dim {
                settings.extend(quote![.cluster_dim(#cluster_dim.into())]);
            }

            quote! {
                #[doc = #kernel_doc]
                pub struct #kernel_name #generics #where_clause {
                    __settings: #kernel_settings,
                    __device_properties: #private::Arc<#device_properties>,
                    __target_properties: #private::Arc<#target_properties>,
                    #(#compilation_args,)*
                    #(#const_params,)*
                    #phantom_data
                }

                #info_ty

                #[allow(clippy::too_many_arguments)]
                impl #generics #kernel_name #generic_names #where_clause {
                    pub fn new(
                        settings: #kernel_settings,
                        device_properties: #private::Arc<#device_properties>,
                        target_properties: #private::Arc<#target_properties>,
                        #(#compilation_args,)*
                        #(#const_params),*) -> Self {
                        Self {
                            __settings: #settings,
                            __device_properties: device_properties,
                            __target_properties: target_properties,
                            #(#args,)*
                            #(#param_names,)*
                            #phantom_data_init
                        }
                    }
                }

                impl #generics #kernel_metadata for #kernel_name #generic_names #where_clause {
                    fn id(&self) -> #kernel_id {
                        // We don't use any other kernel settings with the macro.
                        let cube_dim = self.__settings.cube_dim.clone();
                        let address_type = self.__settings.address_type;

                        #kernel_id::new::<Self>()
                            .address_type(address_type)
                            .cube_dim(self.__settings.cube_dim.clone())
                            .mode(self.__settings.execution_mode)
                            .info(#info_ty_name #info_generics {
                                #(#info_names: self.#info_names.clone(),)*
                                #phantom_data_init
                            })
                    }

                    fn address_type(&self) -> #elem_ty {
                        self.__settings.address_type.unsigned_type()
                    }
                }

                impl #generics #cube_kernel for #kernel_name #generic_names #where_clause {
                    fn define(&self) -> #kernel_definition {
                        #define
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn info_ty(&self, name: &Ident) -> proc_macro2::TokenStream {
        let const_params: Vec<_> = self.comptime_params().collect();
        let param_names = self
            .comptime_params()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>();
        let phantom_data = self.kernel_phantom_data();
        let phantom_data_init = phantom_data
            .as_ref()
            .map(|_| quote![__ty: ::core::marker::PhantomData]);
        let (compilation_args, args) = self.compilation_args_def();
        let info_names = param_names
            .clone()
            .into_iter()
            .chain(args.clone())
            .collect::<Vec<_>>();

        let kernel_source_name = self.kernel_entrypoint_name();
        let mut settings = quote![settings.kernel_name(#kernel_source_name)];
        let cfg_debug = cfg!(debug_symbols) && !self.args.no_debug_symbols.is_present();
        if cfg_debug || self.args.debug_symbols.is_present() {
            settings.extend(quote![.debug_symbols()]);
        }
        if let Some(cluster_dim) = &self.args.cluster_dim {
            settings.extend(quote![.cluster_dim(#cluster_dim.into())]);
        }

        let generics = &self.kernel_generics;
        let (type_generics_names, impl_generics, where_generics) =
            self.kernel_generics.split_for_impl();
        let vis = &self.vis;

        fn map_vec<'a, T: 'a, F: Fn(&T) -> TokenStream>(
            fields: impl Iterator<Item = &'a T>,
            func: F,
        ) -> Vec<TokenStream> {
            fields.map(func).collect::<Vec<_>>()
        }

        let clone = map_vec(info_names.iter(), |name| quote!(#name: self.#name.clone()));
        let hash = map_vec(info_names.iter(), |name| quote!(self.#name.hash(state)));
        let partial_eq = map_vec(
            info_names.iter(),
            |name| quote!(self.#name.eq(&other.#name)),
        );
        let debug = map_vec(info_names.iter(), |name| {
            // For info, we want ignored fields (i.e. dynamic types) unignored
            let name_str = name.to_string();
            let name_str = name_str.strip_prefix("_").unwrap_or(&name_str);
            quote!(.field(#name_str, &self.#name))
        });

        quote! {
            #vis struct #name #generics #where_generics {
                #(#compilation_args,)*
                #(#const_params,)*
                #phantom_data
            }

            impl #type_generics_names Clone for #name #impl_generics #where_generics {
                fn clone(&self) -> Self {
                    Self {
                        #(#clone,)*
                        #phantom_data_init
                    }
                }
            }

            impl #type_generics_names core::hash::Hash for #name #impl_generics #where_generics {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    #(#hash;)*
                }
            }

            impl #type_generics_names core::cmp::PartialEq for #name #impl_generics #where_generics {
                fn eq(&self, other: &Self) -> bool {
                    #(#partial_eq &&)* true
                }
            }

            impl #type_generics_names core::fmt::Debug for #name #impl_generics #where_generics {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct(stringify!(#name))
                    #(#debug)*
                    .finish()
                }
            }
            impl #type_generics_names core::cmp::Eq for #name #impl_generics #where_generics { }
        }
    }
}
