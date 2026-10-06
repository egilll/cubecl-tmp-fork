use darling::{FromDeriveInput, FromField, ast::Data, uses_type_params, util::Flag};
use quote::format_ident;
use syn::{Generics, Ident, Index, Member, Type, Visibility};

use crate::generate::RuntimeField;

#[derive(FromDeriveInput, Debug)]
#[darling(supports(struct_named, struct_tuple, struct_unit), attributes(expand, cube, launch), map = unwrap_fields)]
pub struct CubeTypeStruct {
    pub ident: Ident,
    pub name_launch: Option<Ident>,
    pub name_comptime: Option<Ident>,
    pub name_expand: Option<Ident>,
    data: Data<(), TypeField>,
    #[darling(skip)]
    pub fields: Vec<TypeField>,
    pub generics: Generics,
    pub vis: Visibility,
    pub skip_bounds: Flag,
    pub derive: Option<syn::Meta>,
    /// `#[cube(no_call_arg)]`: a device function never takes the struct, and
    /// a call passing it is traced inline. For a struct whose methods need
    /// what only the kernel's own arguments carry, such as a tensor's
    /// metadata.
    pub no_call_arg: Flag,
}

#[derive(FromField, Clone, Debug)]
#[darling(attributes(expand, cube))]
pub struct TypeField {
    pub vis: Visibility,
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
    #[darling(skip)]
    pub index: usize,
}

uses_type_params!(TypeField, ty);
impl RuntimeField for TypeField {
    fn ty(self) -> Type {
        self.ty
    }
}

fn unwrap_fields(mut ty: CubeTypeStruct) -> CubeTypeStruct {
    // This will be supported inline with the next darling release
    let fields = ty.data.as_ref().take_struct().unwrap().fields;
    ty.fields = fields
        .into_iter()
        .cloned()
        .enumerate()
        .map(|(index, mut field)| {
            field.index = index;
            field
        })
        .collect();

    let name = &ty.ident;
    ty.name_expand
        .get_or_insert_with(|| format_ident!("{name}Expand"));
    ty.name_launch
        .get_or_insert_with(|| format_ident!("{name}Launch"));
    ty.name_comptime
        .get_or_insert_with(|| format_ident!("{name}Comptime"));

    ty
}

impl CubeTypeStruct {
    pub fn expanded_generics(&self) -> Generics {
        self.generics.clone()
    }
}

impl TypeField {
    pub fn member(&self) -> Member {
        match &self.ident {
            Some(ident) => Member::Named(ident.clone()),
            None => Member::Unnamed(Index::from(self.index)),
        }
    }

    pub fn binding(&self) -> Ident {
        self.ident
            .clone()
            .unwrap_or_else(|| format_ident!("field{}", self.index))
    }
}
