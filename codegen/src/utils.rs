use hookdef::last_seg;
use quote::{format_ident, quote_spanned, ToTokens};
use syn::spanned::Spanned as _;
use syn::{Expr, Ident, Type, TypePtr};

/// - "type", - "*mut type"
/// the former is input to native function,
/// the latter is output from native function
#[derive(PartialEq, Eq)]
pub enum ElementMode {
    Input,
    Output,
    Skip,
}

pub struct Element {
    pub name: Ident,
    pub ty: Type,
    pub mode: ElementMode,
    pub pass_by: PassBy,
    pub is_void_ptr: bool,
}

impl Element {
    pub fn get_exe_ptr_ident(&self) -> Ident {
        format_ident!("{}__ptr", self.name)
    }
}

pub enum PassBy {
    InputValue,
    SinglePtr,
    ArrayPtr { len: Box<Expr>, cap: Option<Box<Expr>> },
    InputCStr,
}

pub fn is_shadow_desc_type(ty: &Type) -> bool {
    [
        "cudnnTensorDescriptor_t",
        "cudnnFilterDescriptor_t",
        "cudnnConvolutionDescriptor_t",
    ]
    .contains(&ty.to_token_stream().to_string().as_str())
}

pub fn is_async_return_type(ty: &Type) -> bool {
    [
        "cublasStatus_t",
        "CUresult",
        "cudaError_t",
        "cudnnStatus_t",
        "nvmlReturn_t",
        "ncclResult_t",
    ].contains(&ty.to_token_stream().to_string().as_str())
}

pub fn is_void_ptr(ptr: &TypePtr) -> bool {
    last_seg(&ptr.elem).is_some_and(|seg| seg == "c_void")
}

pub fn is_const_cstr(ptr: &TypePtr) -> bool {
    ptr.const_token.is_some() && last_seg(&ptr.elem).is_some_and(|seg| seg == "c_char")
}

pub fn define_usize_from(ident: &Ident, expr: &Expr) -> proc_macro2::TokenStream {
    quote_spanned! {expr.span()=>
        #[allow(clippy::useless_conversion)]
        let #ident = usize::try_from((#expr).to_owned()).unwrap();
    }
}
