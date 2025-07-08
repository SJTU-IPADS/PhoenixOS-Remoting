#![feature(proc_macro_diagnostic)]

use hookdef::{is_hacked_type, CustomHookAttrs, CustomHookFn};
use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_each_token_spanned, quote_spanned, TokenStreamExt as _};
use syn::{parse_macro_input, Type};

mod utils;
use utils::{define_usize_from, is_shadow_desc_type, Element, ElementMode, PassBy};

mod hook_fn;
use hook_fn::HookFn;

mod use_thread_local;

/// Basic checks on a hook declaration
#[proc_macro_attribute]
pub fn cuda_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func.into_plain_fn().into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn cuda_custom_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    if let Err(err) = CustomHookAttrs::from_macro(args.into()) {
        return err.to_compile_error().into();
    }
    parse_macro_input!(input as CustomHookFn)
        .to_plain_fn()
        .into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The derive macros for auto trait impl.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The derive procedural macro to generate `Transportable` trait implementation for a *fixed-size* type.
///
/// ### Example
/// To use this macro, annotate a struct or enum with `#[derive(Transportable)]`.
///
/// ```ignore
/// #[derive(Transportable)]
/// pub struct MyStruct {
///     ...
/// }
/// ```
///
/// This invocation generates a `Transportable` trait implementation for `MyStruct`.
/// The `Transportable` trait is used for sending and receiving the struct over a communication channel.
#[proc_macro_derive(Transportable)]
pub fn transportable_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl network::TransportableMarker for #name {}
    };

    gen.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for client usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate hijack functions for client intercepting application calls.
///
/// ### Example
/// We have a function `cudaGetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a hijack function for interception, we can write:
///
/// ```ignore
/// #[cuda_hook_hijack(proc_id = 0)]
/// fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// This invocation generates a function `cudaGetDevice` with the same signature as the original function,
/// which will intercept the call to `cudaGetDevice` and send the parameters to the server then wait for the result.
#[proc_macro_attribute]
pub fn cuda_hook_hijack(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);
    let func_str = func.to_string();

    let vars: Box<_> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .collect();

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            let name_str = name.to_string();
            let deref = match &param.pass_by {
                PassBy::InputValue => Default::default(),
                PassBy::SinglePtr => {
                    if is_hacked_type(&param.ty) {
                        quote! { let #name = unsafe { std::ptr::read_unaligned(#name) }; }
                    } else {
                        quote! { let #name = unsafe { *#name }; }
                    }
                }
                PassBy::ArrayPtr { len, .. } => {
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = define_usize_from(&len_ident, len);
                    let span = name.span();
                    quote_each_token_spanned! {tokens span
                        assert!(!#name.is_null());
                        log::trace!(target: #func_str, "[#{}] (input) {} = {:p}", client.id, #name_str, #name);
                        let #name = unsafe { std::slice::from_raw_parts(#name, #len_ident) };
                        match send_slice(#name, channel_sender) {
                            Ok(()) => {}
                            Err(e) => panic!("failed to send {}: {}", #name_str, e),
                        }
                    };
                    return tokens;
                }
                PassBy::InputCStr => {
                    return quote_spanned! {name.span()=>
                        let #name = unsafe { std::ffi::CStr::from_ptr(#name) };
                        log::trace!(target: #func_str, "[#{}] (input) {} = {:?}", client.id, #name_str, #name);
                        match send_slice(#name.to_bytes_with_nul(), channel_sender) {
                            Ok(()) => {}
                            Err(e) => panic!("failed to send {}: {}", #name_str, e),
                        }
                    }
                }
            };
            quote_spanned! {name.span()=>
                #deref
                log::trace!(target: #func_str, "[#{}] (input) {} = {:?}", client.id, #name_str, #name);
                match #name.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", #name_str, e),
                }
            }
        });

    // receive vars
    let recv_statements = vars.iter().map(|var| {
        let name = &var.name;
        let name_str = name.to_string();
        let deref = match &var.pass_by {
            PassBy::InputValue | PassBy::InputCStr => unreachable!(),
            PassBy::SinglePtr => quote! {
                assert!(!#name.is_null());
                let #name = unsafe { &mut *#name };
            },
            PassBy::ArrayPtr { len, .. } => {
                let len_ident = format_ident!("{}_len", name);
                let mut tokens = define_usize_from(&len_ident, len);
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    assert!(!#name.is_null());
                    let #name = unsafe { std::slice::from_raw_parts_mut(#name, #len_ident) };
                    match recv_slice_to(#name, channel_receiver) {
                        Ok(()) => {}
                        Err(e) => panic!("failed to send {}: {}", #name_str, e),
                    }
                };
                return tokens;
            }
        };
        quote_spanned! {name.span()=>
            // FIXME: allocate space for null pointers
            #deref
            match #name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", #name_str, e),
            }
            log::trace!(target: #func_str, "[#{}] (output) {} = {:?}", client.id, #name_str, #name);
        }
    });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        quote! { #name: #ty }
    });
    let result_name = &result.name;
    let result_ty = &result.ty;

    let async_api_return = if let Some(true) = input.is_async_api {
        quote! {
            if client.opt_async_api {
                return #result_name;
            }
        }
    } else {
        Default::default()
    };

    let (shadow_desc_send, shadow_desc_return) = if input.is_create_shadow_desc {
        let name = &vars[0].name;
        let name_str = name.to_string();
        let shadow_desc_send = quote_spanned! {name.span()=>
            if client.opt_shadow_desc {
                let resource_idx = client.resource_idx;
                unsafe {
                    *#name = resource_idx as _;
                }
                client.resource_idx += 1;
                match resource_idx.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", #name_str, e),
                }
            }
        };
        let shadow_desc_return = quote! {
            if client.opt_shadow_desc {
                return #result_name;
            }
        };
        (shadow_desc_send, shadow_desc_return)
    } else {
        Default::default()
    };

    let client_before_send = input.injections.client_before_send.iter();
    let client_extra_send = input.injections.client_extra_send.iter();
    let client_after_recv = input.injections.client_after_recv.iter();

    let modifiers = match input.parent {
        Some(_) => quote!(pub),
        None => quote!(#[no_mangle] pub extern "C"),
    };

    let gen_fn = quote! {
        // manually expanded the following macro to work around rust-analyzer bug
        // otherwise `Expand macro recursively at caret` is bugged
        // #[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
        #modifiers fn #func(#(#params),*) -> #result_ty {
        CLIENT_THREAD.with_borrow_mut(|client| {
            log::debug!(target: #func_str, "[#{}]", client.id);
            let ClientThread { channel_sender, channel_receiver, .. } = client;

            #[cfg(feature = "phos")]
            client.phos_agent.block_until_ready(channel_sender);

            let proc_id: i32 = #proc_id;
            let mut #result_name: #result_ty = Default::default();

            #( #client_before_send )*

            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send {}: {}", "proc_id", e),
            }
            #( #send_statements )*
            #shadow_desc_send

            #( #client_extra_send )*

            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {}", e),
            }

            #shadow_desc_return
            #async_api_return

            #( #recv_statements )*
            match #result_name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", stringify!(#result_name), e),
            }
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", "timestamp", e),
            }
            if #result_name.is_error() {
                log::error!(
                    target: #func_str,
                    "[#{}] returned error: {:?}\n{}",
                    client.id,
                    #result_name,
                    std::backtrace::Backtrace::force_capture(),
                );
            }
            #( #client_after_recv )*

            #[cfg(feature = "phos")]
            client.phos_agent.block_until_ready(channel_sender);

            return #result_name;
        })
        }
    };

    gen_fn.into()
}

#[proc_macro_attribute]
pub fn use_thread_local(args: TokenStream, input: TokenStream) -> TokenStream {
    use_thread_local::use_thread_local(args.into(), input.into()).into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for server usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate execution functions for the server dispatcher.
///
/// ### Example
/// We have a function `cudaSetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a helper function for the server dispatcher, we can write:
///
/// ```ignore
/// #[cuda_hook_exe(proc_id = 1)]
/// fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// This invocation generates a function `cudaSetDeviceExe` with `channel_sender` and `channel_receiver` for communication.
/// `cudaSetDeviceExe` will be called by the server dispatcher to execute the native `cudaSetDevice` function and send the result back to the client.
#[proc_macro_attribute]
pub fn cuda_hook_exe(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    let (func, result, params) = (input.func, input.result, input.params);
    let func_str = func.to_string();
    let func_exe = format_ident!("{}Exe", func);

    // definition statements
    let def_statements = params.iter().map(|param| {
        if param.mode != ElementMode::Output {
            return Default::default();
        }
        let name = &param.name;
        let ptr_ident = param.get_exe_ptr_ident();
        let ty = &param.ty;
        let Type::Ptr(ptr) = ty else { panic!() };
        let ty = ptr.elem.as_ref();
        match &param.pass_by {
            PassBy::InputValue | PassBy::InputCStr => unreachable!(),
            PassBy::SinglePtr => quote_spanned! {name.span()=>
                let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                let #ptr_ident = #name.as_mut_ptr();
            },
            PassBy::ArrayPtr { len, cap } => {
                let cap = cap.as_ref().unwrap_or(len);
                let cap_ident = format_ident!("{}_cap", name);
                let mut tokens = define_usize_from(&cap_ident, cap);
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    let mut #name = Box::<[#ty]>::new_uninit_slice(#cap_ident);
                    let #ptr_ident = std::mem::MaybeUninit::slice_as_mut_ptr(&mut #name);
                };
                tokens
            }
        }
    });
    let assume_init = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            let name_str = name.to_string();
            quote_spanned! {name.span()=>
                let #name = unsafe { #name.assume_init() };
                log::trace!(target: #func_str, "[#{}] (output) {} = {:?}", server.id, #name_str, #name);
            }
        });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            let name_str = name.to_string();
            let recv_single = |ty| {
                quote_spanned! {name.span()=>
                    let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                    match #name.recv(channel_receiver) {
                        Ok(()) => {}
                        Err(e) => panic!("failed to receive {}: {}", #name_str, e),
                    }
                    let #name = unsafe { #name.assume_init() };
                    log::trace!(target: #func_str, "[#{}] (input) {} = {:?}", server.id, #name_str, #name);
                }
            };
            match &param.pass_by {
                PassBy::InputValue => recv_single(&param.ty),
                PassBy::SinglePtr => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    let mut tokens = recv_single(ptr.elem.as_ref());
                    let span = name.span();
                    let ptr_ident = param.get_exe_ptr_ident();
                    quote_each_token_spanned! {tokens span
                        let #ptr_ident = (&raw const #name).cast_mut();
                    }
                    tokens
                }
                PassBy::ArrayPtr { .. } => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    let ty = ptr.elem.as_ref();
                    let ptr_ident = param.get_exe_ptr_ident();
                    quote_spanned! {name.span()=>
                        let #name = match recv_slice::<#ty, _>(channel_receiver) {
                            Ok(slice) => slice,
                            Err(e) => panic!("failed to receive {}: {}", #name_str, e),
                        };
                        log::trace!(target: #func_str, "[#{}] (input) {} = {:p}", server.id, #name_str, #name.as_ptr());
                        let #ptr_ident = #name.as_ptr().cast_mut();
                    }
                }
                PassBy::InputCStr => {
                    let ptr_ident = param.get_exe_ptr_ident();
                    quote_spanned! {name.span()=>
                        let #name = match recv_slice(channel_receiver) {
                            Ok(slice) => {
                                std::ffi::CString::from_vec_with_nul(slice.into_vec()).unwrap()
                            }
                            Err(e) => panic!("failed to receive {}: {}", #name_str, e),
                        };
                        log::trace!(target: #func_str, "[#{}] (input) {} = {:?}", server.id, #name_str, #name);
                        let #ptr_ident = #name.as_ptr();
                    }
                }
            }
        });

    let is_destroy = func.to_string().contains("Destroy");

    // get resource when SR
    let get_resource_statements = params
        .iter()
        .filter(|param| {
            let ty = &param.ty;
            param.mode == ElementMode::Input && is_shadow_desc_type(ty)
        })
        .map(|param| {
            let name = &param.name;
            let ty = &param.ty;
            if is_destroy {
                assert_eq!(params.len(), 1);
                quote! {
                    let #name = if server.opt_shadow_desc {
                        server.resources.remove(&(#name as usize)).unwrap() as #ty
                    } else {
                        #name
                    };
                }
            } else {
                quote! {
                    let #name = if server.opt_shadow_desc {
                        *server.resources.get(&(#name as usize)).unwrap() as #ty
                    } else {
                        #name
                    };
                }
            }
        });

    // execution statement
    let result_name = &result.name;
    let (exec_statement, phos_fn) = {
        let result_ty = &result.ty;
        let params: Box<_> = params
            .iter()
            .filter(|param| matches!(param.mode, ElementMode::Input | ElementMode::Output))
            .collect();
        let exec_args = params.iter().map(|param| {
            let name = &param.name;
            let arg = if let PassBy::InputValue = param.pass_by {
                name
            } else {
                &param.get_exe_ptr_ident()
            };
            if param.is_void_ptr || is_hacked_type(&param.ty) {
                quote_spanned!(name.span()=> (#arg).cast())
            } else {
                quote!(#arg)
            }
        });
        let phos_exec_args = exec_args.clone();
        let phos_process_args = params.iter().map(|Element { name, .. }| {
            quote_spanned![name.span()=>
                &raw const #name as usize,
                size_of_val(&#name),
            ]
        });
        let proc_id = &input.proc_id;
        let phos_func_name = format_ident!("phos_{}", func);
        let phos_params = params.iter().map(|Element { name, ty, .. }| quote! { #name: #ty });
        let phos_fn = quote! {
            #[cfg(feature = "phos")]
            fn #phos_func_name(
                pos_cuda_ws: &crate::phos::POSWorkspace_CUDA,
                #(#phos_params),*
            ) -> #result_ty {
                #result_ty::from_i32(pos_cuda_ws.pos_process(
                    #proc_id,
                    0u64,
                    &[#(#phos_process_args)*],
                )).expect("Illegal result ID")
            }
        };
        let func = input.parent.as_ref().unwrap_or(&func);
        let exec_statement = quote! {
            #[cfg(not(feature = "phos"))]
            let #result_name: #result_ty = unsafe { #func(#(#exec_args),*) };
            #[cfg(feature = "phos")]
            let #result_name: #result_ty = #phos_func_name(
                &server.pos_cuda_ws,
                #(#phos_exec_args),*
            );
        };
        (exec_statement, phos_fn)
    };

    let exec_statement = if !input.injections.server_execution.is_empty() {
        let mut tokens = proc_macro2::TokenStream::new();
        tokens.append_all(&input.injections.server_execution);
        tokens
    } else {
        exec_statement
    };

    // send result
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            let name_str = name.to_string();
            let send = match &param.pass_by {
                PassBy::InputValue | PassBy::InputCStr => unreachable!(),
                PassBy::SinglePtr => quote! { #name.send(channel_sender) },
                PassBy::ArrayPtr { len, .. } => {
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = define_usize_from(&len_ident, len);
                    let span = name.span();
                    quote_each_token_spanned! {tokens span
                        send_slice(&#name[..#len_ident], channel_sender)
                    };
                    tokens
                }
            };
            quote_spanned! {name.span()=>
                match { #send } {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", #name_str, e),
                }
            }
        });

    let (shadow_desc_recv, shadow_desc_return) = if input.is_create_shadow_desc {
        let name = &params[0].name;
        let name_str = name.to_string();
        let shadow_desc_recv = quote_spanned! {name.span()=>
            let mut resource_idx = 0usize;
            if server.opt_shadow_desc {
                match resource_idx.recv(channel_receiver) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to receive {}: {}", #name_str, e),
                }
            }
        };
        let shadow_desc_return = quote_spanned! {name.span()=>
            if server.opt_shadow_desc {
                server.resources.insert(resource_idx, #name as usize);
                return;
            }
        };
        (shadow_desc_recv, shadow_desc_return)
    } else {
        Default::default()
    };

    let async_api_return = if let Some(true) = input.is_async_api {
        quote! {
            if server.opt_async_api {
                return;
            }
        }
    } else {
        Default::default()
    };

    let server_extra_recv = input.injections.server_extra_recv.iter();
    let server_before_execution = input.injections.server_before_execution.iter();
    let server_after_send = input.injections.server_after_send.iter();

    let gen_fn = quote! {
        pub fn #func_exe<C: CommChannel>(server: &mut ServerWorker<C>) {
            let ServerWorker { channel_sender, channel_receiver, .. } = server;
            log::debug!(target: #func_str, "[#{}]", server.id);
            #( #recv_statements )*
            #( #get_resource_statements )*
            #shadow_desc_recv
            #( #server_extra_recv )*
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {:?}", "timestamp", e),
            }
            #( #def_statements )*
            #( #server_before_execution )*
            #exec_statement
            #( #assume_init )*

            if #result_name.is_error() {
                log::error!(target: #func_str, "[#{}] returned error: {:?}", server.id, #result_name);
            }

            #shadow_desc_return
            #async_api_return
            #( #send_statements )*
            match #result_name.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send {}: {}", stringify!(#result_name), e),
            }
            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {}", e),
            }
            #( #server_after_send )*
        }

        #phos_fn
    };

    gen_fn.into()
}
