use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use hookdef::{is_hacked_type, CustomHookAttrs, HookAttrs};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned as _;
use syn::{
    parse_quote, Attribute, Block, FnArg, ForeignItem, Ident, Item, ItemFn, Meta, Signature, Token,
    Type, UseTree, Visibility,
};

struct Hook {
    proc_id: i32,
    name: String,
    is_custom: bool,
}

pub fn generate_impls(
    hooks_path: &str,
    bindings_dir: &str,
    output_dir: &str,
    output_suffix: &str,
    unimplement_suffix: Option<&str>,
    cuda_version: u8,
) {
    let target_attr = match unimplement_suffix {
        Some(_) => "cuda_hook_hijack",
        None => "cuda_hook_exe",
    };
    let mod_file = fs::read_to_string(Path::new(output_dir).join("mod.rs"))
        .expect("failed to read `mod.rs` file");
    let mut all_hooks = Vec::new();
    for (ref module, mut bindings) in parse_bindings_dir(bindings_dir) {
        let hooks_path = hooks_path.replace("{}", module);
        let comment = format!("// Generated from {hooks_path} under CUDA {cuda_version}\n\n");
        let imports = get_imports(module, unimplement_suffix.is_some());
        let output_mod = [module, output_suffix].concat();
        let hooks = convert_hooks(
            &hooks_path,
            &format!("{output_dir}/{output_mod}.rs"),
            &comment,
            &imports,
            &mut bindings,
            target_attr,
            cuda_version,
        );
        if unimplement_suffix.is_some()
            && !hooks.is_empty()
            && !mod_file.contains(&format!("mod {output_mod};"))
        {
            println!("cargo:warning=`mod {output_mod};` is missing from `mod.rs`");
        }
        if unimplement_suffix.is_none() {
            for Hook { proc_id, name, is_custom } in hooks {
                if is_custom {
                    all_hooks.push((proc_id, format!("{output_mod}_custom::{name}Exe")));
                } else {
                    all_hooks.push((proc_id, format!("{output_mod}::{name}Exe")));
                }
            }
        }
        if let Some(unimplement_suffix) = unimplement_suffix {
            let unimplement_mod = [module, unimplement_suffix].concat();
            let count = bindings.len();
            generate_bare_hooks(
                &format!("{output_dir}/{unimplement_mod}.rs"),
                &comment,
                vec![parse_quote! { #![allow(unused_variables)] }],
                &imports[1..],
                bindings,
                |sig| {
                    let name = sig.ident.to_string();
                    Some(parse_quote!({ unimplemented!(#name) }))
                },
            );
            if count != 0 && !mod_file.contains(&format!("mod {unimplement_mod};")) {
                println!("cargo:warning=`mod {unimplement_mod};` is missing from `mod.rs`");
            }
        }
    }
    if unimplement_suffix.is_none() {
        all_hooks.sort_by_key(|&(proc_id, _)| proc_id);
        let mut output_file = fs::File::create(Path::new(output_dir).join("mod_exe.rs")).unwrap();
        write!(
            &mut output_file,
            "// Generated from {} under CUDA {cuda_version}\n\n",
            hooks_path.replace("{}", "*"),
        )
        .unwrap();
        write!(
            &mut output_file,
            "macro_rules! dispatcher_match {{\n($proc_id:ident, $other:ident => $err:tt) => {{\nmatch $proc_id {{\n",
        )
        .unwrap();
        for (proc_id, func) in all_hooks {
            if proc_id < 0 {
                writeln!(
                    &mut output_file,
                    "{proc_id} => compile_error!(\"{func} has invalid proc_id\"),"
                )
                .unwrap();
            } else {
                writeln!(&mut output_file, "{proc_id} => {func},").unwrap();
            }
        }
        write!(&mut output_file, "$other => $err\n}}\n}}\n}}\n").unwrap();
    }
}

pub fn generate_passthrough(
    bindings_dir: &str,
    output_dir: &str,
    body: fn(&Signature) -> Option<Box<Block>>,
) {
    let comment = format!("// Generated from {bindings_dir}\n\n");
    let mut mod_file = fs::File::create(format!("{output_dir}/mod_passthrough.rs")).unwrap();
    mod_file.write_all(comment.as_bytes()).unwrap();
    for (module, bindings) in parse_bindings_dir(bindings_dir) {
        writeln!(&mut mod_file, "mod {module}_passthrough;").unwrap();
        generate_bare_hooks(
            &format!("{output_dir}/{module}_passthrough.rs"),
            &comment,
            Vec::new(),
            &get_imports(&module, true)[1..],
            bindings,
            body,
        );
    }
}

fn parse_bindings_dir(bindings_dir: &str) -> Vec<(String, BTreeMap<String, Signature>)> {
    fs::read_dir(bindings_dir)
        .expect("failed to read bindings directory")
        .map(|entry| {
            let path = entry.unwrap().path();
            let (module, extension) =
                path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap();
            assert_eq!(extension, "rs");
            (module.to_string(), parse_bindings(&path))
        })
        .collect()
}

fn parse_bindings(path: &Path) -> BTreeMap<String, Signature> {
    eprintln!("parsing {path:?}");
    let bindings = fs::read_to_string(path).unwrap();
    let file = syn::parse_file(&bindings).unwrap();
    let mut result = BTreeMap::new();
    for item in file.items {
        let Item::ForeignMod(foreign) = item else { panic!() };
        for item in foreign.items {
            let ForeignItem::Fn(func) = item else { panic!() };
            let name = func.sig.ident.to_string();
            result.insert(name, func.sig);
        }
    }
    result
}

fn get_imports(module: &str, is_client: bool) -> [Item; 2] {
    let module = Ident::new(module, Span::call_site());
    match is_client {
        true => [
            Item::Use(parse_quote! { use super::*; }),
            Item::Use(parse_quote! { use cudasys::types::#module::*; }),
        ],
        false => [
            Item::Use(parse_quote! { use super::*; }),
            Item::Use(parse_quote! { use cudasys::#module::*; }),
        ],
    }
}

fn convert_hooks(
    hooks_path: &str,
    output_path: &str,
    comment: &str,
    imports: &[Item],
    bindings: &mut BTreeMap<String, Signature>,
    target_attr: &str,
    cuda_version: u8,
) -> Vec<Hook> {
    eprintln!("parsing {hooks_path:?}");
    let hooks = fs::read_to_string(hooks_path).unwrap();
    let file = syn::parse_file(&hooks).unwrap();
    let mut output = imports.to_vec();
    let mut hooks = Vec::new();

    for item in file.items {
        match item {
            Item::Use(ref use_item) => {
                if let UseTree::Path(path) = &use_item.tree {
                    if path.ident == "std" {
                        output.push(item);
                    }
                }
            }
            Item::Type(mut item) => {
                item.attrs.clear();
                if let Type::Path(ty) = item.ty.as_mut() {
                    let ident = &mut ty.path.segments[0].ident;
                    if ident == "crate" {
                        *ident = Ident::new("cudasys", ident.span());
                    }
                }
                output.push(Item::Type(item));
            }
            Item::Fn(mut func) => {
                if check_sig_replace_attr(
                    &mut func.attrs,
                    &func.sig,
                    bindings,
                    target_attr,
                    cuda_version,
                    &mut output,
                    &mut hooks,
                ) {
                    output.push(Item::Fn(func));
                }
            }
            Item::Verbatim(tokens) => {
                let mut func: HookDef = syn::parse2(tokens).unwrap();
                if check_sig_replace_attr(
                    &mut func.attrs,
                    &func.sig,
                    bindings,
                    target_attr,
                    cuda_version,
                    &mut output,
                    &mut hooks,
                ) {
                    output.push(Item::Verbatim(func.to_token_stream()));
                }
            }
            _ => {
                output.push(Item::Macro(parse_quote! {
                    compile_error!("unexpected item below");
                }));
                output.push(item);
            }
        }
    }

    let output = prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: Default::default(),
        items: output,
    });

    let mut output_file = fs::File::create(output_path).unwrap();
    output_file.write_all(comment.as_bytes()).unwrap();
    output_file.write_all(output.as_bytes()).unwrap();
    hooks
}

/// Returns true if the function should be emitted.
fn check_sig_replace_attr(
    attrs: &mut [Attribute],
    sig: &Signature,
    bindings: &mut BTreeMap<String, Signature>,
    target_attr: &str,
    cuda_version: u8,
    output: &mut Vec<Item>,
    hooks: &mut Vec<Hook>,
) -> bool {
    let err_item = || {
        Item::Macro(parse_quote! {
            compile_error!("unrecognized hook definition, expected 1 attribute");
        })
    };
    if attrs.len() != 1 {
        output.push(err_item());
        return true;
    }

    let attr = &mut attrs[0];
    match attr.path().segments.last().unwrap().ident.to_string().as_str() {
        "cuda_hook" => {}
        "cuda_custom_hook" => {
            bindings.remove(&sig.ident.to_string());
            match CustomHookAttrs::from_attr(attr) {
                Ok(attrs) => {
                    if let Some(proc_id) = attrs.proc_id {
                        hooks.push(Hook {
                            proc_id: proc_id.base10_parse().unwrap_or(-1),
                            name: sig.ident.to_string(),
                            is_custom: true,
                        });
                    }
                }
                Err(err) => {
                    output.push(Item::Macro(syn::parse2(err.to_compile_error()).unwrap()));
                    return true;
                }
            }
            eprintln!("skipped custom hook `{}`", sig.ident);
            return false;
        }
        _ => {
            output.push(err_item());
            return true;
        }
    }
    let is_internal = match HookAttrs::from_attr(attr) {
        Ok(attrs) => {
            if cuda_version < attrs.min_cuda_version || attrs.max_cuda_version < cuda_version {
                println!(
                    "cargo:warning=not emitting hook for `{}` because it's incompatible with CUDA {}",
                    sig.ident, cuda_version
                );
                return false;
            }
            hooks.push(Hook {
                proc_id: attrs.proc_id.base10_parse().unwrap_or(-1),
                name: sig.ident.to_string(),
                is_custom: false,
            });
            attrs.parent.is_some()
        }
        Err(err) => {
            output.push(Item::Macro(syn::parse2(err.to_compile_error()).unwrap()));
            return true;
        }
    };
    match attr.meta {
        Meta::List(ref mut meta) => {
            meta.path = Ident::new(target_attr, meta.path.span()).into();
        }
        _ => unreachable!(),
    }

    if is_internal {
        return true;
    }

    let Some(mut binding) = bindings.remove(&sig.ident.to_string()) else {
        output.push(Item::Macro(parse_quote! {
            compile_error!("binding not found for the function below");
        }));
        return true;
    };

    if !is_sig_equal_ignore_attr(sig, &binding) {
        output.push(Item::Macro(parse_quote! {
            compile_error!("function signature mismatch");
        }));
        binding.ident = format_ident!("_binding__{}", binding.ident);
        output.push(Item::Fn(ItemFn {
            attrs: vec![parse_quote! { #[expect(unused_variables)] }],
            vis: Visibility::Inherited,
            sig: binding,
            block: parse_quote!({ unimplemented!() }),
        }));
        return true;
    };

    true
}

fn is_sig_equal_ignore_attr(hook: &Signature, binding: &Signature) -> bool {
    if hook.constness != binding.constness
        || hook.asyncness != binding.asyncness
        || hook.unsafety != binding.unsafety
        || hook.abi != binding.abi
        || hook.ident != binding.ident
        || hook.generics != binding.generics
        || hook.variadic != binding.variadic
        || hook.output != binding.output
    {
        return false;
    }

    if hook.inputs.len() != binding.inputs.len() {
        return false;
    }

    for pair in hook.inputs.iter().zip(binding.inputs.iter()) {
        let (FnArg::Typed(hook), FnArg::Typed(binding)) = pair else {
            return false;
        };
        if hook.pat != binding.pat {
            return false;
        }
        if !is_type_equal_ignore_path(&hook.ty, &binding.ty) && !is_hacked_type(&hook.ty) {
            return false;
        }
    }
    true
}

fn is_type_equal_ignore_path(hook: &Type, binding: &Type) -> bool {
    match (hook, binding) {
        (Type::Ptr(hook), Type::Ptr(binding)) => {
            hook.const_token == binding.const_token
                && hook.mutability == binding.mutability
                && is_type_equal_ignore_path(&hook.elem, &binding.elem)
        }
        (Type::Path(hook), Type::Path(binding)) => {
            hook.path.segments.last() == binding.path.segments.last()
        }
        _ => panic!(
            "unsupported type comparison: {} vs {}",
            hook.to_token_stream(),
            binding.to_token_stream()
        ),
    }
}

fn generate_bare_hooks(
    output_path: &str,
    comment: &str,
    file_attrs: Vec<Attribute>,
    imports: &[Item],
    bindings: BTreeMap<String, Signature>,
    body: fn(&Signature) -> Option<Box<Block>>,
) {
    let mut items = imports.to_vec();
    let attrs = vec![parse_quote! { #[no_mangle] }];
    let abi = Some(parse_quote!(extern "C"));
    for mut sig in bindings.into_values() {
        let Some(block) = body(&sig) else { continue };
        sig.abi = abi.clone();
        items.push(Item::Fn(ItemFn {
            attrs: attrs.clone(),
            vis: Visibility::Inherited,
            sig,
            block,
        }));
    }
    let output = prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: file_attrs,
        items,
    });
    let mut file = fs::File::create(output_path).unwrap();
    file.write_all(comment.as_bytes()).unwrap();
    file.write_all(output.as_bytes()).unwrap();
}

struct HookDef {
    attrs: Vec<Attribute>,
    sig: Signature,
    semi: Token![;],
}

impl Parse for HookDef {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            attrs: input.call(Attribute::parse_outer)?,
            sig: input.parse()?,
            semi: input.parse()?,
        })
    }
}

impl ToTokens for HookDef {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.attrs[0].to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
