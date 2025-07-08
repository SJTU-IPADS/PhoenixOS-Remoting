use bindgen::callbacks::{DeriveInfo, ParseCallbacks};
use glob::glob;
use regex::Regex;
use std::{
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        let split_char = ":";
        path.split(split_char).map(PathBuf::from).collect()
    } else {
        vec![]
    }
}

// output (candidates, valid_path)
pub fn find_cuda() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut candidates = read_env();
    candidates.push(PathBuf::from(".")); // bindgen wrapper headers
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/usr"));
    for e in glob("/usr/local/cuda-*").unwrap() {
        if let Ok(path) = e {
            candidates.push(path)
        }
    }

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = base.join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
        }
        // cudnn
        let cudnn_lib = base.join("include/x86_64-linux-gnu");
        if cudnn_lib.is_dir() {
            valid_paths.push(cudnn_lib);
        }
    }
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    (candidates, valid_paths)
}

#[derive(Debug)]
struct DeriveCallback;

impl ParseCallbacks for DeriveCallback {
    fn add_derives(&self, info: &DeriveInfo<'_>) -> Vec<String> {
        if matches!(
            info.name,
            "cudaError_enum"
                | "cudaError"
                | "nvmlReturn_enum"
                | "cudnnStatus_t"
                | "cublasStatus_t"
                | "nvrtcResult"
                | "ncclResult_t"
        ) {
            vec![
                "num_derive::FromPrimitive".to_owned(),
                "codegen::Transportable".to_owned(),
            ]
        } else {
            vec!["codegen::Transportable".to_owned()]
        }
    }
}

/// Read the file, split it into two parts: one with the types and the other with the functions.
/// Remove the original file.
fn split(content: &str, types_file: &Path, funcs_file: &Path) {
    // regex to match `unsafe extern "C" { ... }`
    let re = Regex::new(r#"(?s)unsafe extern "C" \{.*?\}\n"#).unwrap();

    // Extract all blocks that match the regex
    let funcs: Vec<_> = re.find_iter(content).map(|mat| mat.as_str()).collect();
    let types = re.replace_all(content, "");

    // Write the types and functions to separate files.
    if let Some(parent) = types_file.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("Failed to create directory {parent:?}: {e}"));
    }
    let mut types_file = File::create(types_file).unwrap_or_else(|e| panic!("Failed to create file {types_file:?}: {e}"));
    writeln!(types_file, "{}", types).expect("Failed to write types");

    if let Some(parent) = funcs_file.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("Failed to create directory {parent:?}: {e}"));
    }
    let mut funcs_file = File::create(funcs_file).unwrap_or_else(|e| panic!("Failed to create file {funcs_file:?}: {e}"));

    for f in funcs {
        write!(funcs_file, "{f}").expect("Failed to write function");
    }
}

fn bind_gen(
    paths: &[PathBuf],
    library_header: &str,
    output: &str,
    allowlist_types: &[&str],
    allowlist_vars: &[&str],
    allowlist_funcs: &[&str],
    link_lib: &str,
) {
    println!("cargo:rustc-link-lib={link_lib}");

    // find the library header path
    let mut header_path = None;
    for path in paths {
        let mut header = path.clone();
        header.push("include");
        header.push(library_header);
        if header.is_file() {
            header_path = Some(header);
            break;
        }
    }
    let header_path = header_path.expect("Could not find CUDA header file");

    // The bindgen::Builder is the main entry point to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut bindings = bindgen::Builder::default()
        .layout_tests(false)
        .formatter(bindgen::Formatter::Prettyplease)
        // Add CargoCallbacks so build.rs is rerun on header changes
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // The input header we would like to generate bindings for.
        .header(header_path.to_str().unwrap());

    // Whitelist types, functions, and variables
    for ty in allowlist_types {
        bindings = bindings.allowlist_type(ty);
    }
    for var in allowlist_vars {
        bindings = bindings.allowlist_var(var);
    }
    for func in allowlist_funcs {
        bindings = bindings.allowlist_function(func);
    }

    bindings = bindings
        // Set the default enum style to be more Rust-like
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        // Disable documentation comments from being generated
        .generate_comments(false)
        .generate_cstr(true)
        .parse_callbacks(Box::new(DeriveCallback))
        .opaque_type("FILE");

    // Add include paths
    for path in paths {
        bindings = bindings.clang_arg(format!("-I{}/include", path.to_str().unwrap()));
    }

    let bindings = bindings
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    let root = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Split the file into two parts: one with the types and the other with the functions.
    let types_file = PathBuf::from(format!("{root}/src/bindings/types/{output}.rs"));
    let funcs_file = PathBuf::from(format!("{root}/src/bindings/funcs/{output}.rs"));
    split(&bindings.to_string(), &types_file, &funcs_file);
}

fn main() {
    let (cuda_paths, cuda_libs) = find_cuda();

    // Tell rustc to link the CUDA library.
    for path in &cuda_libs {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    eprintln!("CUDA paths: {:?}", cuda_paths);
    eprintln!("CUDA libs: {:?}", cuda_libs);

    // Use bindgen to automatically generate the FFI (in `src/bindings`).
    bind_gen(
        &cuda_paths,
        "cudart_wrapper.hpp",
        "cudart",
        &["^cuda.*", "^surfaceReference", "^textureReference"],
        &["^cuda.*", "CUDART_VERSION"],
        &["^cuda.*", "__cuda[A-Za-z]+"],
        "dylib=cudart",
    );

    bind_gen(
        &cuda_paths,
        "cuda_wrapper.h",
        "cuda",
        &[
            "^CU.*",
            "^cuuint(32|64)_t",
            "^cudaError_enum",
            "^cu.*Complex$",
            "^cuda.*",
            "^libraryPropertyType.*",
        ],
        &["^CU.*"],
        &["^cu.*"],
        "dylib=cuda",
    );

    bind_gen(
        &cuda_paths,
        "nvml.h",
        "nvml",
        &[],
        &[],
        &[],
        "dylib=nvidia-ml",
    );

    bind_gen(
        &cuda_paths,
        "cudnn.h",
        "cudnn",
        &["^cudnn.*", "^CUDNN.*"],
        &["^CUDNN.*", "^cudnn.*"],
        &["^cudnn.*"],
        "dylib=cudnn",
    );

    bind_gen(
        &cuda_paths,
        "cublas.h",
        "cublas",
        &["^cublas.*", "^CUBLAS.*"],
        &["^CUBLAS.*", "^cublas.*"],
        &["^cublas.*"],
        "dylib=cublas",
    );

    bind_gen(
        &cuda_paths,
        "cublasLt.h",
        "cublasLt",
        &["cublasLt.*"],
        &["cublasLt.*"],
        &["cublasLt.*"],
        "dylib=cublasLt",
    );

    bind_gen(
        &cuda_paths,
        "nvrtc.h",
        "nvrtc",
        &["nvrtc.*"],
        &["nvrtc.*"],
        &["nvrtc.*"],
        "dylib=nvrtc",
    );

    bind_gen(
        &cuda_paths,
        "nccl.h",
        "nccl",
        &["nccl.*"],
        &["nccl.*", "NCCL.*"],
        &["nccl.*"],
        "dylib=nccl",
    );
}
