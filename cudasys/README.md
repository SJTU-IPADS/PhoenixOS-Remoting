# CUDASYS

This is Rust bindings for the CUDA toolkit APIs. The bindings are generated using [bindgen](https://github.com/rust-lang/rust-bindgen).

```bash
.
├── build.rs
├── Cargo.toml
├── include
│   └── cuda_wrapper.h
├── README.md
└── src
    ├── bindings
    │   ├── funcs
    │   │   ├── cuda.rs
    │   │   ├── cudart.rs
    │   │   └── nvml.rs
    │   ├── README.md
    │   └── types
    │       ├── cuda.rs
    │       ├── cudart.rs
    │       └── nvml.rs
    ├── cuda.rs
    ├── cudart.rs
    ├── lib.rs
    ├── nvml.rs
    └── types
        ├── cuda.rs
        ├── cudart.rs
        ├── mod.rs
        └── nvml.rs
```

## Build Script

The code is in `build.rs`. Logic:

- Use `bindgen` to generate Rust bindings for the *raw* CUDA toolkit APIs.
- Decorate the generated bindings for usability like being `Transportable` in `network`.
- Split the bindings into separate parts (`types` and `funcs`) for client/server separating usage, `types` part is in `src/types`, `funcs` part is in `src/`, and raw bindings code are left in `src/bindings`.

## Usage

- For client part, only need types, so it `use cudasys::types::cuda::*`, etc.
- For server part, need both types and funcs, so it `use cudasys::cuda::*`, etc.
