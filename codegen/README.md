# Codegen for (most) remoting functions

The crate is a procedural macro lib for generating:

- the remoting functions in both client and server side.
- the trait for transferring the user-defined types between the client and server.

## Modules

- `cuda_hook_hijack`: the macro for generating the hijack functions for client intercepting application calls.
- `cuda_hook_exe`: the macro for generating the execution functions for server dispatching application calls.

Minor tips: to check the expanded macros, we can use the following:

```bash
cargo +nightly install cargo-expand

cargo expand -p client hijack::cudnn_hijack::cudnnCreate
cargo expand -p server --lib dispatcher::cudnn_exe::cudnnCreateExe
```

or use the [rust-analyzer: Expand macro recursively at caret](https://rust-analyzer.github.io/manual.html#expand-macro-recursively) action.
