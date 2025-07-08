# XPURemoting

This is a virtualization framework for CUDA API.  The developer can easily add customization to the execution CUDA API, e.g., execute it as an RPC at a remote machine. 

## Minimal demo

Consits of 4 parts:

- `server`
- `client`
- `network`
- `codegen`

## Getting started

- How to add customization to a CUDA API:  [implement_new_apis.md](docs/implement_new_apis.md) 



## Requirements

See [environment](environment/README.md).

## Build

```shell
cd /path/to/xpuremoting && cargo build
```

You should specify `default` feature in both `client/server`'s Cargo.toml to decide which communication methods will be compiled. For example, if you want to use RDMA communication method, you should add `"rdma"` into the `default` feature in `client/Cargo.toml` and `server/Cargo.toml`. 

## Config

You can use `config.toml` file to config communication type, buffer size, RDMA server listener socket and so on.

Since `cargo` will use cwd as running root folder, you should use absolute path for config file. The default path will be `/workspace/config.toml`. If you want a specific path, you can use the environment variable `NETWORK_CONFIG` to customize it. For example: `NETWORK_CONFIG=/workspace/config.toml cargo run server`.

### Emulator

The network emulator is a component used to simulate system performance under different network conditions. When enabled, it calculates network latency and make the message receiver busy-wait, thus simulating the network overhead caused by RTT and bandwidth. Real-world tests have shown that the emulator's performance deviates from the actual conditions by no more than 5%. You can configure the `rtt` and `bandwidth` settings in `config.toml`.

This feature is only available when using shared memory. To enable the emulator, you can simply set `emulator` to `true` in `config.toml`.

## Test

### Unit test

```shell
cargo test
```

### Integration test

Launch two terminals, one for server and the other for client.

- server side:

```shell
cargo run [--features rdma] server
```

You should use `features` to decide what communication methods will be compiled. SHM is always compiled.

- client side:

```shell
cd tests/cuda_api
mkdir -p build && cd build
cmake .. && make
cd ..
find -type f -executable -name "test_*" -exec ./startclient_debug.sh {} \;
```

P.S. Can use `RUST_LOG` environment to control the log level (default=debug).


### Application test

Please refer to [application-guild](./tests/apps/README.md) to run applicatons
