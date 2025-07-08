## Using Docker Compose

For any environment, copy the `compose.example.yaml` file to `compose.yaml` and update `container_name`. Then run:

```shell
cd environment/<env_name>
docker compose up -d
```

Then you can use VS Code to attach to the container, or just:

```shell
docker exec -it <container_name> bash
```

To stop the container and start it later:

```shell
docker stop <container_name>
docker start <container_name>
```

To remove the container:

```shell
cd environment/<env_name>
docker compose down
```

It is recommended to mount a `docker_root` folder as the home directory so you don't need to install Rust in every container. To get the basic configuration like `.bashrc`, run:

```shell
cp -r /etc/skel/. /path/to/docker_root
```

## Installing requirements

- Note that we need Rust **nightly** toolchain to build the project, which is not installed in the docker image.

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly
. "$HOME/.cargo/env"
```

- Version checklist
  - Rust: `nightly`
  - CMake: at least `3.22.1` (required to run some of the tests)
  - Clang: at least `6.0.0-1ubuntu2`

You need to install Clang manually:

```shell
apt update
apt install clang --no-install-recommends
```

You probably also need to upgrade NCCL:

```shell
apt install libnccl2 libnccl-dev
```

The `vllm` environment does not include cuDNN, so you need to install it manually:

```shell
apt install libcudnn9-dev-cuda-12
```

If you need RDMA support:

```shell
apt install librdmacm-dev ibverbs-utils --no-install-recommends
```

## Appendix

### Build the (legacy PyTorch 1.13 + CUDA 11) docker image

Please refer to the [link](https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf), and use the Dockerfile in this directory.
