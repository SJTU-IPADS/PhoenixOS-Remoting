#![expect(non_snake_case)]

mod cuda_hijack;
mod cuda_hijack_custom;
mod cuda_hijack_utils;
mod cuda_unimplement;
mod cudart_hijack;
mod cudart_hijack_custom;
mod cudart_register;
mod cudart_unimplement;
mod nvml_hijack;
mod nvml_unimplement;
mod cudnn_hijack;
mod cudnn_hijack_custom;
mod cudnn_unimplement;
mod cublas_hijack;
mod cublas_unimplement;
mod cublasLt_hijack;
mod cublasLt_unimplement;
mod nvrtc_hijack;
mod nvrtc_unimplement;
mod nccl_hijack;
mod nccl_unimplement;

#[expect(unused_imports)]
use codegen::{cuda_hook_hijack, use_thread_local};
use network::type_impl::{recv_slice_to, send_slice, MemPtr};
use network::{CommChannel, Transportable};

use crate::elf::{FatBinaryHeader, FatBinaryWrapper};
use crate::{ClientThread, FatBinaryHandle, HostPtr, CLIENT_THREAD, DRIVER_CACHE, RUNTIME_CACHE};
