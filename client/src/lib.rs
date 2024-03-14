#[macro_use]
extern crate lazy_static;

extern crate network;
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    type_impl::{
        basic::MemPtr,
        cuda::CUresult,
        cudart::{
            cudaDeviceProp, cudaError_t, cudaMemcpyKind, cudaStreamCaptureStatus, cudaStream_t,
            dim3,
        },
        nvml::nvmlReturn_t,
    },
    CommChannel, Transportable,
};

extern crate codegen;
use codegen::gen_hijack;

pub mod cuda_hijack;
pub use cuda_hijack::*;

pub mod elf;
use elf::interfaces::{fat_header, kernel_info_t};
use elf::ElfController;

use std::sync::Mutex;

lazy_static! {
    static ref CHANNEL_SENDER: Mutex<RingBuffer<SHMChannelBufferManager>> = {
        let manager = SHMChannelBufferManager::new_client(SHM_NAME_CTOS, SHM_SIZE).unwrap();
        Mutex::new(RingBuffer::new(manager))
    };
    static ref CHANNEL_RECEIVER: Mutex<RingBuffer<SHMChannelBufferManager>> = {
        let manager = SHMChannelBufferManager::new_client(SHM_NAME_STOC, SHM_SIZE).unwrap();
        Mutex::new(RingBuffer::new(manager))
    };
    static ref ELF_CONTROLLER: ElfController = ElfController::new();
}
