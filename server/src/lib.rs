extern crate log;
mod cuda_lib;
mod dispatcher;

use cuda_lib::*;
extern crate codegen;
extern crate network;

use codegen::gen_exe;
use dispatcher::dispatch;
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    type_impl::{
        basic::MemPtr,
        cudart::{cudaError_t, cudaMemcpyKind, cudaStreamCaptureStatus, cudaStream_t},
    },
    CommChannel, CommChannelError, Transportable,
};

#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

fn create_buffer() -> (
    RingBuffer<SHMChannelBufferManager>,
    RingBuffer<SHMChannelBufferManager>,
) {
    let manager_sender = SHMChannelBufferManager::new_server(SHM_NAME_STOC, SHM_SIZE).unwrap();
    let manager_receiver = SHMChannelBufferManager::new_server(SHM_NAME_CTOS, SHM_SIZE).unwrap();
    (
        RingBuffer::new(manager_sender),
        RingBuffer::new(manager_receiver),
    )
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    if let Ok(()) = proc_id.recv(channel_receiver) {
        Ok(proc_id)
    } else {
        Err(CommChannelError::IoError)
    }
}

pub fn launch_server() {
    let (mut channel_sender, mut channel_receiver) = create_buffer();
    info!("[{}:{}] shm buffer created", std::file!(), std::line!());
    if let cudaError_t::cudaSuccess = unsafe { cudaDeviceSynchronize() } {
        info!(
            "[{}:{}] cuda driver initialized",
            std::file!(),
            std::line!()
        );
    } else {
        error!(
            "[{}:{}] failed to initialize cuda driver",
            std::file!(),
            std::line!()
        );
        panic!();
    }

    loop {
        if let Ok(proc_id) = receive_request(&mut channel_receiver) {
            dispatch(proc_id, &mut channel_sender, &mut channel_receiver);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!("[{}:{}] server terminated", std::file!(), std::line!());
}
