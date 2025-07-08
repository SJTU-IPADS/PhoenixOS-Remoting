#![feature(maybe_uninit_slice)]

mod dispatcher;

#[cfg(feature = "phos")]
mod phos;

use cudasys::{
    cuda::CUmodule,
    cudart::{cudaError_t, cudaGetDeviceCount},
};
use dispatcher::dispatch;

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{tcp, Channel, CommChannel, CommChannelError, NetworkConfig, Transportable};

use log::{error, info};

use std::collections::BTreeMap;

struct ServerWorker<C> {
    pub id: i32,
    pub channel_sender: C,
    pub channel_receiver: C,
    pub modules: Vec<CUmodule>,
    pub resources: BTreeMap<usize, usize>,
    opt_async_api: bool,
    opt_shadow_desc: bool,
    #[cfg(feature = "phos")]
    pub pos_cuda_ws: phos::POSWorkspace_CUDA,
}

impl<C> Drop for ServerWorker<C> {
    fn drop(&mut self) {
        #[cfg(not(feature = "phos"))]
        for module in &self.modules {
            unsafe {
                cudasys::cuda::cuModuleUnload(*module);
            }
        }
    }
}

fn create_buffer(
    config: &NetworkConfig,
    id: i32,
    barrier: Option<std::sync::Arc<std::sync::Barrier>>,
) -> (Channel, Channel) {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match config.comm_type.as_str() {
        "shm" => {
            let (receiver, sender) = SHMChannel::new_server_with_id(config, id).unwrap();
            barrier.unwrap().wait();
            if config.emulator {
                return (
                    Channel::new(Box::new(EmulatorChannel::new(sender, config))),
                    Channel::new(Box::new(EmulatorChannel::new(receiver, config))),
                );
            }
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        "tcp" => {
            let (receiver, sender) = tcp::new_server(config, id, &barrier.unwrap()).unwrap();
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            assert!(barrier.is_none());
            let (receiver, sender) = RDMAChannel::new_server(config, id);
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        &_ => panic!("Unsupported communication type in config"),
    }
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    if let Ok(()) = proc_id.recv(channel_receiver) {
        Ok(proc_id)
    } else {
        Err(CommChannelError::IoError)
    }
}

pub fn launch_server(
    config: &NetworkConfig,
    id: i32,
    barrier: Option<std::sync::Arc<std::sync::Barrier>>,
    is_main_thread: bool,
) {
    // PhOS workspace must be created before client initialization barrier
    #[cfg(feature = "phos")]
    let pos_cuda_ws = phos::POSWorkspace_CUDA::new();

    let (channel_sender, channel_receiver) = create_buffer(config, id, barrier);
    info!(
        "[{}:{}] {} buffer created",
        std::file!(),
        std::line!(),
        config.comm_type
    );
    let mut max_devices = 0;
    if let cudaError_t::cudaSuccess =
        unsafe { cudaGetDeviceCount(&mut max_devices as *mut ::std::os::raw::c_int) }
    {
        info!(
            "[{}:{}] found {} cuda devices",
            std::file!(),
            std::line!(),
            max_devices
        );
    } else {
        error!(
            "[{}:{}] failed to find cuda devices",
            std::file!(),
            std::line!()
        );
        panic!();
    }

    let mut server = ServerWorker {
        id,
        channel_sender,
        channel_receiver,
        modules: Default::default(),
        resources: Default::default(),
        opt_async_api: config.opt_async_api,
        opt_shadow_desc: config.opt_shadow_desc,
        #[cfg(feature = "phos")]
        pos_cuda_ws,
    };

    loop {
        if let Ok(proc_id) = receive_request(&mut server.channel_receiver) {
            if proc_id == -1 {
                break;
            }
            #[cfg(feature = "phos")]
            if proc_id == -2 {
                let mut uuid = 0u64;
                uuid.recv(&mut server.channel_receiver).unwrap();
                server.pos_cuda_ws.stop(uuid);
                continue;
            }
            dispatch(proc_id, &mut server);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!("[{}:{}] server #{} terminated", std::file!(), std::line!(), server.id);

    if is_main_thread {
        std::process::exit(0);
    }
}
