#![cfg_attr(feature = "passthrough", expect(dead_code))]
#![feature(thread_local)]

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::ringbufferchannel::{EmulatorChannel, SHMChannel};
use network::{tcp, Channel, CommChannel, Transportable};

#[cfg(not(feature = "passthrough"))]
mod hijack;
#[cfg(feature = "passthrough")]
mod passthrough;

mod elf;
use elf::{FatBinaryHeader, KernelParamInfo};

mod dl;

#[cfg(feature = "phos")]
mod phos;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::c_char;
use std::io::{Read as _, Write as _};
use std::sync::RwLock;

use cudasys::types::cuda::{CUfunction, CUmodule};
type FatBinaryHandle = usize;
type HostPtr = usize;

struct ClientThread {
    id: i32,
    channel_sender: Channel,
    channel_receiver: Channel,
    resource_idx: usize,
    cuda_device: Option<std::ffi::c_int>,
    cuda_device_init: bool,
    #[cfg(feature = "phos")]
    phos_agent: phos::POSAgent,
    opt_async_api: bool,
    opt_shadow_desc: bool,
    opt_local: bool,
}

impl ClientThread {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    fn new() -> Self {
        log::info!("[{}:{}] client init", std::file!(), std::line!());
        for (i, arg) in std::env::args().enumerate() {
            log::info!("arg[{i}]: {arg}");
        }
        for (key, value) in std::env::vars() {
            if key.starts_with("LD_") || key.starts_with("RUST_") {
                log::info!("{key}: {value}");
            }
        }
        let config = network::NetworkConfig::read_from_file();
        let id = {
            let mut stream = std::net::TcpStream::connect(&config.daemon_socket).unwrap();
            stream.write_all(&std::process::id().to_be_bytes()).unwrap();
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).unwrap();
            i32::from_be_bytes(buf)
        };
        log::info!("[#{id}] PID = {}, {:?}", std::process::id(), std::thread::current().id());
        let (channel_sender, channel_receiver) = match config.comm_type.as_str() {
            "shm" => {
                let (sender, receiver) = SHMChannel::new_client_with_id(&config, id).unwrap();
                if config.emulator {
                    (
                        Channel::new(Box::new(EmulatorChannel::new(sender, &config))),
                        Channel::new(Box::new(EmulatorChannel::new(receiver, &config))),
                    )
                } else {
                    (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
                }
            }
            "tcp" => {
                let (sender, receiver) = tcp::new_client(&config, id).unwrap();
                (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                let (sender, receiver) = RDMAChannel::new_client(&config, id);
                (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            &_ => panic!("Unsupported communication type in config"),
        };

        CLIENT_THREAD_INIT.set(true);
        unsafe {
            unsafe extern "C" fn atfork() {
                assert!(!CLIENT_THREAD_INIT.get());
            }
            assert_eq!(0, libc::pthread_atfork(Some(atfork), None, None));

            // HACK: should just send something to the daemon socket
            fn atsignal(_info: &libc::siginfo_t) {
                std::process::exit(0);
            }
            signal_hook_registry::register_sigaction(libc::SIGQUIT, atsignal).unwrap();
            signal_hook_registry::register_sigaction(libc::SIGTERM, atsignal).unwrap();
        }

        Self {
            id,
            channel_sender,
            channel_receiver,
            resource_idx: 0,
            cuda_device: None,
            cuda_device_init: false,
            opt_async_api: config.opt_async_api,
            opt_shadow_desc: config.opt_shadow_desc,
            opt_local: config.opt_local,
            #[cfg(feature = "phos")]
            phos_agent: phos::POSAgent::new(),
        }
    }
}

impl Drop for ClientThread {
    fn drop(&mut self) {
        #[cfg(feature = "phos")]
        self.phos_agent.drop();

        let proc_id = -1;
        proc_id.send(&self.channel_sender).unwrap();
        self.channel_sender.flush_out().unwrap();
    }
}

thread_local! {
    static CLIENT_THREAD: RefCell<ClientThread> = RefCell::new(ClientThread::new());
    static CLIENT_THREAD_INIT: Cell<bool> = const { Cell::new(false) };
}

static DRIVER_CACHE: RwLock<DriverCache> = RwLock::new(DriverCache::new());
static RUNTIME_CACHE: RwLock<RuntimeCache> = RwLock::new(RuntimeCache::new());

struct DriverCache {
    /// Used in `cuModuleGetFunction`, populated by `cuModuleLoadData`.
    images: BTreeMap<CUmodule, Cow<'static, [u8]>>,
    /// Used in `cuLaunchKernel`, populated by `cuModuleGetFunction`.
    function_params: BTreeMap<CUfunction, Box<[KernelParamInfo]>>,
}

// The pointers are server-side.
unsafe impl Send for DriverCache {}
unsafe impl Sync for DriverCache {}

impl DriverCache {
    const fn new() -> Self {
        Self {
            images: BTreeMap::new(),
            function_params: BTreeMap::new(),
        }
    }
}

struct RuntimeCache {
    cuda_device: Option<std::ffi::c_int>,
    /// Populated by `__cudaRegisterFatBinary`.
    lazy_fatbins: Vec<*const FatBinaryHeader>,
    /// Populated by `__cudaRegisterFunction`.
    lazy_functions: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Populated by `__cudaRegisterVar`.
    lazy_variables: BTreeMap<HostPtr, (FatBinaryHandle, *const c_char)>,
    /// Result of `cuModuleLoadData` calls.
    loaded_modules: BTreeMap<FatBinaryHandle, CUmodule>,
    /// Used in `cudaLaunchKernel`. Cache of `cuModuleGetFunction` calls.
    loaded_functions: BTreeMap<HostPtr, CUfunction>,
}

// The pointers are either static or server-side.
unsafe impl Send for RuntimeCache {}
unsafe impl Sync for RuntimeCache {}

impl RuntimeCache {
    const fn new() -> Self {
        Self {
            cuda_device: None,
            lazy_fatbins: Vec::new(),
            lazy_functions: BTreeMap::new(),
            lazy_variables: BTreeMap::new(),
            loaded_modules: BTreeMap::new(),
            loaded_functions: BTreeMap::new(),
        }
    }
}

#[small_ctor::ctor]
unsafe fn init() {
//     core_affinity::set_for_current(1);
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
}
