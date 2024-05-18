#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use std::error::Error;
use std::fmt;
use serde::Deserialize;

#[macro_use]
extern crate lazy_static;

pub mod buffer;
pub use buffer::{BufferError, RawBuffer};

pub mod ringbufferchannel;
pub mod type_impl;

extern crate codegen;

#[derive(Deserialize)]
pub struct NetworkConfig {
    pub comm_type: String,
    pub sender_socket: String,
    pub receiver_socket: String,
    pub stoc_channel_name: String,
    pub ctos_channel_name: String,
    pub buf_size: usize,
}

lazy_static! {
    pub static ref CONFIG: NetworkConfig = {
        // Use environment variable to set config file's path.
        let path = match std::env::var("NETWORK_CONFIG") {
            Ok(val) => val,
            Err(_) => "/workspace/xpuremoting/config.toml".to_string(),
        };
        let content = std::fs::read_to_string(path).expect("Failed to read config.toml");
        toml::from_str(&content).expect("Failed to parse config.toml")
    };
}

/// A raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemory {
    pub ptr: *const u8,
    pub len: usize,
}

impl RawMemory {
    pub fn new<T>(var: &T, len: usize) -> Self {
        RawMemory {
            ptr: var as *const T as *const u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *const u8, len: usize) -> Self {
        RawMemory { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemory {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
    }
}

/// A *mutable* raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemoryMut {
    pub ptr: *mut u8,
    pub len: usize,
}

impl RawMemoryMut {
    pub fn new<T>(var: &mut T, len: usize) -> Self {
        RawMemoryMut {
            ptr: var as *mut T as *mut u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *mut u8, len: usize) -> Self {
        RawMemoryMut { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemoryMut {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
    }
}

#[derive(Debug)]
pub enum CommChannelError {
    // Define error types, for example:
    InvalidOperation,
    IoError,
    Timeout,
    NoLeftSpace,
    // Add other relevant errors
}

impl fmt::Display for CommChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: refine
        write!(f, "Channel Error: {:?}", self)
    }
}

impl Error for CommChannelError {}

///
/// A communication channel allows TBD
pub trait CommChannel {
    /// Write bytes to the channel
    /// It may flush if the channel has no left space
    fn put_bytes(&mut self, src: &RawMemory) -> Result<usize, CommChannelError>;

    /// Non-block version
    /// Immediately return if error such as no left space
    fn try_put_bytes(&mut self, src: &RawMemory) -> Result<usize, CommChannelError>;

    /// Read bytes from the channel
    /// Wait if dont receive enough bytes
    fn get_bytes(&mut self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    /// Non-block version
    /// Immediately return after receive however long bytes (maybe =0 or <len)
    fn try_get_bytes(&mut self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    /// Flush the all the buffered results to the channel
    fn flush_out(&mut self) -> Result<(), CommChannelError>;
}

///
/// The type itself use `CommChannel` to implicitly implement (de-)serialization logic.
///
/// Every type wanted to be transfered should implement this trait.
pub trait Transportable {
    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError>;

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError>;
}
