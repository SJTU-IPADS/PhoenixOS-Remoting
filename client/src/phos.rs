use std::ffi::c_void;

use network::{Channel, CommChannel, Transportable};

#[link(name = "pos")]
extern "C" {
    fn pos_create_agent() -> *mut c_void;
    fn pos_destory_agent(pos_agent: *mut c_void) -> i32;
    fn pos_agent_get_uuid(pos_agent: *mut c_void) -> u64;
    fn pos_query_agent_ready_state(pos_agent: *mut c_void) -> i32;
}

pub struct POSAgent(*mut c_void);

impl POSAgent {
    pub fn new() -> Self {
        let pos_agent = unsafe { pos_create_agent() };
        assert!(!pos_agent.is_null());
        Self(pos_agent)
    }

    fn get_uuid(&self) -> u64 {
        unsafe { pos_agent_get_uuid(self.0) }
    }

    fn is_ready(&self) -> bool {
        0 != unsafe { pos_query_agent_ready_state(self.0) }
    }

    pub fn block_until_ready(&self, sender: &mut Channel) {
        if self.is_ready() {
            return;
        }

        log::info!("Sending checkpoint signal...");
        (-2).send(sender).unwrap();
        self.get_uuid().send(sender).unwrap();
        sender.flush_out().unwrap();

        log::info!("Blocking until ready...");
        while !self.is_ready() {
            std::thread::yield_now();
        }
    }

    pub fn drop(&mut self) {
        unsafe { pos_destory_agent(self.0) };
        self.0 = std::ptr::null_mut();
    }
}
