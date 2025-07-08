use std::ffi::c_void;

#[link(name = "pos")]
extern "C" {
    fn pos_create_workspace_cuda() -> *mut c_void;
    fn pos_process(
        pos_cuda_ws: *mut c_void,
        api_id: u64,
        uuid: u64,
        param_desps: *mut u64,
        param_num: i32,
    ) -> i32;
    fn pos_destory_workspace_cuda(pos_cuda_ws: *mut c_void) -> i32;
    fn pos_remoting_stop_query(pos_cuda_ws: *mut c_void, uuid: u64) -> i32;
    fn pos_remoting_stop_confirm(pos_cuda_ws: *mut c_void, uuid: u64) -> i32;
}

#[expect(non_camel_case_types)]
pub struct POSWorkspace_CUDA(*mut c_void);

impl POSWorkspace_CUDA {
    pub fn new() -> Self {
        log::info!("Starting PhOS server ...");
        let pos_cuda_ws = unsafe { pos_create_workspace_cuda() };
        assert!(!pos_cuda_ws.is_null());
        log::info!("PhOS daemon is running. You can run a program like \"env $phos python3 train.py \" now");
        Self(pos_cuda_ws)
    }

    #[cfg(target_pointer_width = "64")]
    pub fn pos_process(&self, api_id: i32, uuid: u64, param_desps: &[usize]) -> i32 {
        unsafe {
            pos_process(
                self.0,
                api_id as u64,
                uuid,
                param_desps.as_ptr() as *mut u64,
                (param_desps.len() / 2) as i32,
            )
        }
    }

    pub fn stop(&self, uuid: u64) {
        log::info!("Received checkpoint signal from {uuid}");
        assert_eq!(1, unsafe { pos_remoting_stop_query(self.0, uuid) });
        assert_eq!(0, unsafe { pos_remoting_stop_confirm(self.0, uuid) });
    }
}

impl Drop for POSWorkspace_CUDA {
    fn drop(&mut self) {
        unsafe { pos_destory_workspace_cuda(self.0) };
    }
}
