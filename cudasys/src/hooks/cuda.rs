use crate::types::cuda::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 670)]
fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 918, async_api)]
fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    #[skip] kernelParams: *mut *mut c_void,
    #[skip] extra: *mut *mut c_void,
) -> CUresult {
    'client_before_send: {
        assert!(extra.is_null());
        let args = super::cuda_hijack_utils::pack_kernel_args(
            kernelParams,
            DRIVER_CACHE.read().unwrap().function_params.get(&f).unwrap(),
        );
    }
    'client_extra_send: {
        send_slice(&args, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let args = recv_slice::<u8, _>(channel_receiver).unwrap();
    }
    'server_execution: {
        let result = super::cuda_exe_utils::cu_launch_kernel(
            #[cfg(feature = "phos")]
            &server.pos_cuda_ws,
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            &args,
        );
    }
}

#[cuda_custom_hook] // calls the internal API below
fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

#[cuda_hook(proc_id = 701, parent = cuModuleLoadData)]
fn cuModuleLoadDataInternal(
    module: *mut CUmodule,
    #[host(len = len)] image: *const c_void,
    #[skip] is_runtime: bool,
) -> CUresult {
    'client_before_send: {
        let len = if FatBinaryHeader::is_fat_binary(image) {
            let header: &FatBinaryHeader = unsafe { &*image.cast() };
            header.entire_len()
        } else {
            crate::elf::elf_len(image)
        };
    }
    'client_after_recv: {
        let image = if is_runtime {
            std::borrow::Cow::Borrowed(image)
        } else {
            std::borrow::Cow::Owned(image.to_vec())
        };
        assert!(DRIVER_CACHE.write().unwrap().images.insert(*module, image).is_none());
    }
    'server_execution: {
        #[cfg(not(feature = "phos"))]
        let result = unsafe { cuModuleLoadData(module__ptr, image__ptr.cast()) };
        // https://github.com/SJTU-IPADS/PhoenixOS/blob/main/unittest/test_cuda/apis/cuda_driver/cuModuleLoadData.cpp
        #[cfg(feature = "phos")]
        let result = CUresult::from_i32(server.pos_cuda_ws.pos_process(
            701,
            0u64,
            &[
                &raw const module__ptr as usize,
                size_of_val(&module__ptr),
                image__ptr as usize,
                image.len(),
            ],
        ))
        .expect("Illegal result ID");
    }
    'server_after_send: {
        server.modules.push(module);
    }
}

#[cuda_hook(proc_id = 705)]
fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult {
    'client_after_recv: {
        let mut driver = DRIVER_CACHE.write().unwrap();
        let image = driver.images.get(&hmod).unwrap();
        let params = if FatBinaryHeader::is_fat_binary(image.as_ptr()) {
            let fatbin: &FatBinaryHeader = unsafe { &*image.as_ptr().cast() };
            fatbin.find_kernel_params(name.to_str().unwrap())
        } else {
            crate::elf::find_kernel_params(image, name.to_str().unwrap())
        };
        assert!(driver.function_params.insert(*hfunc, params).is_none());
    }
}

#[cuda_hook(proc_id = 640)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 630, async_api = false)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 684)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

#[cuda_hook(proc_id = 650)]
fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

#[cuda_hook(proc_id = 651)]
fn cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;

#[cuda_hook(proc_id = 910)]
fn cuFuncGetAttribute(pi: *mut c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult;

#[cuda_hook(proc_id = 844)]
fn cuPointerGetAttribute(
    #[host(output, len = attribute.data_size())] data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult;

#[cuda_hook(proc_id = 1002)]
fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut c_int,
    func: CUfunction,
    blockSize: c_int,
    dynamicSMemSize: usize,
    flags: c_uint,
) -> CUresult;

#[cuda_hook(proc_id = 912, async_api = false)]
fn cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: c_int) -> CUresult;
