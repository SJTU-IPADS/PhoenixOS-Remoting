use std::os::raw::*;

use cudasys::cuda::*;

pub fn cu_launch_kernel(
    #[cfg(feature = "phos")] pos_cuda_ws: &crate::phos::POSWorkspace_CUDA,
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
) -> CUresult {
    #[cfg(not(feature = "phos"))]
    unsafe {
        let args_len = args.len();
        let extra_array: [*mut c_void; 5] = [
            1 as _, // CU_LAUNCH_PARAM_BUFFER_POINTER
            args.as_ptr() as _,
            2 as _, // CU_LAUNCH_PARAM_BUFFER_SIZE
            &raw const args_len as _,
            std::ptr::null_mut(), // CU_LAUNCH_PARAM_END
        ];
        cuLaunchKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            std::ptr::null_mut(),
            extra_array.as_ptr().cast_mut(),
        )
    }
    // https://github.com/SJTU-IPADS/PhoenixOS/blob/main/unittest/test_cuda/apis/cuda_driver/cuLaunchKernel.cpp
    #[cfg(feature = "phos")]
    {
        use super::*;

        CUresult::from_i32(pos_cuda_ws.pos_process(
            918, // cuLaunchKernel
            0,
            &[
                &raw const f as usize,
                size_of_val(&f),
                &raw const gridDimX as usize,
                size_of_val(&gridDimX),
                &raw const gridDimY as usize,
                size_of_val(&gridDimY),
                &raw const gridDimZ as usize,
                size_of_val(&gridDimZ),
                &raw const blockDimX as usize,
                size_of_val(&blockDimX),
                &raw const blockDimY as usize,
                size_of_val(&blockDimY),
                &raw const blockDimZ as usize,
                size_of_val(&blockDimZ),
                &raw const sharedMemBytes as usize,
                size_of_val(&sharedMemBytes),
                &raw const hStream as usize,
                size_of_val(&hStream),
                args.as_ptr() as usize,
                args.len(),
                1, // work around null check
                0,
            ],
        ))
        .expect("Illegal result ID")
    }
}

#[cfg(not(feature = "phos"))]
pub fn cu_func_get_attributes(
    attr: *mut cudasys::cudart::cudaFuncAttributes,
    func: CUfunction,
) -> CUresult {
    let attr = unsafe { &mut *attr };
    // HACK: implementation with cuFuncGetAttribute depends on CUDA version
    macro_rules! get_attributes {
        ($func:ident -> $struct:ident $($field:ident: $attr:ident,)+) => {
            $(
                let mut i = 0;
                let result =
                    unsafe { cuFuncGetAttribute(&raw mut i, CUfunction_attribute::$attr, $func) };
                if result != Default::default() {
                    return result;
                }
                $struct.$field = i as _;
            )+
        }
    }
    get_attributes! { func -> attr
        sharedSizeBytes: CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        constSizeBytes: CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        localSizeBytes: CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        maxThreadsPerBlock: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        numRegs: CU_FUNC_ATTRIBUTE_NUM_REGS,
        ptxVersion: CU_FUNC_ATTRIBUTE_PTX_VERSION,
        binaryVersion: CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        cacheModeCA: CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
        maxDynamicSharedSizeBytes: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    }
    CUresult::CUDA_SUCCESS
}
