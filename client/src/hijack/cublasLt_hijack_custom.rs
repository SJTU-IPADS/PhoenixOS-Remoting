use super::*;
use cudasys::types::cublasLt::*;
use std::cell::RefCell;
use std::ffi::*;
use std::sync::RwLock;
use std::collections::HashMap;
use std::mem;
use once_cell::sync::Lazy;

type CublasLtMatmulDescT = usize;
type CublasLtPointerModeT = usize;

static MATMUL_DESC_POINTER_MODES: Lazy<RwLock<HashMap<CublasLtMatmulDescT, CublasLtPointerModeT>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));

#[no_mangle]
pub extern "C" fn cublasLtMatmulDescSetAttribute(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *const ::std::os::raw::c_void,
    sizeInBytes: usize,
) -> cublasStatus_t {
    unsafe {
        let res = std::mem::transmute(super::cublasLt_hijack::cublasLtMatmulDescSetAttributeInternal(
            matmulDesc,
            attr,
            buf as *const u8,
            sizeInBytes,
        ));
        if res == cublasStatus_t::CUBLAS_STATUS_SUCCESS && attr == cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE {
            if sizeInBytes == std::mem::size_of::<cublasLtPointerMode_t>() {
                let pointer_mode = unsafe { *(buf as *const cublasLtPointerMode_t) };
                let mut modes = MATMUL_DESC_POINTER_MODES.write().unwrap();
                modes.insert(matmulDesc as usize, pointer_mode as usize);
            }
        }
        res
    }
}

#[no_mangle]
pub extern "C" fn cublasLtMatmul(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    alpha: *const ::std::os::raw::c_void,
    A: *const ::std::os::raw::c_void,
    Adesc: cublasLtMatrixLayout_t,
    B: *const ::std::os::raw::c_void,
    Bdesc: cublasLtMatrixLayout_t,
    beta: *const ::std::os::raw::c_void,
    C: *const ::std::os::raw::c_void,
    Cdesc: cublasLtMatrixLayout_t,
    D: *mut ::std::os::raw::c_void,
    Ddesc: cublasLtMatrixLayout_t,
    algo: *const cublasLtMatmulAlgo_t,
    workspace: *mut ::std::os::raw::c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
) -> cublasStatus_t {
    let desc_key = computeDesc as CublasLtMatmulDescT;

    let is_device_mode = if let Ok(map) = MATMUL_DESC_POINTER_MODES.read() {
        map.get(&desc_key)
            .map(|&mode| mode == cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE as CublasLtPointerModeT)
            .unwrap_or(false)
    } else {
        false
    };

    if is_device_mode {
        super::cublasLt_hijack::cublasLtMatmulDevice(
            lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream
        )
    } else {
        super::cublasLt_hijack::cublasLtMatmulHost(
            lightHandle, computeDesc, alpha as *const u8, A, Adesc, B, Bdesc,
            beta as *const u8, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream
        )
    }
}