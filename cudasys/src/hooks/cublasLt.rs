use crate::types::cublasLt::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;
use std::mem;

#[cuda_hook(proc_id = 1500)]
fn cublasLtCreate(lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1501)]
fn cublasLtDestroy(lightHandle: cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1519)]
fn cublasLtMatmulDescCreate(
    matmulDesc: *mut cublasLtMatmulDesc_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1521)]
fn cublasLtMatmulDescDestroy(
  matmulDesc: cublasLtMatmulDesc_t,
) -> cublasStatus_t;

#[cuda_custom_hook]
fn cublasLtMatmulDescSetAttribute(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *const ::std::os::raw::c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1523, parent = cublasLtMatmulDescSetAttribute)]
fn cublasLtMatmulDescSetAttributeInternal(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    #[host(len = sizeInBytes)] buf: *const ::std::os::raw::c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1524)]
fn cublasLtMatmulPreferenceCreate(
    pref: *mut cublasLtMatmulPreference_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1526)]
fn cublasLtMatmulPreferenceDestroy(
    pref: cublasLtMatmulPreference_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1528)]
fn cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    #[host(len = sizeInBytes)] buf: *const ::std::os::raw::c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1529)]
fn cublasLtMatrixLayoutCreate(
    matLayout: *mut cublasLtMatrixLayout_t,
    type_: cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1531)]
fn cublasLtMatrixLayoutDestroy(
  matLayout: cublasLtMatrixLayout_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1516)]
fn cublasLtMatmulAlgoGetHeuristic(
    lightHandle: cublasLtHandle_t,
    operationDesc: cublasLtMatmulDesc_t,
    Adesc: cublasLtMatrixLayout_t,
    Bdesc: cublasLtMatrixLayout_t,
    Cdesc: cublasLtMatrixLayout_t,
    Ddesc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
    requestedAlgoCount: ::std::os::raw::c_int,
    heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
    returnAlgoCount: *mut ::std::os::raw::c_int,
) -> cublasStatus_t;

#[cuda_custom_hook]
fn cublasLtMatmul(
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
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1511, parent = cublasLtMatmul)]
fn cublasLtMatmulHost(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    #[host(len = mem::size_of::<c_float>())] alpha: *const ::std::os::raw::c_void,
    #[device] A: *const ::std::os::raw::c_void,
    Adesc: cublasLtMatrixLayout_t,
    #[device] B: *const ::std::os::raw::c_void,
    Bdesc: cublasLtMatrixLayout_t,
    #[host(len = mem::size_of::<c_float>())] beta: *const ::std::os::raw::c_void,
    #[device] C: *const ::std::os::raw::c_void,
    Cdesc: cublasLtMatrixLayout_t,
    #[device] D: *mut ::std::os::raw::c_void,
    Ddesc: cublasLtMatrixLayout_t,
    #[host] algo: *const cublasLtMatmulAlgo_t,
    #[device] workspace: *mut ::std::os::raw::c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1512, parent = cublasLtMatmul)]
fn cublasLtMatmulDevice(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    #[device] alpha: *const ::std::os::raw::c_void,
    #[device] A: *const ::std::os::raw::c_void,
    Adesc: cublasLtMatrixLayout_t,
    #[device] B: *const ::std::os::raw::c_void,
    Bdesc: cublasLtMatrixLayout_t,
    #[device] beta: *const ::std::os::raw::c_void,
    #[device] C: *const ::std::os::raw::c_void,
    Cdesc: cublasLtMatrixLayout_t,
    #[device] D: *mut ::std::os::raw::c_void,
    Ddesc: cublasLtMatrixLayout_t,
    #[host] algo: *const cublasLtMatmulAlgo_t,
    #[device] workspace: *mut ::std::os::raw::c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
) -> cublasStatus_t;