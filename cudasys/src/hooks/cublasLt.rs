use crate::types::cublasLt::*;
use codegen::cuda_hook;
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_hook(proc_id = 1500)]
fn cublasLtCreate(lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1501)]
fn cublasLtDestroy(lightHandle: cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1511)]
fn cublasLtMatmul(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    #[host] alpha: *const HackedAssumeDouble, // FIXME: safe until we support setting pointer mode
    #[device] A: *const c_void,
    Adesc: cublasLtMatrixLayout_t,
    #[device] B: *const c_void,
    Bdesc: cublasLtMatrixLayout_t,
    #[host] beta: *const HackedAssumeDouble,
    #[device] C: *const c_void,
    Cdesc: cublasLtMatrixLayout_t,
    #[device] D: *mut c_void,
    Ddesc: cublasLtMatrixLayout_t,
    #[host] algo: *const cublasLtMatmulAlgo_t, // FIXME: nullable
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
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
    requestedAlgoCount: c_int,
    #[host(output, len = requestedAlgoCount)]
    heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
    returnAlgoCount: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1519)]
fn cublasLtMatmulDescCreate(
    matmulDesc: *mut cublasLtMatmulDesc_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1521)]
fn cublasLtMatmulDescDestroy(matmulDesc: cublasLtMatmulDesc_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1523)]
fn cublasLtMatmulDescSetAttribute(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t {
    'client_before_send: {
        assert_ne!(attr, cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE);
    }
}

#[cuda_hook(proc_id = 1524)]
fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1526)]
fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1528)]
fn cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
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
fn cublasLtMatrixLayoutDestroy(matLayout: cublasLtMatrixLayout_t) -> cublasStatus_t;
