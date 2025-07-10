use crate::types::cublas::*;
use codegen::cuda_hook;
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeFloat = f32;

#[cuda_hook(proc_id = 1100)]
fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1101)]
fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1104, async_api)]
fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1119, async_api)]
fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1300, async_api)]
fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const f32, // FIXME: safe until we support cublasSetPointerMode()
    #[device] A: *const f32,
    lda: c_int,
    #[device] B: *const f32,
    ldb: c_int,
    #[host] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1313, async_api)]
fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const f32,
    ldb: c_int,
    strideB: c_longlong,
    #[host] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1120)]
fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1441, async_api)]
fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const HackedAssumeFloat,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[host] beta: *const HackedAssumeFloat,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1443, async_api)]
fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const HackedAssumeFloat,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    strideB: c_longlong,
    #[host] beta: *const HackedAssumeFloat,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1105)]
fn cublasSetWorkspace_v2(
    handle: cublasHandle_t,
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1111)]
fn cublasSetMatrix(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[host(len = rows * cols * elemSize)] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1115, async_api)]
fn cublasSetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[host(len = rows * cols * elemSize)] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
    stream: cudaStream_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1112)]
fn cublasGetMatrix(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[device] A: *const c_void,
    lda: c_int,
    #[host(output, len = rows * cols * elemSize)] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1182)]
fn cublasSscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[host] alpha: *const f32, // FIXME: safe until we support cublasSetPointerMode()
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;
