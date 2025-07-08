use crate::types::cudnn::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_hook(proc_id = 1804)]
fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1814)]
fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1873)]
fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1805)]
fn cudnnCreateActivationDescriptor(
    activationDesc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1861)]
fn cudnnSetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1816)]
fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2126)]
fn cudnnSetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1871, async_api)]
fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1875, async_api)]
fn cudnnSetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: c_int,
    #[host(len = nbDims)] dimA: *const c_int,
    #[host(len = nbDims)] strideA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1826, async_api)]
fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1808)]
fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1820, async_api)]
fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1865, async_api)]
fn cudnnSetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: c_int,
    #[host(len = nbDims)] filterDimA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2104)]
fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2105, async_api)]
fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2129, async_api)]
fn cudnnSetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: c_int,
    #[host(len = arrayLength)] padA: *const c_int,
    #[host(len = arrayLength)] filterStrideA: *const c_int,
    #[host(len = arrayLength)] dilationA: *const c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2127, async_api)]
fn cudnnSetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2128, async_api)]
fn cudnnSetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2130)]
fn cudnnSetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2116)]
fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2103, async_api)]
fn cudnnConvolutionForward(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2008)]
fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2009)]
fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2004, async_api)]
fn cudnnBatchNormalizationForwardTrainingEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    zDesc: cudnnTensorDescriptor_t,
    #[device] zData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    exponentialAverageFactor: f64,
    #[device] resultRunningMean: *mut c_void,
    #[device] resultRunningVariance: *mut c_void,
    epsilon: f64,
    #[device] resultSaveMean: *mut c_void,
    #[device] resultSaveInvVariance: *mut c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workspace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2007)]
fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2002, async_api)]
fn cudnnBatchNormalizationBackwardEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alphaDataDiff: *const HackedAssumeDouble,
    #[host] betaDataDiff: *const HackedAssumeDouble,
    #[host] alphaParamDiff: *const HackedAssumeDouble,
    #[host] betaParamDiff: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dyData: *const c_void,
    dzDesc: cudnnTensorDescriptor_t,
    #[device] dzData: *mut c_void,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dxData: *mut c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    #[device] bnScaleData: *const c_void,
    #[device] bnBiasData: *const c_void,
    #[device] dBnScaleData: *mut c_void,
    #[device] dBnBiasData: *mut c_void,
    epsilon: f64,
    #[device] savedMean: *const c_void,
    #[device] savedInvVariance: *const c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2113)]
fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2101, async_api)]
fn cudnnConvolutionBackwardData(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dx: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2213)]
fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2202, async_api)]
fn cudnnConvolutionBackwardFilter(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dwDesc: cudnnFilterDescriptor_t,
    #[device] dw: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1802, async_api)]
fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    #[device] estimatedMean: *const c_void,
    #[device] estimatedVariance: *const c_void,
    epsilon: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1864)]
fn cudnnSetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2121)]
fn cudnnGetConvolutionNdForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    nbDims: c_int,
    #[host(output, len = nbDims)] tensorOuputDimA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2117)]
fn cudnnGetConvolutionForwardWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_custom_hook] // local
fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

// TODO: shadow_desc
#[cuda_hook(proc_id = 2500)]
fn cudnnBackendCreateDescriptor(
    descriptorType: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2501)]
fn cudnnBackendDestroyDescriptor(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2502)]
fn cudnnBackendExecute(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2503)]
fn cudnnBackendFinalize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992504, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeCount(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[device] arrayOfElements: *mut c_void, // null
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992505, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeData(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // not CUDNN_TYPE_BACKEND_DESCRIPTOR
    requestedElementCount: i64,
    elementCount: *mut i64,
    // `len` and `cap` are required because some callers provide buffers shorter than `requestedElementCount`
    // so that using `len = requestedElementCount` alone will lead to memory corruption
    // https://github.com/NVIDIA/cudnn-frontend/blob/v1.11.0/include/cudnn_frontend_ExecutionPlan.h#L165-L177
    #[host(
        output,
        len = attributeType.data_size() * elementCount.to_owned(),
        cap = attributeType.data_size() * requestedElementCount,
    )]
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992506, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeDescriptors(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // CUDNN_TYPE_BACKEND_DESCRIPTOR
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[host(input, len = attributeType.data_size() * requestedElementCount)]
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2506)]
fn cudnnBackendSetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    elementCount: i64,
    #[host(len = attributeType.data_size() * elementCount)] arrayOfElements: *const c_void,
) -> cudnnStatus_t;
