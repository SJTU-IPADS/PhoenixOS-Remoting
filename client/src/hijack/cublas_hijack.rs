#![allow(non_snake_case)]
use super::*;
use cudasys::types::cublas::*;

gen_hijack!(
    2001,
    "cublasDestroy_v2", 
    "cublasStatus_t", 
    "cublasHandle_t"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    2002,
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    2002,
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);

#[cfg(feature = "async_api")]
gen_hijack_async!(
    2003,
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);
#[cfg(not(feature = "async_api"))]
gen_hijack!(
    2003,
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);

gen_hijack!(
    2006,
    "cublasGetMathMode",
    "cublasStatus_t",
    "cublasHandle_t",
    "*mut cublasMath_t"
);