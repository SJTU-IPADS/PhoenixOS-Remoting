#![allow(non_snake_case)]
use super::*;

gen_hijack!(0, "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_hijack!(2, "cudaGetDeviceCount", "cudaError_t", "*mut ::std::os::raw::c_int");
