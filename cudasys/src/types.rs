#![expect(non_camel_case_types, non_upper_case_globals)]

macro_rules! success_return_value {
    ($ty:ident::$variant:ident) => {
        const _: () = {
            assert!($ty::$variant as u32 == 0);
            const fn test_num_derive<T: crate::FromPrimitive>() {}
            test_num_derive::<$ty>(); // see `DeriveCallback` in `build.rs`
        };
        impl Default for $ty {
            #[inline(always)]
            fn default() -> Self {
                Self::$variant
            }
        }
    };
}

macro_rules! impl_is_error {
    ($ty:ident) => {
        impl $ty {
            #[inline(always)]
            pub fn is_error(self) -> bool {
                self != Self::default()
            }
        }
    };
}

pub mod cuda {
    include!("bindings/types/cuda.rs");

    const _: () = assert!(CUDA_VERSION >= 11030);

    success_return_value!(CUresult::CUDA_SUCCESS);
    impl_is_error!(CUresult);

    impl CUpointer_attribute {
        pub fn data_size(&self) -> usize {
            match self {
                Self::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => size_of::<*mut CUdeviceptr>(),
                Self::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR => size_of::<usize>(),
                _ => panic!("unsupported pointer attribute {self:?}"),
            }
        }
    }
}

pub mod cudart {
    include!("bindings/types/cudart.rs");

    const _: () = assert!(CUDART_VERSION >= 11030);

    success_return_value!(cudaError_t::cudaSuccess);

    impl cudaError_t {
        pub fn is_error(self) -> bool {
            !matches!(
                self,
                Self::cudaSuccess
                    | Self::cudaErrorNotReady
                    | Self::cudaErrorPeerAccessAlreadyEnabled
            )
        }
    }
}

pub mod nvml {
    include!("bindings/types/nvml.rs");

    success_return_value!(nvmlReturn_t::NVML_SUCCESS);
    impl_is_error!(nvmlReturn_t);
}

pub mod cudnn {
    include!("bindings/types/cudnn.rs");

    success_return_value!(cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    impl_is_error!(cudnnStatus_t);

    impl cudnnBackendAttributeType_t {
        pub fn data_size(&self) -> i64 {
            let result = match self {
                Self::CUDNN_TYPE_HANDLE => size_of::<cudnnHandle_t>(),
                Self::CUDNN_TYPE_DATA_TYPE => size_of::<cudnnDataType_t>(),
                // Self::CUDNN_TYPE_BOOLEAN => size_of::<bool>(),
                Self::CUDNN_TYPE_INT64 => size_of::<i64>(),
                Self::CUDNN_TYPE_FLOAT => size_of::<f32>(),
                // Self::CUDNN_TYPE_DOUBLE => size_of::<f64>(),
                Self::CUDNN_TYPE_VOID_PTR => size_of::<*mut std::ffi::c_void>(),
                Self::CUDNN_TYPE_CONVOLUTION_MODE => size_of::<cudnnConvolutionMode_t>(),
                Self::CUDNN_TYPE_HEUR_MODE => size_of::<cudnnBackendHeurMode_t>(),
                Self::CUDNN_TYPE_KNOB_TYPE => size_of::<cudnnBackendKnobType_t>(),
                // Self::CUDNN_TYPE_NAN_PROPOGATION => size_of::<cudnnNanPropagation_t>(),
                Self::CUDNN_TYPE_NUMERICAL_NOTE => size_of::<cudnnBackendNumericalNote_t>(),
                // Self::CUDNN_TYPE_LAYOUT_TYPE => size_of::<cudnnBackendLayoutType_t>(),
                // Self::CUDNN_TYPE_ATTRIB_NAME => size_of::<cudnnBackendAttributeName_t>(),
                // Self::CUDNN_TYPE_POINTWISE_MODE => size_of::<cudnnPointwiseMode_t>(),
                Self::CUDNN_TYPE_BACKEND_DESCRIPTOR => size_of::<cudnnBackendDescriptor_t>(),
                // Self::CUDNN_TYPE_GENSTATS_MODE => size_of::<cudnnGenStatsMode_t>(),
                // Self::CUDNN_TYPE_BN_FINALIZE_STATS_MODE => size_of::<cudnnBnFinalizeStatsMode_t>(),
                // Self::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE => size_of::<cudnnReduceTensorOp_t>(),
                // Self::CUDNN_TYPE_BEHAVIOR_NOTE => size_of::<cudnnBackendBehaviorNote_t>(),
                // Self::CUDNN_TYPE_TENSOR_REORDERING_MODE => size_of::<cudnnBackendTensorReordering_t>(),
                // Self::CUDNN_TYPE_RESAMPLE_MODE => size_of::<cudnnResampleMode_t>(),
                // Self::CUDNN_TYPE_PADDING_MODE => size_of::<cudnnPaddingMode_t>(),
                // Self::CUDNN_TYPE_INT32 => size_of::<i32>(),
                // Self::CUDNN_TYPE_CHAR => size_of::<std::ffi::c_char>(),
                // Self::CUDNN_TYPE_SIGNAL_MODE => size_of::<cudnnSignalMode_t>(),
                // Self::CUDNN_TYPE_FRACTION => size_of::<cudnnFraction_t>(),
                // Self::CUDNN_TYPE_NORM_MODE => size_of::<cudnnBackendNormMode_t>(),
                // Self::CUDNN_TYPE_NORM_FWD_PHASE => size_of::<cudnnBackendNormFwdPhase_t>(),
                // Self::CUDNN_TYPE_RNG_DISTRIBUTION => size_of::<cudnnRngDistribution_t>(),
                _ => panic!("unsupported attribute type {self:?}"),
            };
            result as i64
        }
    }
}

pub mod cublas {
    include!("bindings/types/cublas.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    impl_is_error!(cublasStatus_t);
}

pub mod cublasLt {
    include!("bindings/types/cublasLt.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    impl_is_error!(cublasStatus_t);
}

pub mod nvrtc {
    include!("bindings/types/nvrtc.rs");

    success_return_value!(nvrtcResult::NVRTC_SUCCESS);
    impl_is_error!(nvrtcResult);
}

pub mod nccl {
    include!("bindings/types/nccl.rs");

    const _: () = assert!(NCCL_VERSION_CODE >= 21602, "run `apt install libnccl2 libnccl-dev`");

    success_return_value!(ncclResult_t::ncclSuccess);
    impl_is_error!(ncclResult_t);
}
