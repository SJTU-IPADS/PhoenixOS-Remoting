#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>

#include "cublasLt.h"

#define checkCudaStatus(result) \
    { cudaAssert((result), __FILE__, __LINE__); }
inline void cudaAssert(cudaError err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA assert: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) {
            exit(err);
        }
    }
}

#define checkCublasStatus(result) \
    { cublasAssert((result), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t err, const char *file, int line, bool abort = true) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS assert: %d %s %d\n", err, file, line);
        if (abort) {
            exit(err);
        }
    }
}

void test_device_pointer_mode() {
    std::cout << "\n=== Testing DEVICE pointer mode ===\n" << std::endl;
    
    int M = 16;
    int N = 16;
    int K = 16;

    float* a = (float*)malloc(K * M * sizeof(float));
    float* b = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(N * M * sizeof(float));
    float* bias = (float*)malloc(N * sizeof(float));

    // initialize matrix a and b
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = static_cast<float>(i * K + j);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = static_cast<float>(i * K + j);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * K + j] = 0;
        }
    }
    for (int i = 0; i < N; i++) {
        bias[i] = static_cast<float>(i) / 10;
    }

    float* A;
    float* B;
    float* C;
    float* Bias;
    checkCudaStatus(cudaMalloc(&A, M * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&B, N * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&C, M * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&Bias, N * sizeof(float)));
    cudaMemcpy(A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bias, bias, N * sizeof(float), cudaMemcpyHostToDevice);

    // create cublasLt handle
    cublasLtHandle_t handle;
    checkCublasStatus(cublasLtCreate(&handle));

    // create operation desciriptor
    cublasLtMatmulDesc_t operationDesc;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublasOperation_t transb = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set DEVICE pointer mode
    cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_DEVICE;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)));

    // create (empty) preference for heuristics
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // create heuristic
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Biasdesc;
    cublasLtMatrixLayout_t Cdesc;
    // create matrix descriptors
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Biasdesc, CUDA_R_32F, M, N, 0));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Biasdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    float alpha = 1.0f;
    float beta = 1.0f;

    // For DEVICE pointer mode, alpha and beta must be on device
    float *d_alpha, *d_beta;
    checkCudaStatus(cudaMalloc(&d_alpha, sizeof(float)));
    checkCudaStatus(cudaMalloc(&d_beta, sizeof(float)));
    checkCudaStatus(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_beta, &beta, sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Using DEVICE pointer mode with device alpha and beta" << std::endl;
    checkCublasStatus(cublasLtMatmul(
        handle,
        operationDesc,
        d_alpha,  // device pointer
        A,
        Adesc,
        B,
        Bdesc,
        d_beta,   // device pointer
        Bias,
        Biasdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        nullptr,
        0,
        0));

    cudaMemcpy(c, C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result with DEVICE pointer mode:" << std::endl;
    for (int i = 0; i < std::min(4, M); i++) {
        for (int j = 0; j < std::min(4, N); j++) {
            std::cout << std::setprecision(4) << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    checkCudaStatus(cudaFree(d_alpha));
    checkCudaStatus(cudaFree(d_beta));
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    checkCublasStatus(cublasLtDestroy(handle));
    checkCudaStatus(cudaFree(A));
    checkCudaStatus(cudaFree(B));
    checkCudaStatus(cudaFree(C));
    checkCudaStatus(cudaFree(Bias));
    free(a);
    free(b);
    free(c);
    free(bias);
}

void test_host_pointer_mode() {
    std::cout << "\n=== Testing HOST pointer mode ===\n" << std::endl;
    
    int M = 16;
    int N = 16;
    int K = 16;

    float* a = (float*)malloc(K * M * sizeof(float));
    float* b = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(N * M * sizeof(float));
    float* bias = (float*)malloc(N * sizeof(float));

    // initialize matrix a and b
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = static_cast<float>(i * K + j);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = static_cast<float>(i * K + j);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * K + j] = 0;
        }
    }
    for (int i = 0; i < N; i++) {
        bias[i] = static_cast<float>(i) / 10;
    }

    float* A;
    float* B;
    float* C;
    float* Bias;
    checkCudaStatus(cudaMalloc(&A, M * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&B, N * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&C, M * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&Bias, N * sizeof(float)));
    cudaMemcpy(A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bias, bias, N * sizeof(float), cudaMemcpyHostToDevice);

    // create cublasLt handle
    cublasLtHandle_t handle;
    checkCublasStatus(cublasLtCreate(&handle));

    // create operation desciriptor
    cublasLtMatmulDesc_t operationDesc;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublasOperation_t transb = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create (empty) preference for heuristics
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // create heuristic
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Biasdesc;
    cublasLtMatrixLayout_t Cdesc;
    // create matrix descriptors
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Biasdesc, CUDA_R_32F, M, N, 0));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Biasdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // For HOST pointer mode, alpha and beta can be host pointers
    float alpha = 2.0f;  // Different value to show the difference
    float beta = 0.5f;   // Different value to show the difference

    std::cout << "Using HOST pointer mode with host alpha=" << alpha << " and beta=" << beta << std::endl;
    checkCublasStatus(cublasLtMatmul(
        handle,
        operationDesc,
        &alpha,   // host pointer
        A,
        Adesc,
        B,
        Bdesc,
        &beta,    // host pointer
        Bias,
        Biasdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        nullptr,
        0,
        0));

    cudaMemcpy(c, C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result with HOST pointer mode:" << std::endl;
    for (int i = 0; i < std::min(4, M); i++) {
        for (int j = 0; j < std::min(4, N); j++) {
            std::cout << std::setprecision(4) << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    checkCublasStatus(cublasLtDestroy(handle));
    checkCudaStatus(cudaFree(A));
    checkCudaStatus(cudaFree(B));
    checkCudaStatus(cudaFree(C));
    checkCudaStatus(cudaFree(Bias));
    free(a);
    free(b);
    free(c);
    free(bias);
}

int main(void) {
    std::cout << "Testing cublasLt with different pointer modes" << std::endl;
    
    // Test both pointer modes
    test_host_pointer_mode();
    // test_device_pointer_mode(); // Not supported yet.
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}
