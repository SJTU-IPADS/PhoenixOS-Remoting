#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + 1;
    }
}

int main(int argc, char **argv)
{
    const int size = 1000000;
    const int iterations = 10000;
    int *a = new int[size];
    int *dev_a = nullptr;

    for (int i = 0; i < size; i++) {
        a[i] = i;
    }
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void **)&dev_a, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

    // remove initial overhead
    for (int i = 0; i < 10; i++) {
        addKernel<<<(size + 1023) / 1024, 1024>>>(dev_a, dev_a, dev_a, size);
    }

    if (auto err = cudaGetLastError()) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Launch a kernel on the GPU with one thread for each element.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        addKernel<<<(size + 1023) / 1024, 1024>>>(dev_a, dev_a, dev_a, size);
    }
    // cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double totalElapsedTime = elapsed.count();

    //  Calculate the average elapsed time
    double averageElapsedTime = totalElapsedTime / iterations;

    std::cout << "Total elapsed time: " << totalElapsedTime << " ms" << std::endl;
    std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);

    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
