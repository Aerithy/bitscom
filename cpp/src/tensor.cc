// cpp/src/tensor.cc
#include <cstdlib>
#include <cassert>
#include "cpp/include/tensor.h"
// #include "cuda/include/kernels.cuh"
// #include <cuda_runtime.h>

namespace bitscom {

Tensor::Tensor(const std::vector<int64_t>& shape, Device device)
    : shape_(shape), device_(device) {
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    if (device == Device::CUDA) {
        // cudaMalloc(&data_, total_size * sizeof(float));
    } else {
        data_ = malloc(total_size * sizeof(float));
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 检查维度
    assert(shape_.size() == 2 && other.shape_.size() == 2);
    assert(shape_[1] == other.shape_[0]);
    
    int M = shape_[0];
    int K = shape_[1];
    int N = other.shape_[1];
    
    Tensor result({M, N}, device_);
    
    if (device_ == Device::CUDA) {
        // 调用 CUDA kernel
        // cuda::matmul(
        //     static_cast<const float*>(data_),
        //     static_cast<const float*>(other.data_),
        //     static_cast<float*>(result.data_),
        //     M, N, K,
        //     nullptr  // default stream
        // );
        cudaDeviceSynchronize();
    } else {
        // CPU 实现 (使用 Eigen)
        // ...
    }
    
    return result;
}

}  // namespace mylib