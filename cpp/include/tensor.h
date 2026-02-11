#pragma once
#include <vector>
#include <cuda_runtime.h>

namespace bitscom {

enum class Device {
    CPU,
    CUDA
};

class Tensor {
    public:
        Tensor(const std::vector<int64_t>& shape, Device device);
        ~Tensor();
        Tensor matmul(const Tensor& other) const;
    private:
        std::vector<int64_t> shape_;
        Device device_;
        void* data_;
};

}  // namespace bitscom