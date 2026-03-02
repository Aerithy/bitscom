// cpp/src/process_group_lowbit.cc
#include "cpp/include/process_group_lowbit.h"

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <iostream>

namespace bitscom {

// ==================== WorkLowBit ====================

WorkLowBit::WorkLowBit(
    c10::intrusive_ptr<c10d::Work> nccl_work,
    std::function<void()> post_hook)
    : c10d::Work(),
      nccl_work_(std::move(nccl_work)),
      post_hook_(std::move(post_hook)) {}

bool WorkLowBit::isCompleted() {
    return nccl_work_->isCompleted();
}

bool WorkLowBit::isSuccess() const {
    return nccl_work_->isSuccess();
}

bool WorkLowBit::wait(std::chrono::milliseconds timeout) {
    bool success = nccl_work_->wait(timeout);
    // 通信完成后执行 post_hook（例如 unpack）
    if (success && post_hook_) {
        post_hook_();
    }
    return success;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkLowBit::getFuture() {
    return nccl_work_->getFuture();
}

// ==================== ProcessGroupLowBit ====================

ProcessGroupLowBit::ProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    LowBitOptions options)
    : c10d::Backend(rank, size), options_(std::move(options)) {

    // 创建底层 NCCL ProcessGroup
    auto nccl_options = c10d::ProcessGroupNCCL::Options::create();
    nccl_options->timeout = options_.timeout;
    nccl_pg_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store, rank, size, std::move(nccl_options));

    std::cout << "[LowBit] ProcessGroupLowBit created: rank=" << rank
              << " size=" << size
              << " bitwidth=" << options_.bitwidth << std::endl;
}

// ---- pack/unpack 占位实现 ----

at::Tensor ProcessGroupLowBit::pack(const at::Tensor& input) {
    // TODO: 实现 CUDA quantize + bit-pack kernel
    // 当前占位：直接 view 为 uint8（不做真正的压缩）
    auto flat = input.contiguous().view(-1);
    auto byte_view = at::from_blob(
        flat.data_ptr(),
        {static_cast<int64_t>(flat.nbytes())},
        flat.options().dtype(at::kByte));
    return byte_view.clone();  // clone 保证内存安全
}

void ProcessGroupLowBit::unpack(const at::Tensor& packed, at::Tensor& output) {
    // TODO: 实现 CUDA unpack + dequantize kernel
    // 当前占位：直接 memcpy 回去
    auto flat = output.contiguous().view(-1);
    auto expected_bytes = static_cast<int64_t>(flat.nbytes());
    TORCH_CHECK(
        packed.numel() == expected_bytes,
        "packed size mismatch: got ", packed.numel(), " expected ", expected_bytes);
    memcpy(flat.data_ptr(), packed.data_ptr(), flat.nbytes());
}

// ---- 集合通信原语 ----

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {

    // 1) Pack 每个 tensor
    std::vector<at::Tensor> packed_tensors;
    packed_tensors.reserve(tensors.size());
    for (auto& t : tensors) {
        packed_tensors.push_back(pack(t));
    }

    // 2) 通过底层 NCCL 对 packed buffer 做 allreduce
    //    注意：对于真正的低比特，这里的 allreduce op 需要特殊处理
    //    当前占位直接用 SUM（因为 pack 是 identity）
    auto nccl_work = nccl_pg_->allreduce(packed_tensors, opts);

    // 3) 构造 post_hook：通信完成后 unpack 回原始 tensor
    auto tensors_copy = tensors;  // 捕获拷贝
    auto packed_copy = packed_tensors;
    auto self = this;
    auto post_hook = [self, tensors_copy, packed_copy]() mutable {
        for (size_t i = 0; i < tensors_copy.size(); ++i) {
            self->unpack(packed_copy[i], tensors_copy[i]);
        }
    };

    return c10::make_intrusive<WorkLowBit>(
        std::move(nccl_work), std::move(post_hook));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
    // broadcast 通常不需要压缩，直接转发到 NCCL
    return nccl_pg_->broadcast(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {

    // TODO: 对 input 做 pack，对 output 做 unpack
    // 当前占位：直接转发到 NCCL
    return nccl_pg_->allgather(output_tensors, input_tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {

    // TODO: 对 input 做 pack，对 output 做 unpack
    // 当前占位：直接转发到 NCCL
    return nccl_pg_->reduce_scatter(output_tensors, input_tensors, opts);
}

// ---- 工厂函数 ----

c10::intrusive_ptr<c10d::Backend> createProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout) {

    LowBitOptions opts;
    opts.timeout = timeout;
    return c10::make_intrusive<ProcessGroupLowBit>(
        store, rank, size, std::move(opts));
}

}  // namespace bitscom
