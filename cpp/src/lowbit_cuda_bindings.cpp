#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <utility>

namespace py = pybind11;

namespace bitscom {
std::pair<at::Tensor, double> quantize_cuda(
    const at::Tensor& input,
    int64_t bitwidth,
    bool stochastic_rounding);
at::Tensor dequantize_cuda(
    const at::Tensor& q_tensor,
    double scale);
at::Tensor pack_lowbit_cuda(
    const at::Tensor& q_tensor,
    int64_t bitwidth);
at::Tensor unpack_lowbit_cuda(
    const at::Tensor& packed,
    int64_t bitwidth,
    int64_t numel);
}  // namespace bitscom

PYBIND11_MODULE(_lowbit_cuda, m) {
    m.doc() = "bitscom CUDA quantization kernels";

    m.def(
        "quantize_cuda",
        [](const at::Tensor& input, int64_t bitwidth, bool stochastic_rounding) {
            auto result = bitscom::quantize_cuda(input, bitwidth, stochastic_rounding);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("input"),
        py::arg("bitwidth"),
        py::arg("stochastic_rounding") = false);

    m.def(
        "dequantize_cuda",
        &bitscom::dequantize_cuda,
        py::arg("q_tensor"),
        py::arg("scale"));

    m.def(
        "pack_lowbit_cuda",
        &bitscom::pack_lowbit_cuda,
        py::arg("q_tensor"),
        py::arg("bitwidth"));

    m.def(
        "unpack_lowbit_cuda",
        &bitscom::unpack_lowbit_cuda,
        py::arg("packed"),
        py::arg("bitwidth"),
        py::arg("numel"));
}
