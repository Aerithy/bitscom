cc_library(
    name = "core",
    srcs = ["src/tensor.cc"],
    hdrs = ["include/tensor.h"],
    deps = [
        "@local_cuda//:cudart",
        "@local_cuda//:cublas",
        "@local_cuda//:cuda_headers",
    ],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)

# ProcessGroupLowBit 库
cc_library(
    name = "process_group_lowbit",
    srcs = ["src/process_group_lowbit.cc"],
    hdrs = ["include/process_group_lowbit.h"],
    deps = [
        "@local_cuda//:cudart",
        "@local_libtorch//:torch",
        "@local_nccl//:nccl",
    ],
    copts = [
        "-std=c++17",
        "-DUSE_C10D_NCCL",    # 启用 ProcessGroupNCCL 定义
    ],
    visibility = ["//visibility:public"],
)

# Python extension (pybind11 bindings)
cc_library(
    name = "bindings",
    srcs = ["src/bindings.cc"],
    deps = [
        ":process_group_lowbit",
        "@local_libtorch//:torch",
        "@local_libtorch//:torch_python",
        "@local_python//:python_headers",
    ],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//cpp:core": "",
        "//cpp:process_group_lowbit": "",
        "//cpp:bindings": "",
    },
)

# cc_test(
#     name = "tensor_test",
#     srcs = ["tests/test_tensor.cc"],
#     deps = [
#         ":core",
#         "@com_google_googletest//:gtest_main",
#     ],
# )