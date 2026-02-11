load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "core",
    srcs = [
        "src/tensor.cc",
        # "src/operations.cc",
    ],
    hdrs = [
        "include/tensor.h",
        # "include/operations.h",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fPIC",
        "-isystem",
        "/usr/local/cuda/include",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
    # includes = ["include"],
    # visibility = ["//visibility:public"],
    # deps = [
    #     # "//cuda:kernels",  # 依赖 CUDA 层
    #     # "@eigen",
    # ],
)

# cc_test(
#     name = "tensor_test",
#     srcs = ["tests/test_tensor.cc"],
#     deps = [
#         ":core",
#         "@com_google_googletest//:gtest_main",
#     ],
# )