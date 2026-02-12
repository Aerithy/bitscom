cc_library(
    name = "core",
    srcs = ["src/tensor.cc"],
    hdrs = ["include/tensor.h"],
    deps = [
        "@local_cuda//:cudart",
        "@local_cuda//:cublas",
        "@local_cuda//:cuda_headers",
        # "@com_google_absl//absl/strings",
    ],
    copts = ["-std=c++17"],
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
      "//cpp:core": "",
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