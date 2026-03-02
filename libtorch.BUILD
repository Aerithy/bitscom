# libtorch.BUILD
# 引用系统已安装的 PyTorch (pip install torch) 的头文件和预编译库
# 用于 Bazel 构建时链接 libtorch

package(default_visibility = ["//visibility:public"])

# ==================== 头文件 ====================

cc_library(
    name = "torch_headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
    ]),
    includes = ["include"],
    # PyTorch 头文件里用到了 c10、ATen 等，都在 include/ 下
)

# ==================== 预编译库 ====================

cc_library(
    name = "c10",
    srcs = ["lib/libc10.so"],
    deps = [":torch_headers"],
)

cc_library(
    name = "c10_cuda",
    srcs = ["lib/libc10_cuda.so"],
    deps = [":c10"],
)

cc_library(
    name = "torch_cpu",
    srcs = ["lib/libtorch_cpu.so"],
    deps = [
        ":c10",
        ":torch_headers",
    ],
)

cc_library(
    name = "torch_cuda",
    srcs = ["lib/libtorch_cuda.so"],
    deps = [
        ":c10_cuda",
        ":torch_cpu",
    ],
)

cc_library(
    name = "torch",
    srcs = ["lib/libtorch.so"],
    deps = [
        ":torch_cpu",
        ":torch_cuda",
        ":torch_headers",
    ],
)

cc_library(
    name = "torch_python",
    srcs = ["lib/libtorch_python.so"],
    deps = [":torch"],
)
