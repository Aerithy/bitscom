package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
)

cc_library(
    name = "cudart",
    srcs = ["lib64/libcudart.so"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
)

cc_library(
    name = "cublas",
    srcs = ["lib64/libcublas.so"],
    hdrs = glob(["include/cublas*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)

cc_library(
    name = "cufft",
    srcs = ["lib64/libcufft.so"],
    hdrs = glob(["include/cufft*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)

cc_library(
    name = "curand",
    srcs = ["lib64/libcurand.so"],
    hdrs = glob(["include/curand*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)

cc_library(
    name = "cusparse",
    srcs = ["lib64/libcusparse.so"],
    hdrs = glob(["include/cusparse*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)

cc_library(
    name = "cusolver",
    srcs = ["lib64/libcusolver.so"],
    hdrs = glob(["include/cusolver*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)

cc_library(
    name = "cudnn",
    srcs = ["lib64/libcudnn.so"],
    hdrs = glob(["include/cudnn*.h"]),
    includes = ["include"],
    deps = [":cudart"],
)