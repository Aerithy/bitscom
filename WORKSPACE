workspace(name = "my_cuda_project")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
    sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
    strip_prefix = "rules_cc-0.0.9",
)

new_local_repository(
    name = "local_cuda",
    path = "/usr/local/cuda",
    build_file = "@//:cuda.BUILD",
#     build_file_content = """
# package(default_visibility = ["//visibility:public"])

# cc_library(
#     name = "cuda_headers",
#     hdrs = glob([
#         "include/**/*.h",
#         "include/**/*.hpp",
#     ]),
#     includes = ["include"],
# )

# cc_library(
#     name = "cudart",
#     srcs = ["lib64/libcudart.so"],
#     hdrs = glob(["include/**/*.h"]),
#     includes = ["include"],
# )

# cc_library(
#     name = "cublas",
#     srcs = ["lib64/libcublas.so"],
#     hdrs = glob(["include/cublas*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )

# cc_library(
#     name = "cufft",
#     srcs = ["lib64/libcufft.so"],
#     hdrs = glob(["include/cufft*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )

# cc_library(
#     name = "curand",
#     srcs = ["lib64/libcurand.so"],
#     hdrs = glob(["include/curand*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )

# cc_library(
#     name = "cusparse",
#     srcs = ["lib64/libcusparse.so"],
#     hdrs = glob(["include/cusparse*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )

# cc_library(
#     name = "cusolver",
#     srcs = ["lib64/libcusolver.so"],
#     hdrs = glob(["include/cusolver*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )

# cc_library(
#     name = "cudnn",
#     srcs = ["lib64/libcudnn.so"],
#     hdrs = glob(["include/cudnn*.h"]),
#     includes = ["include"],
#     deps = [":cudart"],
# )
# """,
)

# ========================================
# 常见的 C++ 依赖库示例
# ========================================

# Google Test
http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/release-1.12.1.tar.gz"],
    sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
    strip_prefix = "googletest-release-1.12.1",
)

# Google Benchmark
http_archive(
    name = "com_github_google_benchmark",
    urls = ["https://github.com/google/benchmark/archive/v1.8.0.tar.gz"],
    sha256 = "ea2e94c24ddf6594d15c711c06ccd4486434d9cf3eca954e2af8a20c88f9f172",
    strip_prefix = "benchmark-1.8.0",
)

# Abseil (Google's C++ library)
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20230125.3.tar.gz"],
    sha256 = "5366d7e7fa7ba0d915014d387b66d0d002c03236448e1ba9ef98122c13b35c36",
    strip_prefix = "abseil-cpp-20230125.3",
)

# Eigen (线性代数库)
http_archive(
    name = "eigen",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-3.4.0",
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "hedron_compile_commands",

    # 建议把下面两处 commit hash 换成 github 上最新的版本
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
