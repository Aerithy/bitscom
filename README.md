# bitscom

## Bazel Configuration

Each bazel `BUILD` file should be like this:
```python
# BUILD
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
```

to outsider repositories, project and other libraries. Add the following to the `WORKSPACE` file:
```python
# WORKSPACE
new_local_repository(   # Lib without using bazel
    name = "local_cuda",
    path = "/usr/local/cuda",
    build_file = "@//:cuda.BUILD",
)

local_repository(       # Lib with using bazel
    name = "coworkers_project",
    path = "/path/to/coworkers-project",
)
```

## Build Commands

```shell
bazel build //...
```

or 
```shell
bazel build //cpp:core
```

Get `compile_commands.json`
```shell
bazel run //cpp:refresh_compile_commands
```

`compile_commands.json` is generated for `clangd` language server for better linting. 