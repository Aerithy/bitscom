"""
使用 torch.utils.cpp_extension 构建 C++ extension。
推荐使用此方式构建（而非 Bazel），以最大程度兼容 PyTorch 生态。

安装:
    pip install -e .

验证:
    python -c "from bitscom._lowbit_c import ProcessGroupLowBit; print('OK')"
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="bitscom",
    version="0.1.0",
    description="Low-bit distributed communication primitives for PyTorch",
    author="Aerithy",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=[
        CppExtension(
            name="bitscom._lowbit_c",
            sources=[
                "cpp/src/process_group_lowbit.cc",
                "cpp/src/bindings.cc",
            ],
            include_dirs=["./"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
            },
        ),
        # 后续加 CUDA kernels 时改用 CUDAExtension:
        # CUDAExtension(
        #     name="bitscom._lowbit_cuda",
        #     sources=[
        #         "cuda/src/pack_kernels.cu",
        #     ],
        #     extra_compile_args={
        #         "cxx": ["-std=c++17"],
        #         "nvcc": ["-std=c++17", "--expt-relaxed-constexpr"],
        #     },
        # ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
    ],
)
