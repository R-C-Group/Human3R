# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# 直接使用系统GCC 11.2.0
host_compiler = "/usr/bin/g++"
print(f"使用系统编译器: {host_compiler}")

# 手动指定CUDA架构
all_cuda_archs = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_60,code=sm_60', 
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86'
]

nvcc_args = [
    "-O3", 
    "--ptxas-options=-v", 
    "--use_fast_math",
    "-allow-unsupported-compiler",
    f"-ccbin={host_compiler}"  # 直接指定系统GCC
] + all_cuda_archs

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=nvcc_args,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)