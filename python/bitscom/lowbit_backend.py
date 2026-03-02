"""
注册 lowbit backend 到 torch.distributed。

用法:
    import bitscom
    bitscom.init()
    torch.distributed.init_process_group(backend="lowbit", ...)
"""

import torch
import torch.distributed as dist
from datetime import timedelta

# 导入 C++ extension
from bitscom._lowbit_c import create_backend, ProcessGroupLowBit, LowBitOptions


def _create_lowbit_pg(store, rank, size, timeout):
    """
    工厂函数，由 torch.distributed 在 init_process_group(backend="lowbit") 时调用。
    签名需要匹配 torch.distributed.Backend.register_backend 的要求。
    """
    return create_backend(
        store=store,
        rank=rank,
        size=size,
        timeout=timeout,
    )


def register_lowbit_backend():
    """
    将 'lowbit' 注册为 torch.distributed 的可用 backend。
    注册后即可使用:
        dist.init_process_group(backend="lowbit", ...)
    """
    dist.Backend.register_backend(
        name="lowbit",
        func=_create_lowbit_pg,
    )
    print("[bitscom] 'lowbit' backend registered.")
