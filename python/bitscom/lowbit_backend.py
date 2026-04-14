"""
注册 lowbit backend 到 torch.distributed。

用法:
    import bitscom
    bitscom.init()
    torch.distributed.init_process_group(backend="lowbit", ...)
"""

import torch
import torch.distributed as dist

# 导入 C++ extension
try:
    from bitscom._lowbit_c import create_backend, ProcessGroupLowBit, LowBitOptions

    _HAS_EXTENSION = True
    _EXTENSION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised via unit tests
    create_backend = None
    ProcessGroupLowBit = None
    LowBitOptions = None
    _HAS_EXTENSION = False
    _EXTENSION_IMPORT_ERROR = exc


_REGISTERED = False
_BACKEND_BITWIDTH = 4
_BACKEND_ERROR_FEEDBACK = False


def _create_lowbit_pg(store, rank, size, timeout):
    """
    工厂函数，由 torch.distributed 在 init_process_group(backend="lowbit") 时调用。
    签名需要匹配 torch.distributed.Backend.register_backend 的要求。
    """
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "bitscom C++ extension is not available. "
            "Please build/install the package first."
        ) from _EXTENSION_IMPORT_ERROR
    return create_backend(
        store=store,
        rank=rank,
        size=size,
        timeout=timeout,
        bitwidth=_BACKEND_BITWIDTH,
        error_feedback=_BACKEND_ERROR_FEEDBACK,
    )


def register_lowbit_backend(bitwidth: int = 4, error_feedback: bool = False):
    """
    将 'lowbit' 注册为 torch.distributed 的可用 backend。
    注册后即可使用:
        dist.init_process_group(backend="lowbit", ...)
    """
    global _REGISTERED
    global _BACKEND_BITWIDTH
    global _BACKEND_ERROR_FEEDBACK

    if bitwidth not in (1, 2, 4, 8, 12, 16):
        raise ValueError(
            f"bitwidth must be one of (1, 2, 4, 8, 12, 16), got {bitwidth}"
        )

    if _REGISTERED:
        if bitwidth != _BACKEND_BITWIDTH or error_feedback != _BACKEND_ERROR_FEEDBACK:
            raise RuntimeError(
                "lowbit backend is already registered with different options: "
                f"bitwidth={_BACKEND_BITWIDTH}, error_feedback={_BACKEND_ERROR_FEEDBACK}"
            )
        return

    _BACKEND_BITWIDTH = bitwidth
    _BACKEND_ERROR_FEEDBACK = bool(error_feedback)

    if not _HAS_EXTENSION:
        raise RuntimeError(
            "bitscom C++ extension is not available. "
            "Install with `pip install -e .` before registering backend."
        ) from _EXTENSION_IMPORT_ERROR

    dist.Backend.register_backend(
        name="lowbit",
        func=_create_lowbit_pg,
        devices=["cpu", "cuda"],
    )
    _REGISTERED = True


def is_extension_available() -> bool:
    return _HAS_EXTENSION
