"""
High-level API for low-bit distributed communication.
"""

import torch
import torch.distributed as dist
from typing import Optional, List


class LowBitGroup:
    """
    对 torch.distributed process_group 的封装，
    提供低比特通信原语。

    使用方式:
        # 方式1: 使用 lowbit backend
        bitscom.init()
        dist.init_process_group(backend="lowbit")
        group = LowBitGroup(bitwidth=4)
        group.all_reduce(tensor)

        # 方式2: 使用已有 process group
        dist.init_process_group(backend="nccl")
        group = LowBitGroup(bitwidth=4, process_group=dist.group.WORLD)
        group.all_reduce(tensor)
    """

    def __init__(
        self,
        bitwidth: int = 4,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Args:
            bitwidth: 量化比特宽度 (1, 2, 4, 8, 12, 16)
            process_group: 使用的 process group，None 表示使用默认 group
        """
        assert 1 <= bitwidth <= 16, f"bitwidth must be in [1, 16], got {bitwidth}"
        self.bitwidth = bitwidth
        self.pg = process_group or dist.distributed_c10d._get_default_group()

    @property
    def rank(self) -> int:
        return dist.get_rank(self.pg)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.pg)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        """
        低比特 all_reduce。

        当使用 lowbit backend 时，底层 C++ 会自动进行
        pack -> NCCL allreduce -> unpack 的流程。
        """
        work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool = False,
    ):
        """低比特 all_gather。"""
        work = dist.all_gather(tensor_list, tensor, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        """低比特 reduce_scatter。"""
        work = dist.reduce_scatter(
            output, input_list, op=op, group=self.pg, async_op=True
        )
        if not async_op:
            work.wait()
            return None
        return work

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        async_op: bool = False,
    ):
        """broadcast（通常不需要压缩）。"""
        work = dist.broadcast(tensor, src=src, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work
