"""
High-level API for low-bit distributed communication.
"""

import torch
import torch.distributed as dist
from typing import Optional, List

from .quantization import (
    SUPPORTED_BITWIDTHS,
    CompressedTensor,
    compress_tensor,
    decompress_tensor,
    dequantize_tensor,
    pack_lowbit,
    quantize_tensor,
    roundtrip_tensor,
    unpack_lowbit,
)


class _ImmediateWork:
    """Simple Work-like object for sync-completed Python collectives."""

    def wait(self):
        return True


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
        simulate_quantization: bool = False,
        stochastic_rounding: bool = False,
    ):
        """
        Args:
            bitwidth: 量化比特宽度 (1, 2, 4, 8, 12, 16)
            process_group: 使用的 process group，None 表示使用默认 group
            simulate_quantization: 使用非 lowbit backend 时，
                在通信前做一次量化-反量化模拟
            stochastic_rounding: 量化时使用随机舍入（默认关闭）
        """
        if bitwidth not in SUPPORTED_BITWIDTHS:
            raise ValueError(
                f"bitwidth must be one of {SUPPORTED_BITWIDTHS}, got {bitwidth}"
            )
        self.bitwidth = bitwidth
        self.pg = process_group or dist.distributed_c10d._get_default_group()
        self.simulate_quantization = simulate_quantization
        self.stochastic_rounding = stochastic_rounding

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
        if self._should_use_lowbit_path(op):
            self._lowbit_allreduce_via_alltoall(tensor)
            if async_op:
                return _ImmediateWork()
            return None

        if self.simulate_quantization:
            tensor.copy_(roundtrip_tensor(tensor, self.bitwidth))
        work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
        if not async_op:
            work.wait()
            return None
        return work

    def _should_use_lowbit_path(self, op: dist.ReduceOp) -> bool:
        return (
            self.bitwidth < 8
            and self.world_size > 1
            and op == dist.ReduceOp.SUM
        )

    def _lowbit_allreduce_via_alltoall(self, tensor: torch.Tensor) -> None:
        flat = tensor.contiguous().view(-1)
        world_size = self.world_size

        if flat.numel() % world_size != 0:
            raise ValueError(
                "lowbit all-reduce requires tensor.numel() divisible by world_size; "
                f"got numel={flat.numel()} world_size={world_size}"
            )

        shard_len = flat.numel() // world_size
        q, scale = quantize_tensor(
            flat,
            self.bitwidth,
            stochastic_rounding=self.stochastic_rounding,
        )
        q_shards = list(q.split(shard_len))

        send_packed = [pack_lowbit(shard, self.bitwidth)[0] for shard in q_shards]
        recv_packed = [torch.empty_like(send_packed[0]) for _ in range(world_size)]
        dist.all_to_all(recv_packed, send_packed, group=self.pg)

        scale_tensor = torch.tensor([scale], dtype=torch.float32, device=flat.device)
        send_scales = [scale_tensor.clone() for _ in range(world_size)]
        recv_scales = [torch.empty_like(scale_tensor) for _ in range(world_size)]
        dist.all_to_all(recv_scales, send_scales, group=self.pg)

        local_sum = torch.zeros(shard_len, dtype=torch.float32, device=flat.device)
        for src_rank in range(world_size):
            q_part = unpack_lowbit(recv_packed[src_rank], self.bitwidth, shard_len)
            fp_part = dequantize_tensor(
                q_part,
                float(recv_scales[src_rank].item()),
                dtype=torch.float32,
                device=flat.device,
            )
            
            local_sum.add_(fp_part)

        q_reduced, reduced_scale = quantize_tensor(
            local_sum,
            self.bitwidth,
            stochastic_rounding=self.stochastic_rounding,
        )
        packed_reduced, _ = pack_lowbit(q_reduced, self.bitwidth)

        gathered_packed = [torch.empty_like(packed_reduced) for _ in range(world_size)]
        dist.all_gather(gathered_packed, packed_reduced, group=self.pg)

        reduced_scale_tensor = torch.tensor(
            [reduced_scale], dtype=torch.float32, device=flat.device
        )
        gathered_scales = [torch.empty_like(reduced_scale_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_scales, reduced_scale_tensor, group=self.pg)

        out_shards = []
        for rank_idx in range(world_size):
            q_shard = unpack_lowbit(gathered_packed[rank_idx], self.bitwidth, shard_len)
            fp_shard = dequantize_tensor(
                q_shard,
                float(gathered_scales[rank_idx].item()),
                dtype=torch.float32,
                device=flat.device,
            )
            out_shards.append(fp_shard)

        restored = torch.cat(out_shards, dim=0).view_as(tensor)
        tensor.copy_(restored.to(dtype=tensor.dtype))

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool = False,
    ):
        """低比特 all_gather。"""
        if self.simulate_quantization:
            tensor.copy_(roundtrip_tensor(tensor, self.bitwidth))
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
        if self.simulate_quantization:
            for t in input_list:
                t.copy_(roundtrip_tensor(t, self.bitwidth))
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

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """将浮点 tensor 压缩为低比特打包表示。"""
        return compress_tensor(tensor, self.bitwidth)

    def decompress(
        self,
        compressed: CompressedTensor,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """将打包表示解压回浮点 tensor。"""
        return decompress_tensor(compressed, dtype=dtype, device=device)
