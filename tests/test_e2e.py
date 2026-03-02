"""
端到端测试：验证 Python -> C++ -> NCCL 调用链路。

运行方式（2 GPU）:
    torchrun --nproc_per_node=2 tests/test_e2e.py
"""

import os
import torch
import torch.distributed as dist
import bitscom


def test_all_reduce():
    """测试 all_reduce 基本正确性。"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    tensor = torch.ones(1024, device=device) * (rank + 1)
    print(f"[Rank {rank}] before all_reduce: sum={tensor.sum().item()}")

    group = bitscom.LowBitGroup(bitwidth=4)
    group.all_reduce(tensor)

    # 占位实现下 allreduce SUM: 每个元素 = 1+2+...+world_size
    expected_val = sum(range(1, world_size + 1))
    expected_sum = expected_val * 1024
    actual_sum = tensor.sum().item()
    print(f"[Rank {rank}] after all_reduce: sum={actual_sum}, expected={expected_sum}")
    assert abs(actual_sum - expected_sum) < 1e-3, \
        f"all_reduce mismatch: {actual_sum} != {expected_sum}"
    print(f"[Rank {rank}] all_reduce PASSED")


def test_broadcast():
    """测试 broadcast 基本正确性。"""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        tensor = torch.arange(100, dtype=torch.float32, device=device)
    else:
        tensor = torch.zeros(100, dtype=torch.float32, device=device)

    group = bitscom.LowBitGroup(bitwidth=4)
    group.broadcast(tensor, src=0)

    expected = torch.arange(100, dtype=torch.float32, device=device)
    assert torch.allclose(tensor, expected), \
        f"broadcast mismatch on rank {rank}"
    print(f"[Rank {rank}] broadcast PASSED")


def main():
    # 1) 注册 lowbit backend
    bitscom.init()

    # 2) 初始化 process group
    dist.init_process_group(backend="lowbit")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] initialized, world_size={world_size}")

    # 3) 运行测试
    test_all_reduce()
    test_broadcast()

    # 4) 清理
    dist.destroy_process_group()
    print(f"[Rank {rank}] all tests passed!")


if __name__ == "__main__":
    main()
