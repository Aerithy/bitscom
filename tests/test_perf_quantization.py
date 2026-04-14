import os
import time

import pytest
import torch

from bitscom.quantization import compress_tensor, decompress_tensor, roundtrip_tensor


pytestmark = pytest.mark.performance


def _run_benchmark(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


@pytest.mark.skipif(
    os.getenv("BITSCOM_RUN_PERF", "0") != "1",
    reason="set BITSCOM_RUN_PERF=1 to run performance tests",
)
def test_quantization_roundtrip_benchmark_prints_metrics():
    x = torch.randn(1_000_000, dtype=torch.float32)

    rt_time = _run_benchmark(lambda: roundtrip_tensor(x, bitwidth=4))
    clone_time = _run_benchmark(lambda: x.clone())

    assert rt_time > 0
    assert clone_time > 0

    # Keep this test stable across machines: enforce only a loose upper bound.
    assert rt_time < clone_time * 200


@pytest.mark.skipif(
    os.getenv("BITSCOM_RUN_PERF", "0") != "1",
    reason="set BITSCOM_RUN_PERF=1 to run performance tests",
)
def test_compression_ratio_benchmark():
    x = torch.randn(500_000, dtype=torch.float32)

    compressed = compress_tensor(x, bitwidth=4)
    restored = decompress_tensor(compressed)

    fp32_bytes = x.numel() * x.element_size()
    packed_bytes = compressed.packed_bytes

    assert restored.shape == x.shape
    assert packed_bytes < fp32_bytes
