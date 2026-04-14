# bitscom

bitscom is a low-bit distributed communication library for PyTorch.
It provides:

- A custom `torch.distributed` backend named `lowbit`
- A Python wrapper API `LowBitGroup`
- A Python fallback quantization/bit-packing module for development and tests

## What This Repository Does

Core objective:

- Reduce communication traffic in distributed training by sending quantized low-bit representations instead of full-precision tensors.

Current architecture:

- C++ backend (`ProcessGroupLowBit`) wraps NCCL process group calls.
- Python API (`bitscom.LowBitGroup`) exposes collective operations.
- Python quantization module (`bitscom.quantization`) provides functional compression/decompression logic that can be tested without GPUs.

## Implemented Features

- Supported quantization bitwidths: `1, 2, 4, 8, 12, 16`
- Tensor quantize/dequantize utilities
- Bit-packing and unpacking for the supported bitwidths
- CUDA quantization/pack/unpack fast path for `1/2/4/8` bit
- Stochastic rounding support in quantization (CPU + CUDA)
- `LowBitGroup.compress()` and `LowBitGroup.decompress()` helper APIs
- Optional simulation mode (`simulate_quantization=True`) for validating quantization effect before collectives
- Python low-bit all-reduce pipeline for `<8bit` (`all-to-all -> local reduce -> all-gather`)
- Error-feedback in first-stage quantization for C++ backend low-bit allreduce path
- Explicit backend option registration: `bitwidth` and `error_feedback`
- Robust backend registration with explicit error when C++ extension is missing

## Gaps That Still Remain

The following items are still pending and marked as future work:

- In-backend low-bit paths for `allgather` and `reduce_scatter` (currently forwarded to NCCL)
- End-to-end multi-node benchmark suite and scaling analysis
- Stronger C++/Python path unification to reduce duplicated quantization logic

## Build

Editable install (recommended for development):

```bash
pip install -e .
```

If using the tested conda environment in this repo:

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pip install -e . --no-build-isolation
```

## Test

Unit tests and non-distributed integration-safe tests:

```bash
pytest -q
```

Quantization-focused tests (including CUDA path and stochastic rounding checks):

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_quantization.py
```

Run integration/e2e with distributed launcher:

```bash
torchrun --nproc_per_node=2 tests/test_e2e.py
```

Run single-GPU end-to-end training tests (no multi-GPU required):

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py
```

The single-GPU e2e test prints performance metrics:

- `avg_step_time_ms`
- `throughput_samples_per_s`
- `peak_memory_mb`

To run the CIFAR10 small-step variant (10 steps), allow dataset download:

```bash
BITSCOM_ALLOW_DOWNLOAD=1 /home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py -k cifar10
```

## Performance Test

Performance tests are included but disabled by default to keep CI stable.

```bash
BITSCOM_RUN_PERF=1 pytest -q -m performance
```

### Quantization CPU vs GPU Curve

Generate a latency/speedup curve for `quantize_tensor` (CPU original path vs CUDA path):

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/quantize_perf_curve.py --bitwidth 4 --min-pow2 10 --max-pow2 23
```

Outputs:

- `benchmarks/outputs/quantize_curve_bw4.csv`
- `benchmarks/outputs/quantize_curve_bw4.png`

Preview:

![Quantization CPU vs GPU curve](benchmarks/outputs/quantize_curve_bw4.png)

### Error-Feedback Curve (Single Bitwidth)

Generate communication-error and training-loss comparison curves for EF vs no-EF:

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/error_feedback_comparison.py --bitwidth 2 --steps 180 --train-steps 140 --vec-size 524288 --batch-size 256 --device cuda:0
```

Outputs:

- `benchmarks/outputs/ef_benefit_curve_bw2.csv`
- `benchmarks/outputs/ef_training_curve_bw2.csv`
- `benchmarks/outputs/ef_comparison_bw2.png`

Preview:

![Error-feedback single-bitwidth comparison](benchmarks/outputs/ef_comparison_bw2.png)

### Error-Feedback Summary (Multi-Bitwidth)

Generate one summary figure across multiple bitwidths (default `2/4/8`):

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/error_feedback_multibw.py --bitwidths 2 4 8 --device cuda:0 --steps 160 --train-steps 120 --vec-size 524288 --batch-size 256
```

Key output:

- `benchmarks/outputs/ef_multibitwidth_summary.png`

Preview:

![Error-feedback multi-bitwidth summary](benchmarks/outputs/ef_multibitwidth_summary.png)

## Backend Registration Options

Use explicit backend options when registering:

```python
import bitscom

# Register lowbit backend with explicit options.
bitscom.init(bitwidth=4, error_feedback=True)
```

Notes:

- `error_feedback=True` currently applies to first-stage quantization in low-bit allreduce.
- Re-registering backend with different options in the same process raises an error by design.

## Project Layout

- `cpp/`: C++ backend and bindings
- `python/bitscom/`: Python package
- `tests/`: unit tests, integration tests, and performance tests
