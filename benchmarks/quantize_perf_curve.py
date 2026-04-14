import argparse
import csv
import os
import statistics
import time

import matplotlib.pyplot as plt
import torch

from bitscom.quantization import quantize_tensor


def _measure_cpu(x: torch.Tensor, bitwidth: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        quantize_tensor(x, bitwidth, stochastic_rounding=False)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        quantize_tensor(x, bitwidth, stochastic_rounding=False)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def _measure_gpu(x: torch.Tensor, bitwidth: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        quantize_tensor(x, bitwidth, stochastic_rounding=False)
    torch.cuda.synchronize(x.device)

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(x.device)
        t0 = time.perf_counter()
        quantize_tensor(x, bitwidth, stochastic_rounding=False)
        torch.cuda.synchronize(x.device)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def _iters_for_size(n: int) -> int:
    if n <= (1 << 14):
        return 200
    if n <= (1 << 18):
        return 120
    if n <= (1 << 21):
        return 60
    return 30


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot compare CPU vs GPU curves")

    sizes = [1 << k for k in range(args.min_pow2, args.max_pow2 + 1)]
    device = torch.device(args.device)

    rows = []
    for n in sizes:
        x_cpu = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.float32)
        x_gpu = x_cpu.to(device)

        iters = _iters_for_size(n)
        warmup = min(args.warmup, max(5, iters // 4))

        cpu_ms = _measure_cpu(x_cpu, args.bitwidth, warmup, iters)
        gpu_ms = _measure_gpu(x_gpu, args.bitwidth, warmup, iters)

        cpu_melems = (n / 1e6) / (cpu_ms / 1e3)
        gpu_melems = (n / 1e6) / (gpu_ms / 1e3)

        rows.append(
            {
                "numel": n,
                "cpu_ms": cpu_ms,
                "gpu_ms": gpu_ms,
                "speedup_gpu_vs_cpu": cpu_ms / gpu_ms,
                "cpu_melems_per_s": cpu_melems,
                "gpu_melems_per_s": gpu_melems,
            }
        )
        print(
            f"numel={n:>9d} cpu={cpu_ms:>8.4f} ms gpu={gpu_ms:>8.4f} ms "
            f"speedup={cpu_ms / gpu_ms:>6.2f}x"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"quantize_curve_bw{args.bitwidth}.csv")
    png_path = os.path.join(args.out_dir, f"quantize_curve_bw{args.bitwidth}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "numel",
                "cpu_ms",
                "gpu_ms",
                "speedup_gpu_vs_cpu",
                "cpu_melems_per_s",
                "gpu_melems_per_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    x = [r["numel"] for r in rows]
    cpu_ms = [r["cpu_ms"] for r in rows]
    gpu_ms = [r["gpu_ms"] for r in rows]
    speedup = [r["speedup_gpu_vs_cpu"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x, cpu_ms, marker="o", label="CPU (original path)")
    axes[0].plot(x, gpu_ms, marker="o", label="GPU (CUDA kernel path)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Tensor numel")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title(f"Quantize Call Latency (bitwidth={args.bitwidth})")
    axes[0].grid(True, which="both", ls=":")
    axes[0].legend()

    axes[1].plot(x, speedup, marker="o", color="#2ca02c")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Tensor numel")
    axes[1].set_ylabel("Speedup (CPU / GPU)")
    axes[1].set_title("GPU Speedup Curve")
    axes[1].grid(True, which="both", ls=":")

    fig.suptitle("bitscom quantize_tensor Performance Curve")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU original path vs GPU CUDA path for quantize_tensor"
    )
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--min-pow2", type=int, default=10)
    parser.add_argument("--max-pow2", type=int, default=23)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
