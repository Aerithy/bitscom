import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_single(script: Path, bitwidth: int, device: str, out_dir: Path, steps: int, train_steps: int, vec_size: int, batch_size: int) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--bitwidth",
        str(bitwidth),
        "--steps",
        str(steps),
        "--train-steps",
        str(train_steps),
        "--vec-size",
        str(vec_size),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-bitwidth EF comparison summary")
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--vec-size", type=int, default=1 << 19)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent
    single_script = root / "error_feedback_comparison.py"

    for bw in args.bitwidths:
        run_single(
            script=single_script,
            bitwidth=bw,
            device=args.device,
            out_dir=out_dir,
            steps=args.steps,
            train_steps=args.train_steps,
            vec_size=args.vec_size,
            batch_size=args.batch_size,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for bw in args.bitwidths:
        ef_path = out_dir / f"ef_benefit_curve_bw{bw}.csv"
        train_path = out_dir / f"ef_training_curve_bw{bw}.csv"

        ef_df = pd.read_csv(ef_path)
        train_df = pd.read_csv(train_path)

        ratio = ef_df["noef_rel_error"] / (ef_df["ef_rel_error"] + 1e-12)
        axes[0].plot(ef_df["step"], ratio, label=f"bw={bw}")

        gain = train_df["loss_noef"] - train_df["loss_ef"]
        smooth = gain.rolling(window=5, min_periods=1).mean()
        axes[1].plot(train_df["step"], smooth, label=f"bw={bw}")

    axes[0].set_title("EF Error Reduction Factor")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("noEF_error / EF_error")
    axes[0].grid(True, ls=":")
    axes[0].legend()

    axes[1].set_title("EF Training Loss Gain")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("loss_noEF - loss_EF")
    axes[1].grid(True, ls=":")
    axes[1].legend()

    fig.suptitle(f"Error Feedback Summary Across Bitwidths ({args.device})")
    fig.tight_layout()

    out_png = out_dir / "ef_multibitwidth_summary.png"
    fig.savefig(out_png, dpi=180)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
