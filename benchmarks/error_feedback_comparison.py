import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from bitscom.quantization import dequantize_tensor, quantize_tensor


@dataclass
class RunResult:
    losses: List[float]


def _quant_roundtrip(
    x: torch.Tensor,
    bitwidth: int,
    residual: torch.Tensor | None,
    use_ef: bool,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if use_ef:
        if residual is None or residual.shape != x.shape or residual.device != x.device:
            residual = torch.zeros_like(x, dtype=torch.float32)
        corrected = x.to(torch.float32) + residual
        q, scale = quantize_tensor(corrected, bitwidth=bitwidth)
        x_hat = dequantize_tensor(q, scale, dtype=torch.float32, device=x.device)
        new_residual = corrected - x_hat
        return x_hat.to(dtype=x.dtype), new_residual

    q, scale = quantize_tensor(x, bitwidth=bitwidth)
    x_hat = dequantize_tensor(q, scale, dtype=torch.float32, device=x.device)
    return x_hat.to(dtype=x.dtype), residual


def run_ef_benefit_curve(
    device: torch.device,
    bitwidth: int,
    steps: int,
    vec_size: int,
    seed: int,
) -> List[Tuple[int, float, float]]:
    torch.manual_seed(seed)

    true_cum = torch.zeros(vec_size, device=device, dtype=torch.float32)
    noef_cum = torch.zeros_like(true_cum)
    ef_cum = torch.zeros_like(true_cum)
    ef_residual = torch.zeros_like(true_cum)

    rows = []
    for step in range(1, steps + 1):
        g = torch.randn(vec_size, device=device, dtype=torch.float32) * 0.05
        true_cum.add_(g)

        g_noef_hat, _ = _quant_roundtrip(g, bitwidth, None, use_ef=False)
        noef_cum.add_(g_noef_hat)

        g_ef_hat, ef_residual = _quant_roundtrip(g, bitwidth, ef_residual, use_ef=True)
        ef_cum.add_(g_ef_hat)

        denom = true_cum.norm().item() + 1e-12
        noef_rel = (noef_cum - true_cum).norm().item() / denom
        ef_rel = (ef_cum - true_cum).norm().item() / denom
        rows.append((step, noef_rel, ef_rel))

    return rows


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_toy_batch(batch_size: int, in_dim: int, num_classes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, in_dim, device=device)
    # Construct a learnable-ish target mapping for stable curves.
    w = torch.linspace(-1.0, 1.0, in_dim, device=device).unsqueeze(1).repeat(1, num_classes)
    logits = x @ w
    y = torch.argmax(logits, dim=1)
    return x, y


def run_training(
    device: torch.device,
    bitwidth: int,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    use_ef: bool,
) -> RunResult:
    torch.manual_seed(seed)
    model = TinyMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    residuals: Dict[int, torch.Tensor] = {}
    losses: List[float] = []

    for _ in range(steps):
        x, y = _make_toy_batch(batch_size=batch_size, in_dim=64, num_classes=10, device=device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                key = p.data_ptr()
                resid = residuals.get(key)
                g_hat, new_resid = _quant_roundtrip(
                    p.grad,
                    bitwidth=bitwidth,
                    residual=resid,
                    use_ef=use_ef,
                )
                p.grad.copy_(g_hat)
                if use_ef and new_resid is not None:
                    residuals[key] = new_resid

        optimizer.step()
        losses.append(float(loss.item()))

    return RunResult(losses=losses)


def save_rows_csv(path: str, header: List[str], rows: List[Tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curves(
    out_png: str,
    ef_rows: List[Tuple[int, float, float]],
    loss_noef: List[float],
    loss_ef: List[float],
    bitwidth: int,
    device: torch.device,
) -> None:
    steps = [r[0] for r in ef_rows]
    noef_err = [r[1] for r in ef_rows]
    ef_err = [r[2] for r in ef_rows]

    train_steps = list(range(1, len(loss_noef) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(steps, noef_err, label="No EF", color="#d62728")
    axes[0].plot(steps, ef_err, label="With EF", color="#2ca02c")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Relative Cumulative Error")
    axes[0].set_title("EF Benefit Curve (Communication Error)")
    axes[0].grid(True, ls=":")
    axes[0].legend()

    axes[1].plot(train_steps, loss_noef, label="Train Loss (No EF)", color="#d62728")
    axes[1].plot(train_steps, loss_ef, label="Train Loss (With EF)", color="#2ca02c")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Curve Comparison")
    axes[1].grid(True, ls=":")
    axes[1].legend()

    fig.suptitle(f"Error Feedback Comparison | bitwidth={bitwidth} | device={device}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EF vs No-EF with curves")
    parser.add_argument("--bitwidth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--vec-size", type=int, default=1 << 20)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="benchmarks/outputs")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for requested device")

    os.makedirs(args.out_dir, exist_ok=True)

    ef_rows = run_ef_benefit_curve(
        device=device,
        bitwidth=args.bitwidth,
        steps=args.steps,
        vec_size=args.vec_size,
        seed=args.seed,
    )

    run_noef = run_training(
        device=device,
        bitwidth=args.bitwidth,
        steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        use_ef=False,
    )
    run_ef = run_training(
        device=device,
        bitwidth=args.bitwidth,
        steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        use_ef=True,
    )

    ef_csv = os.path.join(args.out_dir, f"ef_benefit_curve_bw{args.bitwidth}.csv")
    train_csv = os.path.join(args.out_dir, f"ef_training_curve_bw{args.bitwidth}.csv")
    out_png = os.path.join(args.out_dir, f"ef_comparison_bw{args.bitwidth}.png")

    save_rows_csv(ef_csv, ["step", "noef_rel_error", "ef_rel_error"], ef_rows)
    train_rows = [(i + 1, run_noef.losses[i], run_ef.losses[i]) for i in range(len(run_noef.losses))]
    save_rows_csv(train_csv, ["step", "loss_noef", "loss_ef"], train_rows)

    plot_curves(
        out_png=out_png,
        ef_rows=ef_rows,
        loss_noef=run_noef.losses,
        loss_ef=run_ef.losses,
        bitwidth=args.bitwidth,
        device=device,
    )

    print(f"Saved: {ef_csv}")
    print(f"Saved: {train_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
