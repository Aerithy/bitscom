import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import bitscom


pytestmark = pytest.mark.integration


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _init_single_rank_lowbit_pg() -> str:
    bitscom.init()
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    init_path = tmp.name
    dist.init_process_group(
        backend="lowbit",
        init_method=f"file://{init_path}",
        rank=0,
        world_size=1,
    )
    return init_path


def _cleanup_pg(init_path: str) -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    if os.path.exists(init_path):
        os.remove(init_path)


def _run_small_training_loop(
    model: nn.Module,
    group: bitscom.LowBitGroup,
    data_iter,
    *,
    steps: int,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    samples = 0

    for _ in range(steps):
        x, target = next(data_iter)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, target)
        assert torch.isfinite(loss).item(), "loss must be finite"

        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                group.all_reduce(p.grad, async_op=False)
                p.grad.div_(group.world_size)
        optimizer.step()
        torch.cuda.synchronize(device)

        step_times.append(time.perf_counter() - step_start)
        samples += int(x.size(0))

    total_time = sum(step_times)
    metrics = {
        "steps": steps,
        "avg_step_time_ms": (total_time / steps) * 1000.0,
        "throughput_samples_per_s": samples / max(total_time, 1e-9),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
    }
    print(f"[single-gpu-e2e] metrics={metrics}")
    return metrics


def _infinite_synth_loader(batch_size: int):
    while True:
        x = torch.randn(batch_size, 3, 32, 32)
        target = torch.randint(0, 10, (batch_size,))
        yield x, target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="single-GPU e2e needs CUDA")
def test_single_gpu_lowbit_training_e2e():
    torch.cuda.set_device(0)
    init_path = _init_single_rank_lowbit_pg()

    try:
        device = torch.device("cuda:0")
        model = TinyResNet(num_classes=10).to(device)
        group = bitscom.LowBitGroup(bitwidth=4)

        before = model.fc.weight.detach().clone()

        metrics = _run_small_training_loop(
            model,
            group,
            _infinite_synth_loader(batch_size=16),
            steps=5,
            device=device,
        )

        after = model.fc.weight.detach().clone()
        assert not torch.allclose(before, after), "model weights should update"
        assert metrics["avg_step_time_ms"] > 0
        assert metrics["throughput_samples_per_s"] > 0
        assert metrics["peak_memory_mb"] > 0
    finally:
        _cleanup_pg(init_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="single-GPU e2e needs CUDA")
def test_single_gpu_lowbit_training_cifar10_small_steps():
    torchvision = pytest.importorskip("torchvision")
    transforms = torchvision.transforms

    root = os.path.join(os.path.dirname(__file__), "..", ".cache", "cifar10")
    allow_download = os.getenv("BITSCOM_ALLOW_DOWNLOAD", "0") == "1"

    try:
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            transform=transforms.ToTensor(),
            download=allow_download,
        )
    except RuntimeError as exc:
        pytest.skip(f"CIFAR10 not available locally and download disabled: {exc}")

    subset_len = min(len(dataset), 320)
    subset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    it = iter(loader)

    torch.cuda.set_device(0)
    init_path = _init_single_rank_lowbit_pg()

    try:
        device = torch.device("cuda:0")
        model = TinyResNet(num_classes=10).to(device)
        group = bitscom.LowBitGroup(bitwidth=4)

        before = model.fc.weight.detach().clone()
        metrics = _run_small_training_loop(
            model,
            group,
            it,
            steps=10,
            device=device,
        )
        after = model.fc.weight.detach().clone()

        assert not torch.allclose(before, after), "model weights should update"
        assert metrics["avg_step_time_ms"] > 0
        assert metrics["throughput_samples_per_s"] > 0
        assert metrics["peak_memory_mb"] > 0
    finally:
        _cleanup_pg(init_path)
