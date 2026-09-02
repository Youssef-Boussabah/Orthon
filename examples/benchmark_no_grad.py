"""Benchmark eval forward with grad enabled versus orthon.no_grad()."""

from __future__ import annotations

import argparse
import time

import orthon
from orthon.nn import Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential


class SmallCIFAR(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, channels, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        self.classifier = Linear(channels * 16 * 16, 10)
        self._flat_features = channels * 16 * 16

    def forward(self, x):
        out = self.features(x)
        out = orthon.reshape(out, [x.shape[0], self._flat_features])
        return self.classifier(out)


def synchronize(device: str) -> None:
    if device == "cuda":
        orthon.cuda_synchronize()


def measure(fn, device: str, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iterations


def run_mlp(device: str, batch: int, hidden: int, warmup: int, iterations: int):
    model = Sequential(Linear(784, hidden), ReLU(), Linear(hidden, 10)).to(device)
    model.eval()
    x = orthon.full([batch, 784], 0.01).to(device)

    def grad_enabled():
        return model(x)

    def no_grad():
        with orthon.no_grad():
            return model(x)

    return (
        measure(grad_enabled, device, warmup, iterations),
        measure(no_grad, device, warmup, iterations),
    )


def run_cnn(device: str, batch: int, channels: int, warmup: int, iterations: int):
    model = SmallCIFAR(channels).to(device)
    model.eval()
    x = orthon.full([batch, 3, 32, 32], 0.01).to(device)

    def grad_enabled():
        return model(x)

    def no_grad():
        with orthon.no_grad():
            return model(x)

    return (
        measure(grad_enabled, device, warmup, iterations),
        measure(no_grad, device, warmup, iterations),
    )


def fmt(seconds: float) -> str:
    return f"{seconds * 1000.0:8.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark orthon.no_grad eval forward.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--channels", type=int, default=4)
    args = parser.parse_args()

    if args.device == "cuda" and not orthon.cuda_enabled():
        raise RuntimeError("Orthon CUDA is unavailable in this build")

    rows = [
        ("mnist_mlp", run_mlp(args.device, args.batch, args.hidden, args.warmup, args.iterations)),
        ("cifar_cnn", run_cnn(args.device, args.batch, args.channels, args.warmup, args.iterations)),
    ]

    print("no_grad eval-forward benchmark")
    print("workload      grad enabled   no_grad      speedup")
    for name, (enabled, disabled) in rows:
        speedup = enabled / disabled if disabled > 0 else 0.0
        print(f"{name:<12} {fmt(enabled)}  {fmt(disabled)}  {speedup:6.2f}x")


if __name__ == "__main__":
    main()
