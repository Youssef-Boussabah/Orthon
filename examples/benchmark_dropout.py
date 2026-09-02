import argparse
import time

import orthon
from orthon import nn


def _sync(device):
    if device == "cuda":
        orthon.cuda_synchronize()


def _time(fn, device, iterations):
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iterations


def main():
    parser = argparse.ArgumentParser(description="Benchmark Orthon Dropout.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--shape", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not orthon.cuda_enabled():
        raise RuntimeError("CUDA device requested but Orthon was built without CUDA")

    x = orthon.ones(args.shape)
    if args.device == "cuda":
        x = x.to("cuda")
    drop = nn.Dropout(args.p, seed=123)

    for _ in range(args.warmup):
        drop(x)
    fwd = _time(lambda: drop(x), args.device, args.iterations)

    def forward_backward():
        x_local = orthon.ones(args.shape)
        if args.device == "cuda":
            x_local = x_local.to("cuda")
        x_local.requires_grad_(True)
        drop(x_local).backward()

    for _ in range(max(1, args.warmup // 2)):
        forward_backward()
    bwd = _time(forward_backward, args.device, args.iterations)

    print(
        f"dropout device={args.device} shape={args.shape} p={args.p} "
        f"forward_ms={fwd * 1000.0:.4f} forward_backward_ms={bwd * 1000.0:.4f}"
    )


if __name__ == "__main__":
    main()
