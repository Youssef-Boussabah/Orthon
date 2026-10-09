"""Small deterministic CUDA workloads for Nsight profiling.

The script avoids datasets and external profilers. Run it directly for a quick
workload check, or wrap it with Nsight Systems / Nsight Compute.
"""

from __future__ import annotations

import argparse
import random

import orthon


from benchmarks._support.environment import cuda_available, synchronizer
from benchmarks._support.timing import hot_loop
from benchmarks._support.results import document, emit
from benchmarks._support.cli import add_options

_PROFILE_CASES = []


def require_cuda(device):
    if not cuda_available(device):
        raise SystemExit('no usable Orthon CUDA device')


def sync():
    synchronizer('cuda:0')()


def timed_loop(fn, warmup, iterations):
    hot_loop(fn,sync,warmup,iterations)
    return dict(warmup=warmup,iterations=iterations)


def print_summary(workload,config,metrics):
    _PROFILE_CASES.append(dict(name=workload,implementation='orthon',device='cuda:0',
        dtype='float32',mode='profile',shape=config,notes='fixed hot loop; Nsight owns kernel timing',**metrics))


def run_conv2d(warmup: int, iterations: int) -> None:
    batch, cin, cout, height, width, kernel, padding = 8, 8, 16, 32, 32, 3, 1
    x = orthon.full([batch, cin, height, width], 0.05).to("cuda")
    weight = orthon.full([cout, cin, kernel, kernel], 0.02).to("cuda")
    bias = orthon.full([cout], 0.01).to("cuda")
    x.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    def step() -> None:
        x.zero_grad()
        weight.zero_grad()
        bias.zero_grad()
        out = orthon.conv2d(x, weight, bias, 1, 1, padding, padding)
        out.backward()

    metrics = timed_loop(step, warmup, iterations)
    print_summary("conv2d", {
        "device": "cuda",
        "input": [batch, cin, height, width],
        "weight": [cout, cin, kernel, kernel],
        "padding": padding,
        "operation": "Conv2d forward + backward",
    }, metrics)


def run_fusion(warmup: int, iterations: int) -> None:
    m, k, n = 128, 256, 128
    x = orthon.full([m, k], 0.05).to("cuda")
    weight = orthon.full([k, n], 0.02).to("cuda")
    bias = orthon.full([n], 0.10).to("cuda")

    def step() -> None:
        orthon.relu(orthon.matmul(x, weight) + bias)
        orthon.fused_linear_relu(x, weight, bias)

    metrics = timed_loop(step, warmup, iterations)
    print_summary("fusion", {
        "device": "cuda",
        "x": [m, k],
        "weight": [k, n],
        "bias": [n],
        "operation": "separate matmul+bias+relu, then fused_linear_relu",
    }, metrics)


def run_training(warmup: int, iterations: int) -> None:
    from orthon.nn import Conv2d, Linear, MaxPool2d, ReLU
    from orthon.optim import SGD

    random.seed(0)
    orthon.manual_seed(0)   # The framework RNG is separate from Python's global random.
    batch, cin, height, width = 4, 3, 16, 16
    cout, kernel, pool_kernel, classes = 8, 3, 2, 10
    flat = cout * (height // pool_kernel) * (width // pool_kernel)

    conv = Conv2d(cin, cout, kernel, padding=1).to("cuda")
    relu = ReLU()
    pool = MaxPool2d(pool_kernel, stride=pool_kernel)
    fc = Linear(flat, classes).to("cuda")
    optimizer = SGD(list(conv.parameters()) + list(fc.parameters()), lr=0.01)

    x = orthon.full([batch, cin, height, width], 0.05).to("cuda")
    targets = [i % classes for i in range(batch)]

    last_loss = None

    def step() -> None:
        nonlocal last_loss
        optimizer.zero_grad()
        out = conv(x)
        out = relu(out)
        out = pool(out)
        out = orthon.reshape(out, [batch, flat])
        out = fc(out)
        loss = orthon.cross_entropy(out, targets)
        loss.backward()
        optimizer.step()
        last_loss = loss

    metrics = timed_loop(step, warmup, iterations)
    sync()
    final_loss = last_loss.item() if last_loss is not None else float("nan")
    print_summary("training", {
        "device": "cuda",
        "input": [batch, cin, height, width],
        "conv_weight": [cout, cin, kernel, kernel],
        "linear_weight": [flat, classes],
        "optimizer": "SGD(lr=0.01)",
        "final_loss": f"{final_loss:.6f}",
    }, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile deterministic Orthon CUDA workloads.")
    parser.add_argument("--workload", choices=["conv2d", "training", "fusion"], required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    add_options(parser,timing=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.warmup,args.iterations=1,2
    _PROFILE_CASES.clear()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    require_cuda(args.device)
    if args.workload == "conv2d":
        run_conv2d(args.warmup, args.iterations)
    elif args.workload == "fusion":
        run_fusion(args.warmup, args.iterations)
    else:
        run_training(args.warmup, args.iterations)

    emit(document('profile_cuda_workload',_PROFILE_CASES,dict(vars(args),representative=not args.quick),mode='profile'),args.json)

if __name__ == "__main__":
    main()
