"""Benchmark Orthon Conv2d variants against PyTorch.

The benchmark uses CIFAR-shaped convolution workloads and keeps data creation
outside the timed forward path. Backward timings recreate tensors per iteration
because Orthon builds a fresh autograd graph for each forward pass.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import orthon
import orthon._C as orthon_c

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass(frozen=True)
class ConvConfig:
    name: str
    n: int
    cin: int
    cout: int
    h: int
    w: int
    k: int
    padding: int


CONFIGS = [
    ConvConfig("cifar_conv1", n=8, cin=3, cout=8, h=32, w=32, k=3, padding=1),
    ConvConfig("cifar_conv2", n=8, cin=8, cout=8, h=32, w=32, k=3, padding=1),
    ConvConfig("synthetic", n=16, cin=16, cout=16, h=32, w=32, k=3, padding=1),
    ConvConfig("minivgg_32", n=128, cin=32, cout=32, h=32, w=32, k=3, padding=1),
    ConvConfig("minivgg_64", n=128, cin=64, cout=64, h=16, w=16, k=3, padding=1),
    ConvConfig("minivgg_128", n=128, cin=128, cout=128, h=8, w=8, k=3, padding=1),
]


def deterministic_array(shape: tuple[int, ...], scale: float, offset: int) -> np.ndarray:
    values = (np.arange(np.prod(shape), dtype=np.float32) + offset) % 29
    return ((values - 14.0) * scale).reshape(shape).astype(np.float32)


def make_arrays(cfg: ConvConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = deterministic_array((cfg.n, cfg.cin, cfg.h, cfg.w), scale=0.01, offset=1)
    w = deterministic_array((cfg.cout, cfg.cin, cfg.k, cfg.k), scale=0.02, offset=5)
    b = deterministic_array((cfg.cout,), scale=0.01, offset=7)
    return x, w, b


def make_orthon_tensors(
    x_np: np.ndarray,
    w_np: np.ndarray,
    b_np: np.ndarray,
    *,
    requires_grad: bool = False,
):
    x = orthon.from_numpy(x_np)
    w = orthon.from_numpy(w_np)
    b = orthon.from_numpy(b_np)
    if requires_grad:
        x.requires_grad_(True)
        w.requires_grad_(True)
        b.requires_grad_(True)
    return x, w, b


def measure(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def with_env(name: str, value: str | None, fn):
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        return fn()
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def benchmark_orthon(cfg: ConvConfig, forward_iters: int, backward_iters: int):
    x_np, w_np, b_np = make_arrays(cfg)
    x, w, b = make_orthon_tensors(x_np, w_np, b_np)

    def direct_forward():
        orthon_c._conv2d_direct(x, w, b, 1, 1, cfg.padding, cfg.padding)

    def im2col_forward():
        orthon_c._conv2d_im2col(x, w, b, 1, 1, cfg.padding, cfg.padding)

    direct_forward()
    im2col_forward()
    direct_forward_seconds = measure(direct_forward, forward_iters)
    im2col_forward_seconds = measure(im2col_forward, forward_iters)

    def backward():
        xb, wb, bb = make_orthon_tensors(x_np, w_np, b_np, requires_grad=True)
        out = orthon.conv2d(xb, wb, bb, 1, 1, cfg.padding, cfg.padding)
        out.backward()

    backward()
    backward_seconds = measure(backward, backward_iters)
    return direct_forward_seconds, im2col_forward_seconds, backward_seconds


def benchmark_torch(cfg: ConvConfig, forward_iters: int, backward_iters: int):
    if not HAS_TORCH:
        return None

    x_np, w_np, b_np = make_arrays(cfg)
    x = torch.tensor(x_np)
    w = torch.tensor(w_np)
    b = torch.tensor(b_np)

    def forward():
        F.conv2d(x, w, b, stride=1, padding=cfg.padding)

    forward()
    forward_seconds = measure(forward, forward_iters)

    def backward():
        xb = torch.tensor(x_np, requires_grad=True)
        wb = torch.tensor(w_np, requires_grad=True)
        bb = torch.tensor(b_np, requires_grad=True)
        out = F.conv2d(xb, wb, bb, stride=1, padding=cfg.padding)
        out.sum().backward()

    backward()
    backward_seconds = measure(backward, backward_iters)
    return forward_seconds, backward_seconds


def benchmark_orthon_cuda(cfg: ConvConfig, forward_iters: int, backward_iters: int):
    if not orthon.cuda_enabled():
        return None
    try:
        x_np, w_np, b_np = make_arrays(cfg)
        x, w, b = make_orthon_tensors(x_np, w_np, b_np)
        x = x.to("cuda")
        w = w.to("cuda")
        b = b.to("cuda")
        x.requires_grad_(True)
        w.requires_grad_(True)
        b.requires_grad_(True)
    except RuntimeError:
        return None

    def forward_default():
        orthon.conv2d(x, w, b, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def forward_naive():
        orthon_c._conv2d_cuda_naive(x, w, b, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def forward_optimized():
        orthon_c._conv2d_cuda_optimized(x, w, b, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    forward_default()
    forward_naive()
    forward_optimized()
    default_forward_seconds = measure(forward_default, forward_iters)
    naive_forward_seconds = measure(forward_naive, forward_iters)
    optimized_forward_seconds = measure(forward_optimized, forward_iters)

    h_out = (cfg.h + 2 * cfg.padding - cfg.k) + 1
    w_out = (cfg.w + 2 * cfg.padding - cfg.k) + 1
    grad_output = orthon.ones([cfg.n, cfg.cout, h_out, w_out]).to("cuda")

    def backward_input():
        orthon_c._conv2d_cuda_backward_input(
            x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def backward_input_naive():
        orthon_c._conv2d_cuda_backward_input_naive(
            x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def backward_input_optimized():
        orthon_c._conv2d_cuda_backward_input_optimized(
            x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def backward_input_optimized_v2():
        if hasattr(orthon_c, "_conv2d_cuda_backward_input_optimized_v2"):
            orthon_c._conv2d_cuda_backward_input_optimized_v2(
                x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
            orthon.cuda_synchronize()
        else:
            with_env(
                "ORTHON_CUDA_CONV2D_GRAD_INPUT",
                "optimized_v2",
                backward_input_optimized,
            )

    def backward_weight_naive():
        orthon_c._conv2d_cuda_backward_weight_naive(
            x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def backward_weight_optimized():
        def run():
            orthon_c._conv2d_cuda_backward_weight_optimized(
                x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
            orthon.cuda_synchronize()
        with_env("ORTHON_CUDA_CONV2D_GRAD_WEIGHT", "optimized", run)

    def backward_weight_grouped3x3():
        def run():
            orthon_c._conv2d_cuda_backward_weight_optimized(
                x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
            orthon.cuda_synchronize()
        with_env(
            "ORTHON_CUDA_CONV2D_GRAD_WEIGHT",
            "grouped3x3",
            run,
        )

    def backward_weight_optimized_v2():
        if hasattr(orthon_c, "_conv2d_cuda_backward_weight_optimized_v2"):
            orthon_c._conv2d_cuda_backward_weight_optimized_v2(
                x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
            orthon.cuda_synchronize()
        else:
            with_env(
                "ORTHON_CUDA_CONV2D_GRAD_WEIGHT",
                "optimized_v2",
                backward_weight_optimized,
            )

    def backward_weight_default():
        orthon_c._conv2d_cuda_backward_weight(
            x, w, grad_output, 1, 1, cfg.padding, cfg.padding)
        orthon.cuda_synchronize()

    def backward_bias_naive():
        orthon_c._conv2d_cuda_backward_bias_naive(b, grad_output)
        orthon.cuda_synchronize()

    def backward_bias_optimized():
        orthon_c._conv2d_cuda_backward_bias_optimized(b, grad_output)
        orthon.cuda_synchronize()

    def backward_bias_default():
        orthon_c._conv2d_cuda_backward_bias(b, grad_output)
        orthon.cuda_synchronize()

    backward_input()
    backward_input_naive()
    backward_input_optimized()
    backward_input_optimized_v2()
    backward_weight_naive()
    backward_weight_optimized()
    backward_weight_grouped3x3()
    backward_weight_optimized_v2()
    backward_weight_default()
    backward_bias_naive()
    backward_bias_optimized()
    backward_bias_default()
    grad_input_seconds = measure(backward_input, backward_iters)
    grad_input_naive_seconds = measure(backward_input_naive, backward_iters)
    grad_input_optimized_seconds = measure(backward_input_optimized, backward_iters)
    grad_input_optimized_v2_seconds = measure(backward_input_optimized_v2, backward_iters)
    grad_weight_naive_seconds = measure(backward_weight_naive, backward_iters)
    grad_weight_optimized_seconds = measure(backward_weight_optimized, backward_iters)
    grad_weight_grouped3x3_seconds = measure(backward_weight_grouped3x3, backward_iters)
    grad_weight_optimized_v2_seconds = measure(backward_weight_optimized_v2, backward_iters)
    grad_weight_default_seconds = measure(backward_weight_default, backward_iters)
    grad_bias_naive_seconds = measure(backward_bias_naive, backward_iters)
    grad_bias_optimized_seconds = measure(backward_bias_optimized, backward_iters)
    grad_bias_default_seconds = measure(backward_bias_default, backward_iters)

    def backward_default():
        x.zero_grad()
        w.zero_grad()
        b.zero_grad()
        out = orthon.conv2d(x, w, b, 1, 1, cfg.padding, cfg.padding)
        out.backward()
        orthon.cuda_synchronize()

    def backward_naive_grad_weight():
        with_env("ORTHON_CUDA_CONV2D_GRAD_WEIGHT", "naive", backward_default)

    def backward_naive_grad_input():
        with_env("ORTHON_CUDA_CONV2D_GRAD_INPUT", "naive", backward_default)

    def backward_naive_grad_bias():
        with_env("ORTHON_CUDA_CONV2D_BIAS_GRAD", "naive", backward_default)

    def backward_fused_grad_bias():
        with_env("ORTHON_CUDA_CONV2D_BIAS_GRAD", "fused", backward_default)

    backward_default()
    backward_naive_grad_input()
    backward_naive_grad_weight()
    backward_naive_grad_bias()
    backward_fused_grad_bias()
    backward_default_seconds = measure(backward_default, backward_iters)
    backward_naive_grad_input_seconds = measure(backward_naive_grad_input, backward_iters)
    backward_naive_grad_weight_seconds = measure(backward_naive_grad_weight, backward_iters)
    backward_naive_grad_bias_seconds = measure(backward_naive_grad_bias, backward_iters)
    backward_fused_grad_bias_seconds = measure(backward_fused_grad_bias, backward_iters)
    return (
        naive_forward_seconds,
        optimized_forward_seconds,
        default_forward_seconds,
        grad_input_seconds,
        grad_input_naive_seconds,
        grad_input_optimized_seconds,
        grad_input_optimized_v2_seconds,
        grad_weight_naive_seconds,
        grad_weight_optimized_seconds,
        grad_weight_grouped3x3_seconds,
        grad_weight_optimized_v2_seconds,
        grad_weight_default_seconds,
        grad_bias_naive_seconds,
        grad_bias_optimized_seconds,
        grad_bias_default_seconds,
        backward_naive_grad_input_seconds,
        backward_naive_grad_weight_seconds,
        backward_naive_grad_bias_seconds,
        backward_fused_grad_bias_seconds,
        backward_default_seconds,
    )


def benchmark_torch_cuda(cfg: ConvConfig, forward_iters: int, backward_iters: int):
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    x_np, w_np, b_np = make_arrays(cfg)
    x = torch.tensor(x_np, device="cuda")
    w = torch.tensor(w_np, device="cuda")
    b = torch.tensor(b_np, device="cuda")

    def forward():
        F.conv2d(x, w, b, stride=1, padding=cfg.padding)
        torch.cuda.synchronize()

    forward()
    forward_seconds = measure(forward, forward_iters)

    xb = torch.tensor(x_np, device="cuda", requires_grad=True)
    wb = torch.tensor(w_np, device="cuda", requires_grad=True)
    bb = torch.tensor(b_np, device="cuda", requires_grad=True)

    def backward():
        xb.grad = None
        wb.grad = None
        bb.grad = None
        out = F.conv2d(xb, wb, bb, stride=1, padding=cfg.padding)
        out.sum().backward()
        torch.cuda.synchronize()

    backward()
    backward_seconds = measure(backward, backward_iters)
    return forward_seconds, backward_seconds


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000.0:8.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Orthon Conv2d")
    parser.add_argument("--forward-iters", type=int, default=20)
    parser.add_argument("--backward-iters", type=int, default=10)
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated config names to run; defaults to all configs.",
    )
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Skip CPU and PyTorch CPU timings; useful for large MiniVGG shapes.",
    )
    args = parser.parse_args()
    selected = {name.strip() for name in args.configs.split(",") if name.strip()}

    print("Conv2d benchmark (average per iteration)")
    print("Note: gw default uses grouped3x3 automatically for 3x3 stride=1 padding=1 CUDA workloads.")
    print("Note: gi default uses optimized_v2 automatically for 3x3 stride=1 padding=1 CUDA workloads.")
    print(
        "config        cpu direct   cpu im2col   cuda naive  cuda opt    "
        "cuda default gi default gi naive   gi opt     gi opt_v2  gw naive   gw opt     "
        "gw grouped  gw opt_v2  gw default gb naive   gb opt     gb default bwd naive-gi bwd naive-gw "
        "bwd naive-gb bwd fused   bwd opt    torch cuda  torch cbwd"
    )
    for cfg in CONFIGS:
        if selected and cfg.name not in selected:
            continue
        if args.cuda_only:
            d_direct = d_im2col = d_bwd = None
            t_fwd = t_bwd = None
        else:
            d_direct, d_im2col, d_bwd = benchmark_orthon(
                cfg, args.forward_iters, args.backward_iters
            )
            torch_result = benchmark_torch(cfg, args.forward_iters, args.backward_iters)
            if torch_result is None:
                t_fwd = t_bwd = None
            else:
                t_fwd, t_bwd = torch_result
        d_cuda_result = benchmark_orthon_cuda(cfg, args.forward_iters, args.backward_iters)
        t_cuda_result = benchmark_torch_cuda(cfg, args.forward_iters, args.backward_iters)
        if d_cuda_result is None:
            d_cuda_naive = d_cuda_opt = d_cuda_default = None
            d_grad_input = d_gi_naive = d_gi_opt = d_gi_opt_v2 = None
            d_gw_naive = d_gw_opt = d_gw_grouped = d_gw_opt_v2 = d_gw_default = None
            d_gb_naive = d_gb_opt = d_gb_default = None
            d_cuda_bwd_naive_gi = d_cuda_bwd_naive_gw = None
            d_cuda_bwd_naive_gb = d_cuda_bwd_fused_gb = d_cuda_bwd_opt = None
        else:
            (
                d_cuda_naive,
                d_cuda_opt,
                d_cuda_default,
                d_grad_input,
                d_gi_naive,
                d_gi_opt,
                d_gi_opt_v2,
                d_gw_naive,
                d_gw_opt,
                d_gw_grouped,
                d_gw_opt_v2,
                d_gw_default,
                d_gb_naive,
                d_gb_opt,
                d_gb_default,
                d_cuda_bwd_naive_gi,
                d_cuda_bwd_naive_gw,
                d_cuda_bwd_naive_gb,
                d_cuda_bwd_fused_gb,
                d_cuda_bwd_opt,
            ) = d_cuda_result
        if t_cuda_result is None:
            t_cuda_fwd = t_cuda_bwd = None
        else:
            t_cuda_fwd, t_cuda_bwd = t_cuda_result

        print(
            f"{cfg.name:<12} {fmt(d_direct)}  {fmt(d_im2col)}  {fmt(d_cuda_naive)}  "
            f"{fmt(d_cuda_opt)}  {fmt(d_cuda_default)}  {fmt(d_grad_input)}  "
            f"{fmt(d_gi_naive)}  {fmt(d_gi_opt)}  {fmt(d_gi_opt_v2)}  "
            f"{fmt(d_gw_naive)}  {fmt(d_gw_opt)}  {fmt(d_gw_grouped)}  "
            f"{fmt(d_gw_opt_v2)}  {fmt(d_gw_default)}  "
            f"{fmt(d_gb_naive)}  {fmt(d_gb_opt)}  {fmt(d_gb_default)}  "
            f"{fmt(d_cuda_bwd_naive_gi)}  {fmt(d_cuda_bwd_naive_gw)}  "
            f"{fmt(d_cuda_bwd_naive_gb)}  {fmt(d_cuda_bwd_fused_gb)}  {fmt(d_cuda_bwd_opt)}  "
            f"{fmt(t_cuda_fwd)}  {fmt(t_cuda_bwd)}"
        )


if __name__ == "__main__":
    main()
