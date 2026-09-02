#!/usr/bin/env python3
"""
Benchmark: reduction ops  (sum, mean)

Run from repo root:
    python benchmarks/bench_reductions.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _bench_utils import (
    ortho, np, torch,
    _has_ortho, _has_cuda, _has_numpy, _has_torch, _has_torch_cuda,
    ortho_sync, torch_sync,
    bench_fn, print_table, fmt_ms, fmt_speedup, print_env,
    WARMUP, REPS, REPS_PROFILE, parse_profile_args,
)

if not _has_ortho:
    print("orthon not found.  Run: pip install -e .")
    sys.exit(1)

_is_profile, _shape_arg = parse_profile_args()
SIZES = [_shape_arg[0]] if _shape_arg else [1_000_000, 4_000_000, 16_000_000]
_reps = REPS_PROFILE if _is_profile else REPS

HEADERS = ["shape", "backend", "device",
           f"avg ms (w={WARMUP} r={_reps})", "speedup vs Orthon CPU"]


def bench_sum():
    for n in SIZES:
        shape = [n]
        rows  = []
        base  = None

        x_cpu = ortho.ones(shape)
        ms = bench_fn(lambda: ortho.sum(x_cpu), reps=_reps)
        base = ms
        rows.append([f"[{n}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            x_gpu = ortho.to(x_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.sum(x_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{n}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_numpy:
            x_np = np.ones(n, dtype=np.float32)
            ms = bench_fn(lambda: np.sum(x_np), reps=_reps)
            rows.append([f"[{n}]", "NumPy", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            x_pt = torch.ones(n)
            ms = bench_fn(lambda: x_pt.sum(), reps=_reps)
            rows.append([f"[{n}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            x_ptc = torch.ones(n, device="cuda")
            ms = bench_fn(lambda: x_ptc.sum(),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{n}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"sum  |  shape [{n}]", HEADERS, rows)


def bench_mean():
    for n in SIZES:
        shape = [n]
        rows  = []
        base  = None

        x_cpu = ortho.ones(shape)
        ms = bench_fn(lambda: ortho.mean(x_cpu), reps=_reps)
        base = ms
        rows.append([f"[{n}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            x_gpu = ortho.to(x_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.mean(x_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{n}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_numpy:
            x_np = np.ones(n, dtype=np.float32)
            ms = bench_fn(lambda: np.mean(x_np), reps=_reps)
            rows.append([f"[{n}]", "NumPy", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            x_pt = torch.ones(n)
            ms = bench_fn(lambda: x_pt.mean(), reps=_reps)
            rows.append([f"[{n}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            x_ptc = torch.ones(n, device="cuda")
            ms = bench_fn(lambda: x_ptc.mean(),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{n}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"mean  |  shape [{n}]", HEADERS, rows)


if __name__ == "__main__":
    print("Reduction Benchmarks")
    print_env()
    bench_sum()
    bench_mean()
