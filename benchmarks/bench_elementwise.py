#!/usr/bin/env python3
"""
Benchmark: elementwise ops  (add, relu)

Run from repo root:
    python benchmarks/bench_elementwise.py
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


def bench_add():
    for n in SIZES:
        shape = [n]
        rows  = []
        base  = None

        a_cpu = ortho.ones(shape)
        b_cpu = ortho.ones(shape)
        ms = bench_fn(lambda: ortho.add(a_cpu, b_cpu), reps=_reps)
        base = ms
        rows.append([f"[{n}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            a_gpu = ortho.to(a_cpu, "cuda:0")
            b_gpu = ortho.to(b_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.add(a_gpu, b_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{n}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_numpy:
            a_np = np.ones(n, dtype=np.float32)
            b_np = np.ones(n, dtype=np.float32)
            ms = bench_fn(lambda: np.add(a_np, b_np), reps=_reps)
            rows.append([f"[{n}]", "NumPy", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            a_pt = torch.ones(n)
            b_pt = torch.ones(n)
            ms = bench_fn(lambda: torch.add(a_pt, b_pt), reps=_reps)
            rows.append([f"[{n}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            a_ptc = torch.ones(n, device="cuda")
            b_ptc = torch.ones(n, device="cuda")
            ms = bench_fn(lambda: torch.add(a_ptc, b_ptc),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{n}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"add  |  shape [{n}]", HEADERS, rows)


def bench_relu():
    for n in SIZES:
        shape = [n]
        rows  = []
        base  = None

        x_cpu = ortho.ones(shape)
        ms = bench_fn(lambda: ortho.relu(x_cpu), reps=_reps)
        base = ms
        rows.append([f"[{n}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            x_gpu = ortho.to(x_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.relu(x_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{n}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_numpy:
            x_np = np.ones(n, dtype=np.float32)
            ms = bench_fn(lambda: np.maximum(x_np, 0), reps=_reps)
            rows.append([f"[{n}]", "NumPy", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            x_pt = torch.ones(n)
            ms = bench_fn(lambda: torch.relu(x_pt), reps=_reps)
            rows.append([f"[{n}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            x_ptc = torch.ones(n, device="cuda")
            ms = bench_fn(lambda: torch.relu(x_ptc),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{n}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"relu  |  shape [{n}]", HEADERS, rows)


if __name__ == "__main__":
    print("Elementwise Benchmarks")
    print_env()
    bench_add()
    bench_relu()
