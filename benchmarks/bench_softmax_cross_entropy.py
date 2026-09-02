#!/usr/bin/env python3
"""
Benchmark: softmax and cross_entropy

Run from repo root:
    python benchmarks/bench_softmax_cross_entropy.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _bench_utils import (
    ortho, torch,
    _has_ortho, _has_cuda, _has_torch, _has_torch_cuda,
    ortho_sync, torch_sync,
    bench_fn, print_table, fmt_ms, fmt_speedup, print_env,
    WARMUP, REPS, REPS_PROFILE, parse_profile_args,
)

if not _has_ortho:
    print("orthon not found.  Run: pip install -e .")
    sys.exit(1)

_is_profile, _shape_arg = parse_profile_args()
# Accept --shape B C (two ints) for 2-D shapes
SHAPES = [tuple(_shape_arg)] if (_shape_arg and len(_shape_arg) == 2) \
         else [(512, 1024), (1024, 2048), (2048, 4096)]
_reps = REPS_PROFILE if _is_profile else REPS

HEADERS = ["shape", "backend", "device",
           f"avg ms (w={WARMUP} r={_reps})", "speedup vs Orthon CPU"]


def bench_softmax():
    for (B, C) in SHAPES:
        rows = []
        base = None

        x_cpu = ortho.ones([B, C])
        ms = bench_fn(lambda: ortho.softmax(x_cpu), reps=_reps)
        base = ms
        rows.append([f"[{B},{C}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            x_gpu = ortho.to(x_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.softmax(x_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{B},{C}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            import torch.nn.functional as F
            x_pt = torch.ones(B, C)
            ms = bench_fn(lambda: F.softmax(x_pt, dim=1), reps=_reps)
            rows.append([f"[{B},{C}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            import torch.nn.functional as F
            x_ptc = torch.ones(B, C, device="cuda")
            ms = bench_fn(lambda: F.softmax(x_ptc, dim=1),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{B},{C}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"softmax  |  shape [{B},{C}]", HEADERS, rows)


def bench_cross_entropy():
    for (B, C) in SHAPES:
        rows = []
        base = None
        # Fixed targets: class i % C for each row i
        targets = [i % C for i in range(B)]

        x_cpu = ortho.ones([B, C])
        ms = bench_fn(lambda: ortho.cross_entropy(x_cpu, targets), reps=_reps)
        base = ms
        rows.append([f"[{B},{C}]", "Orthon", "cpu", fmt_ms(ms), "1.00x"])

        if _has_cuda:
            x_gpu = ortho.to(x_cpu, "cuda:0")
            ms = bench_fn(lambda: ortho.cross_entropy(x_gpu, targets),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"[{B},{C}]", "Orthon", "cuda:0",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            import torch.nn.functional as F
            x_pt   = torch.ones(B, C)
            t_pt   = torch.tensor(targets, dtype=torch.long)
            ms = bench_fn(lambda: F.cross_entropy(x_pt, t_pt), reps=_reps)
            rows.append([f"[{B},{C}]", "PyTorch", "cpu",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            import torch.nn.functional as F
            x_ptc  = torch.ones(B, C, device="cuda")
            t_ptc  = torch.tensor(targets, dtype=torch.long, device="cuda")
            ms = bench_fn(lambda: F.cross_entropy(x_ptc, t_ptc),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"[{B},{C}]", "PyTorch", "cuda",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"cross_entropy  |  shape [{B},{C}]", HEADERS, rows)


if __name__ == "__main__":
    print("Softmax & Cross-Entropy Benchmarks")
    print_env()
    bench_softmax()
    bench_cross_entropy()
