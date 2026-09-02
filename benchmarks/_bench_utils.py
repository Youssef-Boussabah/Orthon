"""Shared timing, formatting, and detection utilities for all benchmarks."""
import argparse
import time

# Orthon import.
try:
    import orthon as ortho
    _has_ortho   = True
    _has_cuda  = ortho.cuda_enabled()
except ImportError:
    ortho        = None
    _has_ortho   = False
    _has_cuda  = False

# NumPy import.
try:
    import numpy as np
    _has_numpy = True
except ImportError:
    np         = None
    _has_numpy = False

# PyTorch import.
try:
    import torch
    _has_torch      = True
    _has_torch_cuda = torch.cuda.is_available()
except ImportError:
    torch           = None
    _has_torch      = False
    _has_torch_cuda = False


# CUDA synchronization helpers.

def ortho_sync():
    """Block until all queued Orthon CUDA work completes."""
    if _has_cuda:
        ortho.cuda_synchronize()


def torch_sync():
    """Block until all queued PyTorch CUDA work completes."""
    if _has_torch_cuda:
        torch.cuda.synchronize()


# Timing helpers.

WARMUP        = 5
REPS          = 20
REPS_PROFILE  = 100   # higher rep count used in --profile mode


def parse_profile_args():
    """Parse --profile / --shape from sys.argv.

    Returns (is_profile: bool, shape: list[int] | None).
    The caller interprets 'shape' according to its own dimensionality:
      - 1 int  → flat size (elementwise / reductions)
      - 1 int  → square side (matmul)
      - 2 ints → (B, C) (softmax / cross_entropy)
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--shape', nargs='+', type=int, default=None)
    args, _ = p.parse_known_args()
    return args.profile, args.shape


def bench_fn(fn, warmup=WARMUP, reps=REPS, pre=None, post=None):
    """Run fn with warmup iterations then time reps iterations.

    pre  -- called before starting each timed iteration (e.g. cuda_sync to
            drain queued work before the clock starts)
    post -- called after fn() inside the timed window (e.g. cuda_sync to wait
            for async GPU kernels before stopping the clock)

    Returns average latency in milliseconds.
    """
    for _ in range(warmup):
        fn()
    if post:
        post()
    total = 0.0
    for _ in range(reps):
        if pre:
            pre()
        t0 = time.perf_counter()
        fn()
        if post:
            post()
        t1 = time.perf_counter()
        total += (t1 - t0) * 1000.0
    return total / reps


# Output formatting helpers.

def fmt_ms(ms):
    return f"{ms:.3f}" if ms is not None else "—"


def fmt_speedup(base_ms, this_ms):
    if base_ms is None or this_ms is None or this_ms <= 0.0:
        return "—"
    return f"{base_ms / this_ms:.2f}x"


def print_table(title, headers, rows):
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")
    if not rows:
        print("  (no data)")
        return
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(fmt.format(*[str(h) for h in headers]))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


def print_env():
    print(f"  Orthon          : {'yes' if _has_ortho else 'MISSING (pip install -e .)'}")
    print(f"  Orthon CUDA     : {'yes' if _has_cuda else 'no'}")
    print(f"  NumPy           : {'yes' if _has_numpy else 'no (skipped)'}")
    print(f"  PyTorch         : {'yes' if _has_torch else 'no (skipped)'}",
          end="")
    if _has_torch:
        print(f"  (CUDA: {'yes' if _has_torch_cuda else 'no'})")
    else:
        print()
    print(f"  warmup={WARMUP}  reps={REPS}")
