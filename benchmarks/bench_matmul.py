#!/usr/bin/env python3
"""
Benchmark: matrix multiply  (naive vs tiled CUDA, vs NumPy and PyTorch)

Run from repo root:
    python benchmarks/bench_matmul.py
"""
import sys
import os
import argparse
import textwrap

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

_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--profile", action="store_true")
_parser.add_argument("--shape", nargs="+", type=int, default=None)
_parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
_parser.add_argument("--device", default="cuda:0")
_parser.add_argument("--clean-dispatch", action="store_true",
                     help="Only time normal dispatch rows; skip forced cuBLAS and event-only helpers.")
_parser.add_argument("--profile-cublas-overhead", action="store_true",
                     help="Print internal cuBLAS matmul overhead timings below each table.")
_args, _ = _parser.parse_known_args()

SIZES = [_shape_arg[0]] if _shape_arg and len(_shape_arg) == 1 else [256, 512, 1024]
BATCHED_SHAPE = _shape_arg if _shape_arg and len(_shape_arg) == 4 else [32, 64, 64, 64]
RUN_2D = not (_shape_arg and len(_shape_arg) == 4)
_reps = REPS_PROFILE if _is_profile else REPS
DTYPE = _args.dtype
CUDA_DEVICE = _args.device
CLEAN_DISPATCH = _args.clean_dispatch
PROFILE_CUBLAS_OVERHEAD = _args.profile_cublas_overhead

HEADERS = ["shape", "backend", "device", "kernel",
           f"avg ms (w={WARMUP} r={_reps})", "speedup vs Orthon CPU"]


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _custom_label(dtype):
    return "custom-tiled-bfloat16" if dtype == "bfloat16" else "custom-tiled-float16"


def _dispatch_label(dtype, path):
    return f"dispatch-{dtype} ({path})"


def _forced_cublas_label(dtype):
    return "forced-cublas-bfloat16" if dtype == "bfloat16" else "forced-cublas-float16"


def _event_only_label(dtype):
    return "cublas-event-only-bfloat16" if dtype == "bfloat16" else "cublas-event-only-float16"


def _print_guard_reports(reports):
    if not reports:
        return
    print("\nGuard report:")
    for title, detail in reports:
        print(f"  {title}:")
        predicates = [p.strip() for p in detail.split(";") if p.strip()]
        if predicates:
            for predicate in predicates:
                for line in textwrap.wrap(predicate, width=100):
                    print(f"    {line}")
        else:
            for line in textwrap.wrap(detail, width=100):
                print(f"    {line}")


def bench_matmul():
    # The _matmul_* diagnostics are None in CPU-only builds
    matmul_naive = getattr(ortho, '_matmul_naive', None)
    matmul_tiled = getattr(ortho, '_matmul_tiled', None)
    matmul_custom = getattr(ortho, '_matmul_tiled_custom', None)
    matmul_cublas = getattr(ortho, '_matmul_tiled_cublas', None)
    matmul_path = getattr(ortho, '_matmul_dispatch_path', None)
    guard_report = getattr(ortho, '_matmul_cublas_guard_report', None)
    cublas_elapsed_ms = getattr(ortho, '_matmul_cublas_elapsed_ms', None)
    cublas_profile = getattr(ortho, '_matmul_cublas_profile_report', None)

    for sz in SIZES:
        M = K = N = sz
        rows = []
        guard_reports = []
        base = None

        if DTYPE == "float32":
            a_cpu = ortho.ones([M, K])
            b_cpu = ortho.ones([K, N])
            ms = bench_fn(lambda: ortho.matmul(a_cpu, b_cpu), reps=_reps)
            base = ms
            rows.append([f"{M}x{N}", "Orthon", "cpu", "—",
                         fmt_ms(ms), "1.00x"])
        elif _has_numpy:
            # Same construction rule as the batched case below. NumPy has a
            # native float16, so from_numpy INFERS that dtype and the explicit
            # override must not be passed; it has no bfloat16, which is the one
            # dtype the override exists for (it takes a float32 array and
            # converts on the host).
            np_dtype = np.float16 if DTYPE == "float16" else np.float32
            a_np = np.ones((M, K), dtype=np_dtype)
            b_np = np.ones((K, N), dtype=np_dtype)
            if DTYPE == "bfloat16":
                a_cpu = ortho.from_numpy(a_np, dtype=DTYPE)
                b_cpu = ortho.from_numpy(b_np, dtype=DTYPE)
            else:
                a_cpu = ortho.from_numpy(a_np)
                b_cpu = ortho.from_numpy(b_np)
        else:
            a_cpu = b_cpu = None

        if _has_cuda:
            try:
                if a_cpu is None:
                    a_gpu = ortho.ones([M, K], dtype=DTYPE).to(CUDA_DEVICE)
                    b_gpu = ortho.ones([K, N], dtype=DTYPE).to(CUDA_DEVICE)
                else:
                    a_gpu = ortho.to(a_cpu, CUDA_DEVICE)
                    b_gpu = ortho.to(b_cpu, CUDA_DEVICE)
            except RuntimeError as exc:
                rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                             f"unavailable: {exc}", "—", "—"])
                a_gpu = b_gpu = None

            if a_gpu is not None and DTYPE == "float32" and matmul_naive is not None:
                ms = bench_fn(lambda: matmul_naive(a_gpu, b_gpu),
                              reps=_reps, pre=ortho_sync, post=ortho_sync)
                rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE, "naive",
                             fmt_ms(ms), fmt_speedup(base, ms)])

            if a_gpu is not None and DTYPE in ("float16", "bfloat16") and matmul_custom is not None:
                ms_custom = bench_fn(lambda: matmul_custom(a_gpu, b_gpu),
                                     reps=_reps, pre=ortho_sync, post=ortho_sync)
                base = ms_custom if base is None else base
                rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                             _custom_label(DTYPE), fmt_ms(ms_custom),
                             "1.00x" if base == ms_custom else fmt_speedup(base, ms_custom)])
            else:
                ms_custom = None

            if a_gpu is not None and matmul_tiled is not None:
                path = "unknown"
                if matmul_path is not None:
                    try:
                        path = matmul_path(a_gpu, b_gpu)
                    except RuntimeError:
                        pass
                dispatch_label = _dispatch_label(DTYPE, path)
                ms = bench_fn(lambda: matmul_tiled(a_gpu, b_gpu),
                              reps=_reps, pre=ortho_sync, post=ortho_sync)
                rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                             dispatch_label,
                             fmt_ms(ms), fmt_speedup(base, ms)])
            elif a_gpu is not None:
                ms = bench_fn(lambda: ortho.matmul(a_gpu, b_gpu),
                              reps=_reps, pre=ortho_sync, post=ortho_sync)
                rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE, "dispatch",
                             fmt_ms(ms), fmt_speedup(base, ms)])

            if (not CLEAN_DISPATCH and a_gpu is not None and
                    DTYPE in ("float16", "bfloat16") and matmul_cublas is not None):
                try:
                    ms = bench_fn(lambda: matmul_cublas(a_gpu, b_gpu),
                                  reps=_reps, pre=ortho_sync, post=ortho_sync)
                    rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                                 _forced_cublas_label(DTYPE), fmt_ms(ms),
                                 fmt_speedup(base, ms)])
                except RuntimeError as exc:
                    detail = str(exc)
                    if guard_report is not None:
                        try:
                            detail = guard_report(a_gpu, b_gpu)
                        except RuntimeError:
                            pass
                    guard_reports.append((f"{M}x{K} @ {K}x{N}", detail))
                    rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                                 "forced-cublas-unavailable", "—", "—"])

            if not CLEAN_DISPATCH and a_gpu is not None and cublas_elapsed_ms is not None:
                try:
                    ms = cublas_elapsed_ms(a_gpu, b_gpu, _reps)
                    rows.append([f"{M}x{N}", "Orthon", CUDA_DEVICE,
                                 _event_only_label(DTYPE), fmt_ms(ms),
                                 fmt_speedup(base, ms)])
                except RuntimeError:
                    pass

            if (PROFILE_CUBLAS_OVERHEAD and a_gpu is not None and
                    cublas_profile is not None):
                try:
                    guard_reports.append((
                        f"profile {M}x{K} @ {K}x{N}",
                        cublas_profile(a_gpu, b_gpu, _reps),
                    ))
                except RuntimeError as exc:
                    guard_reports.append((f"profile {M}x{K} @ {K}x{N}", str(exc)))

        if _has_numpy and DTYPE == "float32":
            a_np = np.ones((M, K), dtype=np.float32)
            b_np = np.ones((K, N), dtype=np.float32)
            ms = bench_fn(lambda: np.matmul(a_np, b_np), reps=_reps)
            rows.append([f"{M}x{N}", "NumPy", "cpu", "BLAS",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch:
            a_pt = torch.ones(M, K, dtype=_torch_dtype(DTYPE))
            b_pt = torch.ones(K, N, dtype=_torch_dtype(DTYPE))
            ms = bench_fn(lambda: torch.matmul(a_pt, b_pt), reps=_reps)
            rows.append([f"{M}x{N}", "PyTorch", "cpu", "BLAS",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        if _has_torch_cuda:
            a_ptc = torch.ones(M, K, dtype=_torch_dtype(DTYPE), device=CUDA_DEVICE)
            b_ptc = torch.ones(K, N, dtype=_torch_dtype(DTYPE), device=CUDA_DEVICE)
            ms = bench_fn(lambda: torch.matmul(a_ptc, b_ptc),
                          reps=_reps, pre=torch_sync, post=torch_sync)
            rows.append([f"{M}x{N}", "PyTorch", CUDA_DEVICE, f"cuBLAS-{DTYPE}",
                         fmt_ms(ms), fmt_speedup(base, ms)])

        print_table(f"matmul {DTYPE}  |  {M}x{K} @ {K}x{N}", HEADERS, rows)
        _print_guard_reports(guard_reports)


def bench_batched_low_precision_matmul():
    if not (_has_cuda and _has_numpy):
        return
    if DTYPE not in ("float16", "bfloat16"):
        return
    custom = getattr(ortho, "_matmul_batched_tiled_custom", None)
    cublas = getattr(ortho, "_matmul_batched_tiled_cublas", None)
    matmul_path = getattr(ortho, "_matmul_dispatch_path", None)
    guard_report = getattr(ortho, "_matmul_cublas_guard_report", None)
    cublas_elapsed_ms = getattr(ortho, "_matmul_cublas_elapsed_ms", None)
    cublas_profile = getattr(ortho, "_matmul_cublas_profile_report", None)
    B, M, K, N = BATCHED_SHAPE
    rows = []
    guard_reports = []

    np_dtype = np.float16 if DTYPE == "float16" else np.float32
    a_np = np.ones((B, M, K), dtype=np_dtype)
    b_np = np.ones((B, K, N), dtype=np_dtype)
    if DTYPE == "bfloat16":
        a_cpu = ortho.from_numpy(a_np, dtype=DTYPE)
        b_cpu = ortho.from_numpy(b_np, dtype=DTYPE)
    else:
        a_cpu = ortho.from_numpy(a_np)
        b_cpu = ortho.from_numpy(b_np)
    try:
        a_gpu = a_cpu.to(CUDA_DEVICE)
        b_gpu = b_cpu.to(CUDA_DEVICE)
    except RuntimeError as exc:
        rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                     f"unavailable: {exc}", "—", "—"])
        print_table(
            f"batched {DTYPE} matmul  |  {B}x{M}x{K} @ {B}x{K}x{N}",
            HEADERS,
            rows,
        )
        return

    if custom is not None:
        ms_custom = bench_fn(lambda: custom(a_gpu, b_gpu),
                             reps=_reps, pre=ortho_sync, post=ortho_sync)
        rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                     _custom_label(DTYPE), fmt_ms(ms_custom), "1.00x"])
    else:
        ms_custom = None

    path = "unknown"
    if matmul_path is not None:
        try:
            path = matmul_path(a_gpu, b_gpu)
        except RuntimeError:
            pass
    dispatch_label = _dispatch_label(DTYPE, path)
    ms_dispatch = bench_fn(lambda: ortho.matmul(a_gpu, b_gpu),
                           reps=_reps, pre=ortho_sync, post=ortho_sync)
    rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                 dispatch_label, fmt_ms(ms_dispatch),
                 fmt_speedup(ms_custom, ms_dispatch)])

    if not CLEAN_DISPATCH and cublas is not None:
        try:
            ms = bench_fn(lambda: cublas(a_gpu, b_gpu),
                          reps=_reps, pre=ortho_sync, post=ortho_sync)
            rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                         _forced_cublas_label(DTYPE), fmt_ms(ms),
                         fmt_speedup(ms_custom, ms)])
        except RuntimeError as exc:
            detail = str(exc)
            if guard_report is not None:
                try:
                    detail = guard_report(a_gpu, b_gpu)
                except RuntimeError:
                    pass
            guard_reports.append((f"{B}x{M}x{K} @ {B}x{K}x{N}", detail))
            rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                         "forced-cublas-unavailable", "—", "—"])

    if not CLEAN_DISPATCH and cublas_elapsed_ms is not None:
        try:
            ms = cublas_elapsed_ms(a_gpu, b_gpu, _reps)
            rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "Orthon", CUDA_DEVICE,
                         _event_only_label(DTYPE), fmt_ms(ms),
                         fmt_speedup(ms_custom, ms)])
        except RuntimeError:
            pass

    if PROFILE_CUBLAS_OVERHEAD and cublas_profile is not None:
        try:
            guard_reports.append((
                f"profile {B}x{M}x{K} @ {B}x{K}x{N}",
                cublas_profile(a_gpu, b_gpu, _reps),
            ))
        except RuntimeError as exc:
            guard_reports.append((f"profile {B}x{M}x{K} @ {B}x{K}x{N}", str(exc)))

    if _has_torch_cuda:
        a_pt = torch.ones((B, M, K), dtype=_torch_dtype(DTYPE), device=CUDA_DEVICE)
        b_pt = torch.ones((B, K, N), dtype=_torch_dtype(DTYPE), device=CUDA_DEVICE)
        ms_torch = bench_fn(lambda: torch.matmul(a_pt, b_pt),
                            reps=_reps, pre=torch_sync, post=torch_sync)
        rows.append([f"{B}x{M}x{K} @ {B}x{K}x{N}", "PyTorch", CUDA_DEVICE,
                     f"cuBLAS-{DTYPE}", fmt_ms(ms_torch),
                     fmt_speedup(ms_custom, ms_torch)])

    print_table(
        f"batched {DTYPE} matmul  |  {B}x{M}x{K} @ {B}x{K}x{N}",
        HEADERS,
        rows,
    )
    _print_guard_reports(guard_reports)


if __name__ == "__main__":
    print("Matmul Benchmarks")
    print_env()
    if RUN_2D:
        bench_matmul()
    bench_batched_low_precision_matmul()
