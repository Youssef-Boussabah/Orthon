#!/usr/bin/env python3
"""Orthon vs. PyTorch: the canonical head-to-head comparison harness.

Every number this script prints is measured in the process that prints it.
Nothing is copied forward from an earlier run or an earlier table.

The contract, stated once and applied to every row:

* Inputs are allocated on the GPU **before** the clock starts. Import, CUDA
  context creation, host-to-device transfer, input generation and setup are
  all outside the timed region.
* Both frameworks are warmed up, then each sample times a batch of `inner`
  repetitions of the same call, with an explicit framework synchronization
  immediately before the clock starts and immediately before it stops. The
  reported latency is the batch wall time divided by `inner`, so sub-millisecond
  kernels are not measured through Python timer noise.
* Samples alternate order between rounds (Orthon first on even rounds, PyTorch
  first on odd rounds) so neither framework systematically gets the warmer or
  colder GPU.
* Every case proves numerical equivalence before it is timed. A row whose
  numerical check fails is reported as INVALID and carries no performance
  claim.
* The PyTorch baseline is **eager**. `--torch-compile` adds a separately
  labelled compiled comparison; the two are never mixed in one row.

Usage
-----

    python benchmarks/compare_pytorch.py                       # full matrix
    python benchmarks/compare_pytorch.py --list                # what exists
    python benchmarks/compare_pytorch.py --benchmark cross_entropy
    python benchmarks/compare_pytorch.py --benchmark bf16_matmul --shape 1024
    python benchmarks/compare_pytorch.py --highlight cross_entropy --shape 2048 4096
    python benchmarks/compare_pytorch.py --check-only          # equivalence only
    python benchmarks/compare_pytorch.py --markdown out.md --json out.json
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# --------------------------------------------------------------------------
# Imports. Both frameworks are mandatory here: this script exists only to
# compare them, so a missing one is an error, not a skipped row.
# --------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("compare_pytorch.py needs NumPy: pip install numpy")

try:
    import orthon as ortho
except ImportError:
    sys.exit("compare_pytorch.py needs Orthon: pip install -e .")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit('compare_pytorch.py needs PyTorch: pip install "torch>=2.0"')


ORTHON_DEVICE = "cuda:0"
TORCH_DEVICE = "cuda:0"

# A meaningful Orthon win needs at least this much margin. Anything inside the
# band is reported as "similar", never as a victory.
WIN_THRESHOLD = 0.10

# Relative IQR above this marks a measurement noisy; it does not by itself
# invalidate the ordering (the speedup band does that).
NOISE_LIMIT = 0.15


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def _nvidia_smi(query: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except Exception:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _orthon_version() -> str:
    try:
        from importlib.metadata import version
        return version("orthon")
    except Exception:
        return "unknown"


def _orthon_sm_targets() -> str:
    """Which SM architectures the installed Orthon extension actually carries.

    Reported because a PTX-JIT fallback and a native cubin are not the same
    measurement. Best-effort: cuobjdump is not required to run the benchmark.
    """
    try:
        import orthon._C as _c
        path = _c.__file__
    except Exception:
        return "unavailable"
    try:
        out = subprocess.run(["cuobjdump", "--list-elf", path],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return "unavailable (cuobjdump not found)"
    if out.returncode != 0:
        return "unavailable"
    seen = []
    for line in out.stdout.splitlines():
        for token in line.replace(".", " ").split():
            if token.startswith("sm_") and token not in seen:
                seen.append(token)
    return ", ".join(seen) if seen else "none (PTX only)"


def collect_environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": f"sm_{props.major}{props.minor}",
        "gpu_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        "gpu_l2_cache_mib": round(getattr(props, "L2_cache_size", 0) / (1024 ** 2), 1),
        "gpu_sm_count": props.multi_processor_count,
        "driver": _nvidia_smi("driver_version") or "unknown",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "orthon_version": _orthon_version(),
        "orthon_cuda_enabled": bool(ortho.cuda_enabled()),
        "orthon_extension_sm": _orthon_sm_targets(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
        "torch_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def print_environment(env: dict) -> None:
    print("=" * 86)
    print("  Environment")
    print("=" * 86)
    order = [
        ("GPU", "gpu"),
        ("Compute capability", "gpu_capability"),
        ("GPU memory (GiB)", "gpu_memory_gb"),
        ("GPU L2 cache (MiB)", "gpu_l2_cache_mib"),
        ("SM count", "gpu_sm_count"),
        ("NVIDIA driver", "driver"),
        ("Platform", "platform"),
        ("Python", "python"),
        ("NumPy", "numpy"),
        ("PyTorch", "torch_version"),
        ("PyTorch CUDA", "torch_cuda_version"),
        ("cuDNN", "torch_cudnn_version"),
        ("Orthon", "orthon_version"),
        ("Orthon CUDA build", "orthon_cuda_enabled"),
        ("Orthon extension SM", "orthon_extension_sm"),
        ("torch matmul TF32", "torch_matmul_allow_tf32"),
        ("torch cuDNN TF32", "torch_cudnn_allow_tf32"),
        ("torch fp32 matmul precision", "torch_float32_matmul_precision"),
    ]
    for label, key in order:
        value = env.get(key)
        if value is None:
            continue
        print(f"  {label:<30} {value}")
    print()


# --------------------------------------------------------------------------
# Tensor conversion helpers
# --------------------------------------------------------------------------

_NUMPY_OF = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
    "bool": np.bool_,
}


def ortho_to_numpy(tensor) -> np.ndarray:
    """Read an Orthon tensor back as float64-comparable NumPy data.

    Uses the raw-byte accessor rather than tolist() so a 2048x4096 check costs
    a memcpy, not four million Python objects. float16/bfloat16 are widened to
    float32 exactly (bfloat16 is the top 16 bits of a float32).
    """
    cpu = ortho.to(tensor, "cpu")
    if not cpu.is_contiguous:
        cpu = ortho.contiguous(cpu)
    name = str(cpu.dtype)
    raw = ortho._tensor_to_bytes(cpu)
    if name == "bfloat16":
        widened = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        flat = widened.view(np.float32)
    elif name == "float16":
        flat = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        flat = np.frombuffer(raw, dtype=_NUMPY_OF[name])
    return np.array(flat, copy=True).reshape(tuple(cpu.shape))


def torch_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().to("cpu", torch.float32).numpy()


def compare(reference: np.ndarray, candidate: np.ndarray, tol: float) -> dict:
    """Max absolute difference, scaled by the reference's own magnitude."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    got = np.asarray(candidate, dtype=np.float64).ravel()
    if ref.shape != got.shape:
        return {"ok": False, "max_abs": float("inf"), "rel": float("inf"),
                "tol": tol, "note": f"shape mismatch {ref.shape} vs {got.shape}"}
    diff = np.abs(ref - got)
    max_abs = float(diff.max()) if diff.size else 0.0
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    rel = max_abs / max(scale, 1e-12)
    return {"ok": bool(rel <= tol), "max_abs": max_abs, "rel": rel,
            "tol": tol, "note": ""}


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def ortho_sync() -> None:
    ortho.cuda_synchronize()


def torch_sync() -> None:
    torch.cuda.synchronize()


@dataclass
class TimingConfig:
    warmup: int = 25
    samples: int = 11
    target_ms: float = 40.0
    max_sample_ms: float = 500.0
    min_inner: int = 8
    max_inner: int = 50000
    fixed_inner: Optional[int] = None


def _batch_ms(fn: Callable, inner: int, sync: Callable) -> float:
    sync()
    start = time.perf_counter()
    for _ in range(inner):
        fn()
    sync()
    return (time.perf_counter() - start) * 1000.0


def _estimate_ms(fn: Callable, sync: Callable, trials: int = 5) -> float:
    fn()
    sync()
    best = None
    for _ in range(trials):
        elapsed = _batch_ms(fn, 1, sync)
        best = elapsed if best is None else min(best, elapsed)
    return max(best, 1e-4)


def _choose_inner(est_a: float, est_b: float, cfg: TimingConfig) -> int:
    if cfg.fixed_inner:
        return cfg.fixed_inner
    fast, slow = min(est_a, est_b), max(est_a, est_b)
    wanted = math.ceil(cfg.target_ms / fast)
    slow_cap = max(1, int(cfg.max_sample_ms // slow))
    return max(cfg.min_inner, min(wanted, slow_cap, cfg.max_inner))


def _quartiles(values: Sequence[float]):
    ordered = sorted(values)
    n = len(ordered)

    def pick(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        low, high = math.floor(pos), math.ceil(pos)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    return pick(0.25), pick(0.5), pick(0.75)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class Case:
    benchmark: str
    shape: tuple
    label: str
    dtype: str
    mode: str                      # "forward" or "forward+backward"
    note: str
    orthon_fn: Callable
    torch_fn: Callable
    check: Callable[[], dict]
    # Built only when --torch-compile is passed: constructing a compiled callable
    # is itself expensive, and on platforms with no Triton wheel it fails.
    torch_compile_factory: Optional[Callable[[], Callable]] = None


@dataclass
class Result:
    case: Case
    inner: int
    orthon_samples: list
    torch_samples: list
    check: dict
    compiled_samples: Optional[list] = None
    compile_error: Optional[str] = None

    # Derived, filled in by finalize().
    orthon_median: float = 0.0
    orthon_iqr: float = 0.0
    torch_median: float = 0.0
    torch_iqr: float = 0.0
    compiled_median: Optional[float] = None
    speedup: float = 0.0
    band: tuple = (0.0, 0.0)
    verdict: str = ""
    stability: str = ""

    def finalize(self) -> "Result":
        o25, o50, o75 = _quartiles(self.orthon_samples)
        t25, t50, t75 = _quartiles(self.torch_samples)
        self.orthon_median, self.orthon_iqr = o50, o75 - o25
        self.torch_median, self.torch_iqr = t50, t75 - t25
        self.speedup = t50 / o50 if o50 > 0 else float("inf")
        self.band = (t25 / o75 if o75 > 0 else 0.0,
                     t75 / o25 if o25 > 0 else float("inf"))
        if self.compiled_samples:
            self.compiled_median = _quartiles(self.compiled_samples)[1]

        noisy = ((self.orthon_iqr / self.orthon_median) > NOISE_LIMIT or
                 (self.torch_iqr / self.torch_median) > NOISE_LIMIT)
        self.stability = "noisy" if noisy else "stable"

        if not self.check.get("ok", False):
            self.verdict = "invalid"
        elif self.band[0] <= 1.0 <= self.band[1]:
            self.verdict = "unreliable"
        elif self.speedup >= 1.0 + WIN_THRESHOLD:
            self.verdict = "orthon"
        elif self.speedup <= 1.0 / (1.0 + WIN_THRESHOLD):
            self.verdict = "pytorch"
        else:
            self.verdict = "tie"
        return self


VERDICT_TEXT = {
    "orthon": "Orthon faster",
    "pytorch": "PyTorch faster",
    "tie": "similar",
    "unreliable": "ordering unreliable",
    "invalid": "NUMERICAL CHECK FAILED",
}


def _seeded(*key) -> np.random.Generator:
    """One deterministic generator per case, derived from its identity."""
    seed_text = "|".join(str(k) for k in key).encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], "big")
    return np.random.default_rng(digest)


def _ortho_cuda(array: np.ndarray, dtype: Optional[str] = None):
    if dtype is None:
        return ortho.to(ortho.from_numpy(np.ascontiguousarray(array)), ORTHON_DEVICE)
    return ortho.to(ortho.from_numpy(np.ascontiguousarray(array), dtype=dtype),
                    ORTHON_DEVICE)


def _matmul_path(a, b) -> str:
    reporter = getattr(ortho, "_matmul_dispatch_path", None)
    if reporter is None:
        return "unknown"
    try:
        return reporter(a, b)
    except Exception:
        return "unknown"


# ---- bfloat16 matmul ------------------------------------------------------

def build_bf16_matmul(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("bf16_matmul", n)
    a32 = rng.standard_normal((n, n), dtype=np.float32)
    b32 = rng.standard_normal((n, n), dtype=np.float32)

    # Round through Orthon's own bfloat16 conversion first, then hand the
    # already-bfloat16-representable float32 values to PyTorch. Both frameworks
    # therefore multiply bit-identical inputs; neither pays a conversion the
    # other does not.
    a_cpu = ortho.from_numpy(a32, dtype="bfloat16")
    b_cpu = ortho.from_numpy(b32, dtype="bfloat16")
    a_exact = ortho_to_numpy(a_cpu)
    b_exact = ortho_to_numpy(b_cpu)

    ao = ortho.to(a_cpu, ORTHON_DEVICE)
    bo = ortho.to(b_cpu, ORTHON_DEVICE)
    at = torch.from_numpy(a_exact).to(TORCH_DEVICE).to(torch.bfloat16)
    bt = torch.from_numpy(b_exact).to(TORCH_DEVICE).to(torch.bfloat16)

    def check():
        result = compare(torch_to_numpy(at @ bt), ortho_to_numpy(ortho.matmul(ao, bo)),
                         tol=3e-2)
        result["detail"] = "bfloat16 in, bfloat16 out, float32 accumulate on both sides"
        return result

    return Case(
        benchmark="bf16_matmul",
        shape=(n, n),
        label=f"{n}x{n}",
        dtype="bfloat16",
        mode="forward",
        note=f"Orthon dispatch: {_matmul_path(ao, bo)}",
        orthon_fn=lambda: ortho.matmul(ao, bo),
        torch_fn=lambda: at @ bt,
        check=check,
    )


# ---- float32 matmul -------------------------------------------------------

def build_fp32_matmul(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("fp32_matmul", n)
    a32 = rng.standard_normal((n, n), dtype=np.float32)
    b32 = rng.standard_normal((n, n), dtype=np.float32)

    ao, bo = _ortho_cuda(a32), _ortho_cuda(b32)
    at = torch.from_numpy(a32).to(TORCH_DEVICE)
    bt = torch.from_numpy(b32).to(TORCH_DEVICE)

    tf32 = torch.backends.cuda.matmul.allow_tf32

    def check():
        result = compare(torch_to_numpy(at @ bt), ortho_to_numpy(ortho.matmul(ao, bo)),
                         tol=1e-2 if tf32 else 1e-4)
        result["detail"] = ("PyTorch TF32 enabled - reduced-precision reference"
                            if tf32 else
                            "both sides strict float32 (TF32 disabled)")
        return result

    return Case(
        benchmark="fp32_matmul",
        shape=(n, n),
        label=f"{n}x{n}",
        dtype="float32",
        mode="forward",
        note=(f"Orthon dispatch: {_matmul_path(ao, bo)}; "
              f"PyTorch TF32 {'ON' if tf32 else 'OFF'}"),
        orthon_fn=lambda: ortho.matmul(ao, bo),
        torch_fn=lambda: at @ bt,
        check=check,
    )


# ---- softmax --------------------------------------------------------------

def build_softmax(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("softmax", rows, cols)
    x32 = rng.standard_normal((rows, cols), dtype=np.float32)

    xo = _ortho_cuda(x32)
    xt = torch.from_numpy(x32).to(TORCH_DEVICE)

    def check():
        result = compare(torch_to_numpy(F.softmax(xt, dim=-1)),
                         ortho_to_numpy(ortho.softmax(xo, -1)), tol=1e-6)
        result["detail"] = "row-wise softmax over the last dimension"
        return result

    return Case(
        benchmark="softmax",
        shape=(rows, cols),
        label=f"{rows}x{cols}",
        dtype="float32",
        mode="forward",
        note="Orthon fused row-wise CUDA kernel vs PyTorch eager softmax",
        orthon_fn=lambda: ortho.softmax(xo, -1),
        torch_fn=lambda: F.softmax(xt, dim=-1),
        check=check,
    )


# ---- cross entropy, forward only ------------------------------------------

def build_cross_entropy(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("cross_entropy", rows, cols)
    logits32 = rng.standard_normal((rows, cols), dtype=np.float32)
    targets_np = rng.integers(0, cols, size=rows, dtype=np.int64)
    targets_list = targets_np.tolist()

    xo = _ortho_cuda(logits32)
    xt = torch.from_numpy(logits32).to(TORCH_DEVICE)
    tt = torch.from_numpy(targets_np).to(TORCH_DEVICE)

    def check():
        result = compare(np.array([float(F.cross_entropy(xt, tt).item())]),
                         np.array([float(ortho.cross_entropy(xo, targets_list).item())]),
                         tol=1e-5)
        result["detail"] = "mean reduction, dense class-index targets, float32 logits"
        return result

    return Case(
        benchmark="cross_entropy",
        shape=(rows, cols),
        label=f"{rows}x{cols}",
        dtype="float32",
        mode="forward",
        note=("Orthon takes host class indices (list[int]) and uploads them per "
              "call; PyTorch reads a pre-placed int64 CUDA tensor"),
        orthon_fn=lambda: ortho.cross_entropy(xo, targets_list),
        torch_fn=lambda: F.cross_entropy(xt, tt),
        check=check,
    )


# ---- cross entropy, forward + backward ------------------------------------

def build_cross_entropy_backward(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("cross_entropy_backward", rows, cols)
    logits32 = rng.standard_normal((rows, cols), dtype=np.float32)
    targets_np = rng.integers(0, cols, size=rows, dtype=np.int64)
    targets_list = targets_np.tolist()

    xo = _ortho_cuda(logits32)
    xo.requires_grad_(True)
    xt = torch.from_numpy(logits32).to(TORCH_DEVICE).requires_grad_(True)
    tt = torch.from_numpy(targets_np).to(TORCH_DEVICE)

    def orthon_step():
        ortho.cross_entropy(xo, targets_list).backward()

    def torch_step():
        F.cross_entropy(xt, tt).backward()

    def check():
        # Fresh tensors: the timed tensors accumulate gradients across every
        # iteration, so they cannot answer a gradient-equality question.
        co = _ortho_cuda(logits32)
        co.requires_grad_(True)
        ct = torch.from_numpy(logits32).to(TORCH_DEVICE).requires_grad_(True)
        loss_o = ortho.cross_entropy(co, targets_list)
        loss_t = F.cross_entropy(ct, tt)
        loss_o.backward()
        loss_t.backward()
        forward = compare(np.array([float(loss_t.item())]),
                          np.array([float(loss_o.item())]), tol=1e-5)
        grads = compare(torch_to_numpy(ct.grad), ortho_to_numpy(co.grad), tol=1e-5)
        merged = grads if grads["rel"] > forward["rel"] else forward
        merged = dict(merged)
        merged["ok"] = forward["ok"] and grads["ok"]
        merged["detail"] = (f"forward rel {forward['rel']:.2e}, "
                            f"d(loss)/d(logits) rel {grads['rel']:.2e}")
        return merged

    return Case(
        benchmark="cross_entropy_backward",
        shape=(rows, cols),
        label=f"{rows}x{cols}",
        dtype="float32",
        mode="forward+backward",
        note="loss forward plus d(loss)/d(logits); gradients accumulate on both sides",
        orthon_fn=orthon_step,
        torch_fn=torch_step,
        check=check,
    )


# ---- fused linear + relu --------------------------------------------------

def build_fused_linear_relu(shape: tuple) -> Case:
    m, k, n = shape
    rng = _seeded("fused_linear_relu", m, k, n)
    x32 = rng.standard_normal((m, k), dtype=np.float32)
    w32 = rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.05)
    b32 = rng.standard_normal((n,), dtype=np.float32)

    xo, wo, bo = _ortho_cuda(x32), _ortho_cuda(w32), _ortho_cuda(b32)
    xt = torch.from_numpy(x32).to(TORCH_DEVICE)
    wt = torch.from_numpy(w32).to(TORCH_DEVICE)
    bt = torch.from_numpy(b32).to(TORCH_DEVICE)

    # PyTorch's Linear weight is (out_features, in_features) = N x K, the
    # transpose of Orthon's K x N. F.linear on its native layout reaches a
    # cuBLAS(Lt) addmm with a fused bias epilogue - two dispatches, not three -
    # and that is the fastest idiomatic eager spelling of Linear + ReLU, so it
    # is what Orthon's fused kernel is measured against.
    #
    # The transpose and the contiguous copy are done HERE, once, before any
    # timing. Putting a layout conversion inside the timed region would charge
    # PyTorch for work its own nn.Linear never does at call time. Both sides
    # hold the same mathematical weight; each holds it in its own framework's
    # preferred physical layout.
    wt_linear = wt.t().contiguous()

    def torch_eager():
        return F.relu(F.linear(xt, wt_linear, bt))

    # Retained as a labelled diagnostic only. It is the explicit
    # matmul + bias + ReLU composition - three dispatches - and it is NOT the
    # baseline: it does not set the speedup, the verdict, or any published
    # claim. It exists so the matrix can show how much of Orthon's margin was
    # ever an artefact of the eager spelling rather than of the fusion.
    def torch_explicit_composition():
        return torch.relu(xt @ wt + bt)

    def check():
        reference = torch_to_numpy(torch_eager())
        result = compare(reference,
                         ortho_to_numpy(ortho.fused_linear_relu(xo, wo, bo)),
                         tol=1e-4)
        # The two PyTorch spellings must agree with each other too, or the
        # diagnostic row would be timing a different computation.
        spelling = compare(reference,
                           torch_to_numpy(torch_explicit_composition()),
                           tol=1e-4)
        result["ok"] = bool(result["ok"] and spelling["ok"])
        result["detail"] = (
            "relu(F.linear(x, W^T, b)); PyTorch-native N x K weight prepared "
            "before timing")
        return result

    def compile_factory():
        """A compiled variant of the SAME eager expression, never merged into
        the eager row. Compilation happens on the first call, which is inside
        the warmup, so no compile time reaches a timed sample."""
        compiled = torch.compile(torch_eager)
        compiled()
        torch_sync()
        return compiled

    return Case(
        benchmark="fused_linear_relu",
        shape=(m, k, n),
        label=f"M={m} K={k} N={n}",
        dtype="float32",
        mode="forward",
        note=("Orthon runs ONE fused CUDA kernel; PyTorch eager runs "
              "F.relu(F.linear(x, W, b)) on its native N x K weight, prepared "
              "before timing"),
        orthon_fn=lambda: ortho.fused_linear_relu(xo, wo, bo),
        torch_fn=torch_eager,
        check=check,
        torch_compile_factory=compile_factory,
    )


# ---- conv2d ---------------------------------------------------------------

def build_conv2d(shape: tuple) -> Case:
    n, c_in, hw, c_out = shape
    k, pad = 3, 1
    rng = _seeded("conv2d", n, c_in, hw, c_out)
    x32 = rng.standard_normal((n, c_in, hw, hw), dtype=np.float32)
    w32 = rng.standard_normal((c_out, c_in, k, k), dtype=np.float32) * np.float32(0.1)
    b32 = rng.standard_normal((c_out,), dtype=np.float32)

    xo, wo, bo = _ortho_cuda(x32), _ortho_cuda(w32), _ortho_cuda(b32)
    xt = torch.from_numpy(x32).to(TORCH_DEVICE)
    wt = torch.from_numpy(w32).to(TORCH_DEVICE)
    bt = torch.from_numpy(b32).to(TORCH_DEVICE)

    def check():
        result = compare(
            torch_to_numpy(F.conv2d(xt, wt, bt, stride=1, padding=pad)),
            ortho_to_numpy(ortho.conv2d(xo, wo, bo, 1, 1, pad, pad)),
            tol=3e-4)
        result["detail"] = f"{k}x{k} stride 1 padding {pad}, NCHW"
        return result

    return Case(
        benchmark="conv2d",
        shape=(n, c_in, hw, c_out),
        label=f"N={n} C={c_in} {hw}x{hw} -> {c_out}ch 3x3",
        dtype="float32",
        mode="forward",
        note="Orthon handwritten CUDA conv vs PyTorch eager (cuDNN)",
        orthon_fn=lambda: ortho.conv2d(xo, wo, bo, 1, 1, pad, pad),
        torch_fn=lambda: F.conv2d(xt, wt, bt, stride=1, padding=pad),
        check=check,
    )


# ---- elementwise add ------------------------------------------------------

def build_elementwise_add(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("elementwise_add", n)
    a32 = rng.standard_normal((n,), dtype=np.float32)
    b32 = rng.standard_normal((n,), dtype=np.float32)

    ao, bo = _ortho_cuda(a32), _ortho_cuda(b32)
    at = torch.from_numpy(a32).to(TORCH_DEVICE)
    bt = torch.from_numpy(b32).to(TORCH_DEVICE)

    def check():
        result = compare(torch_to_numpy(at + bt),
                         ortho_to_numpy(ortho.add(ao, bo)), tol=1e-7)
        result["detail"] = "bandwidth-bound elementwise add"
        return result

    return Case(
        benchmark="elementwise_add",
        shape=(n,),
        label=f"{n:,} elements",
        dtype="float32",
        mode="forward",
        note="memory-bandwidth bound on both sides",
        orthon_fn=lambda: ortho.add(ao, bo),
        torch_fn=lambda: at + bt,
        check=check,
    )


# ---- full reduction -------------------------------------------------------

def build_reduce_sum(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("reduce_sum", n)
    # Uniform positive values: a near-zero sum would make the relative
    # tolerance meaningless without saying anything about the kernel.
    a32 = rng.random((n,), dtype=np.float32)

    ao = _ortho_cuda(a32)
    at = torch.from_numpy(a32).to(TORCH_DEVICE)

    def check():
        result = compare(torch_to_numpy(torch.sum(at).reshape(1)),
                         ortho_to_numpy(ortho.sum(ao)).reshape(1), tol=1e-5)
        result["detail"] = "full reduction to a scalar; reduction order differs"
        return result

    return Case(
        benchmark="reduce_sum",
        shape=(n,),
        label=f"{n:,} elements",
        dtype="float32",
        mode="forward",
        note="full sum reduction",
        orthon_fn=lambda: ortho.sum(ao),
        torch_fn=lambda: torch.sum(at),
        check=check,
    )


BENCHMARKS = {
    "bf16_matmul": {
        "build": build_bf16_matmul,
        "shapes": [(256,), (512,), (1024,), (2048,)],
        "shape_help": "N (square N x N matmul)",
        "title": "BF16 matmul",
    },
    "fp32_matmul": {
        "build": build_fp32_matmul,
        "shapes": [(256,), (512,), (1024,), (2048,)],
        "shape_help": "N (square N x N matmul)",
        "title": "FP32 matmul",
    },
    "softmax": {
        "build": build_softmax,
        "shapes": [(256, 512), (512, 1024), (1024, 2048), (2048, 4096)],
        "shape_help": "ROWS COLS",
        "title": "Softmax",
    },
    "cross_entropy": {
        "build": build_cross_entropy,
        "shapes": [(512, 1024), (1024, 2048), (2048, 4096), (4096, 4096),
                   (8192, 512)],
        "shape_help": "BATCH CLASSES",
        "title": "Cross entropy (forward)",
    },
    "cross_entropy_backward": {
        "build": build_cross_entropy_backward,
        "shapes": [(512, 1024), (1024, 2048), (2048, 4096), (4096, 4096)],
        "shape_help": "BATCH CLASSES",
        "title": "Cross entropy (forward + backward)",
    },
    "fused_linear_relu": {
        "build": build_fused_linear_relu,
        "shapes": [(128, 256, 128), (256, 256, 256), (256, 512, 256),
                   (512, 512, 512), (512, 1024, 512), (1024, 1024, 1024)],
        "shape_help": "M K N",
        "title": "Fused Linear + ReLU",
    },
    "conv2d": {
        "build": build_conv2d,
        "shapes": [(128, 3, 32, 32), (64, 32, 32, 32), (32, 64, 16, 64)],
        "shape_help": "N C_IN HW C_OUT (3x3 stride 1 padding 1)",
        "title": "Conv2d 3x3",
    },
    "elementwise_add": {
        "build": build_elementwise_add,
        "shapes": [(1_000_000,), (4_000_000,), (16_000_000,)],
        "shape_help": "N (element count)",
        "title": "Elementwise add",
    },
    "reduce_sum": {
        "build": build_reduce_sum,
        "shapes": [(1_000_000,), (4_000_000,), (16_000_000,)],
        "shape_help": "N (element count)",
        "title": "Sum reduction",
    },
}

# Order the report the way a reader should read it: the specialized paths
# first, the general ones after, the expected losses last.
BENCHMARK_ORDER = [
    "cross_entropy",
    "cross_entropy_backward",
    "softmax",
    "fused_linear_relu",
    "bf16_matmul",
    "elementwise_add",
    "reduce_sum",
    "fp32_matmul",
    "conv2d",
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def global_warmup(seconds: float = 1.0) -> None:
    """Bring the GPU out of its idle clock state before anything is measured."""
    a = torch.randn(1024, 1024, device=TORCH_DEVICE)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            a = a @ a * 0.0001
    torch_sync()
    ao = _ortho_cuda(np.ones((512, 512), dtype=np.float32))
    for _ in range(50):
        ortho.matmul(ao, ao)
    ortho_sync()
    del a


def run_case(case: Case, cfg: TimingConfig, want_compile: bool) -> Result:
    check = case.check()

    for _ in range(cfg.warmup):
        case.orthon_fn()
    ortho_sync()
    for _ in range(cfg.warmup):
        case.torch_fn()
    torch_sync()

    est_o = _estimate_ms(case.orthon_fn, ortho_sync)
    est_t = _estimate_ms(case.torch_fn, torch_sync)
    inner = _choose_inner(est_o, est_t, cfg)

    orthon_samples, torch_samples = [], []
    for round_index in range(cfg.samples):
        if round_index % 2 == 0:
            orthon_samples.append(_batch_ms(case.orthon_fn, inner, ortho_sync) / inner)
            torch_samples.append(_batch_ms(case.torch_fn, inner, torch_sync) / inner)
        else:
            torch_samples.append(_batch_ms(case.torch_fn, inner, torch_sync) / inner)
            orthon_samples.append(_batch_ms(case.orthon_fn, inner, ortho_sync) / inner)

    compiled_samples = None
    compile_error = None
    if want_compile:
        if case.torch_compile_factory is None:
            compile_error = "no compiled variant defined for this benchmark"
        else:
            try:
                fn = case.torch_compile_factory()
                for _ in range(max(cfg.warmup, 5)):
                    fn()
                torch_sync()
                compiled_samples = [
                    _batch_ms(fn, inner, torch_sync) / inner
                    for _ in range(cfg.samples)
                ]
            except Exception as exc:      # torch.compile is optional, never fatal
                compile_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    return Result(case=case, inner=inner, orthon_samples=orthon_samples,
                  torch_samples=torch_samples, check=check,
                  compiled_samples=compiled_samples,
                  compile_error=compile_error).finalize()


def select_cases(args) -> list:
    names = BENCHMARK_ORDER if args.benchmark is None else list(args.benchmark)
    for name in names:
        if name not in BENCHMARKS:
            sys.exit(f"unknown benchmark {name!r}; try --list")
    wanted_shape = tuple(args.shape) if args.shape else None

    cases = []
    for name in names:
        spec = BENCHMARKS[name]
        for shape in spec["shapes"]:
            if wanted_shape is not None and tuple(shape) != wanted_shape:
                continue
            cases.append((name, shape))
    if wanted_shape is not None and not cases:
        sys.exit(f"no case matches --shape {' '.join(str(s) for s in wanted_shape)}; "
                 f"try --list")
    return cases


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_check_table(results: list) -> None:
    print("=" * 86)
    print("  Numerical equivalence (PyTorch eager is the reference)")
    print("=" * 86)
    header = f"  {'benchmark':<24}{'shape':<30}{'max rel diff':>14}  {'tol':>9}  status"
    print(header)
    print("  " + "-" * 82)
    for result in results:
        c = result.check
        status = "PASS" if c.get("ok") else "FAIL"
        print(f"  {result.case.benchmark:<24}{result.case.label:<30}"
              f"{c['rel']:>14.2e}  {c['tol']:>9.1e}  {status}")
    print()


def print_results(results: list) -> None:
    by_benchmark = {}
    for result in results:
        by_benchmark.setdefault(result.case.benchmark, []).append(result)

    for name in BENCHMARK_ORDER:
        group = by_benchmark.get(name)
        if not group:
            continue
        spec = BENCHMARKS[name]
        first = group[0].case
        print("=" * 86)
        print(f"  {spec['title']}  |  {first.dtype}  |  {first.mode}")
        print(f"  {first.note}")
        print("=" * 86)
        print(f"  {'shape':<30}{'Orthon ms':>11}{'IQR':>9}"
              f"{'PyTorch ms':>12}{'IQR':>9}{'speedup':>10}  verdict")
        print("  " + "-" * 82)
        for result in group:
            print(f"  {result.case.label:<30}"
                  f"{result.orthon_median:>11.4f}{result.orthon_iqr:>9.4f}"
                  f"{result.torch_median:>12.4f}{result.torch_iqr:>9.4f}"
                  f"{result.speedup:>9.2f}x  {VERDICT_TEXT[result.verdict]}"
                  + ("" if result.stability == "stable" else "  [noisy]"))
            if result.compiled_median is not None:
                print(f"  {'  (torch.compile)':<30}{'':>11}{'':>9}"
                      f"{result.compiled_median:>12.4f}"
                      f"{'':>9}{result.torch_median / result.compiled_median:>9.2f}x"
                      f"  vs PyTorch eager, compile time excluded")
            elif result.compile_error:
                print(f"  {'  (torch.compile)':<30}unavailable: {result.compile_error}")
        print()


def print_summary(results: list) -> None:
    buckets = {"orthon": [], "pytorch": [], "tie": [], "unreliable": [], "invalid": []}
    for result in results:
        buckets[result.verdict].append(result)

    print("=" * 86)
    print("  Summary")
    print("=" * 86)
    print(f"  cases measured             {len(results)}")
    print(f"  Orthon >= +10% faster      {len(buckets['orthon'])}")
    print(f"  PyTorch >= +10% faster     {len(buckets['pytorch'])}")
    print(f"  within +/-10% (similar)    {len(buckets['tie'])}")
    print(f"  ordering unreliable        {len(buckets['unreliable'])}")
    print(f"  numerical check failed     {len(buckets['invalid'])}")
    print()

    if buckets["orthon"]:
        print("  Stable Orthon wins (>= 10%, ordering reliable):")
        for result in sorted(buckets["orthon"], key=lambda r: -r.speedup):
            flag = "" if result.stability == "stable" else "   [noisy]"
            print(f"    {result.speedup:>6.2f}x  {result.case.benchmark} "
                  f"{result.case.label}{flag}")
        print()
    if buckets["invalid"]:
        print("  INVALID rows carry no performance claim:")
        for result in buckets["invalid"]:
            print(f"    {result.case.benchmark} {result.case.label}: "
                  f"rel {result.check['rel']:.2e} > tol {result.check['tol']:.1e}")
        print()


def print_highlight(result: Result, env: dict) -> None:
    case = result.case
    title = BENCHMARKS[case.benchmark]["title"]
    width = 60
    print()
    print("=" * width)
    print("  Orthon vs. PyTorch eager")
    print(f"  {title}  |  {case.label}  |  {case.dtype}")
    print(f"  GPU: {env['gpu']}")
    print("=" * width)
    print(f"  Orthon      {result.orthon_median:8.3f} ms")
    print(f"  PyTorch     {result.torch_median:8.3f} ms")
    print(f"  Speedup     {result.speedup:8.2f}x")
    print("=" * width)
    print(f"  median of {len(result.orthon_samples)} alternating samples, "
          f"{result.inner} iterations each")
    print(f"  max relative difference vs PyTorch: {result.check['rel']:.2e} "
          f"(tolerance {result.check['tol']:.0e})")
    print("=" * width)
    print()


def write_markdown(path: str, env: dict, results: list) -> None:
    lines = ["# Orthon vs. PyTorch - generated result matrix", ""]
    lines.append(f"Generated by `benchmarks/compare_pytorch.py` on "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")
    lines.append("| Environment | Value |")
    lines.append("| --- | --- |")
    for key, value in env.items():
        if value is None:
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")

    by_benchmark = {}
    for result in results:
        by_benchmark.setdefault(result.case.benchmark, []).append(result)

    for name in BENCHMARK_ORDER:
        group = by_benchmark.get(name)
        if not group:
            continue
        spec = BENCHMARKS[name]
        first = group[0].case
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append(f"dtype `{first.dtype}`, {first.mode}. {first.note}")
        lines.append("")
        lines.append("| Shape | Orthon median ms | Orthon IQR | PyTorch median ms "
                     "| PyTorch IQR | Speedup | Verdict | Max rel diff |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for result in group:
            verdict = VERDICT_TEXT[result.verdict]
            if result.stability != "stable":
                verdict += " (noisy)"
            lines.append(
                f"| {result.case.label} | {result.orthon_median:.4f} | "
                f"{result.orthon_iqr:.4f} | {result.torch_median:.4f} | "
                f"{result.torch_iqr:.4f} | {result.speedup:.2f}x | {verdict} | "
                f"{result.check['rel']:.2e} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  markdown written to {path}")


def write_json(path: str, env: dict, results: list) -> None:
    payload = {
        "environment": env,
        "threshold": WIN_THRESHOLD,
        "results": [
            {
                "benchmark": r.case.benchmark,
                "shape": list(r.case.shape),
                "label": r.case.label,
                "dtype": r.case.dtype,
                "mode": r.case.mode,
                "note": r.case.note,
                "inner_iterations": r.inner,
                "orthon_samples_ms": r.orthon_samples,
                "torch_samples_ms": r.torch_samples,
                "orthon_median_ms": r.orthon_median,
                "orthon_iqr_ms": r.orthon_iqr,
                "torch_median_ms": r.torch_median,
                "torch_iqr_ms": r.torch_iqr,
                "speedup": r.speedup,
                "speedup_band": list(r.band),
                "verdict": r.verdict,
                "stability": r.stability,
                "numerical_check": r.check,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"  json written to {path}")


def print_list() -> None:
    print("Available benchmarks (--benchmark NAME, --shape follows the shape form):")
    print()
    for name in BENCHMARK_ORDER:
        spec = BENCHMARKS[name]
        print(f"  {name}")
        print(f"      {spec['title']}   --shape {spec['shape_help']}")
        for shape in spec["shapes"]:
            print(f"        --shape {' '.join(str(s) for s in shape)}")
        print()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_pytorch.py",
        description=("Reproducible Orthon vs. PyTorch eager comparison on one "
                     "CUDA device. Inputs are pre-placed on the GPU; every case "
                     "proves numerical equivalence before it is timed."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python benchmarks/compare_pytorch.py\n"
            "  python benchmarks/compare_pytorch.py --benchmark cross_entropy\n"
            "  python benchmarks/compare_pytorch.py --benchmark bf16_matmul "
            "--shape 1024\n"
            "  python benchmarks/compare_pytorch.py --highlight cross_entropy "
            "--shape 2048 4096\n"
        ),
    )
    parser.add_argument("--list", action="store_true",
                        help="list benchmarks and their shapes, then exit")
    parser.add_argument("--benchmark", nargs="+", metavar="NAME",
                        help="run only these benchmarks (default: all)")
    parser.add_argument("--shape", nargs="+", type=int, metavar="INT",
                        help="run only the case with exactly this shape")
    parser.add_argument("--highlight", metavar="NAME",
                        help="print the screenshot-ready card for one benchmark "
                             "(combine with --shape to pick the case)")
    parser.add_argument("--check-only", action="store_true",
                        help="run the numerical equivalence checks and stop")
    parser.add_argument("--samples", type=int, default=11,
                        help="independent samples per framework (default: 11)")
    parser.add_argument("--warmup", type=int, default=25,
                        help="warmup iterations per framework per case (default: 25)")
    parser.add_argument("--target-ms", type=float, default=40.0,
                        help="GPU work per sample for the faster framework, ms "
                             "(default: 40)")
    parser.add_argument("--max-sample-ms", type=float, default=500.0,
                        help="cap on a single sample for the slower framework, ms "
                             "(default: 500)")
    parser.add_argument("--inner", type=int, default=None,
                        help="fix the per-sample iteration count instead of "
                             "deriving it")
    parser.add_argument("--tf32", action="store_true",
                        help="allow PyTorch TF32 for float32 matmul/conv; rows are "
                             "labelled accordingly (default: strict float32)")
    parser.add_argument("--torch-compile", action="store_true",
                        help="additionally time torch.compile where a case "
                             "provides a compiled variant; never merged into the "
                             "eager row")
    parser.add_argument("--markdown", metavar="PATH",
                        help="write the full matrix as a Markdown table")
    parser.add_argument("--json", metavar="PATH",
                        help="write the full matrix, including raw samples, as JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print_list()
        return 0

    if not ortho.cuda_enabled():
        sys.exit("this comparison needs a CUDA-enabled Orthon build "
                 "(orthon.cuda_enabled() is False)")
    if not torch.cuda.is_available():
        sys.exit("this comparison needs a CUDA-enabled PyTorch build")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    env = collect_environment()
    print()
    print_environment(env)

    if args.highlight:
        args.benchmark = [args.highlight]

    selected = select_cases(args)

    cfg = TimingConfig(warmup=args.warmup, samples=args.samples,
                       target_ms=args.target_ms,
                       max_sample_ms=args.max_sample_ms,
                       fixed_inner=args.inner)

    if not args.check_only:
        print("  warming up the GPU ...")
        global_warmup()

    results = []
    for name, shape in selected:
        case = BENCHMARKS[name]["build"](tuple(shape))
        if args.check_only:
            check = case.check()
            status = "PASS" if check["ok"] else "FAIL"
            print(f"  {name:<24}{case.label:<30}rel {check['rel']:>10.2e}  "
                  f"tol {check['tol']:.1e}  {status}")
            continue
        print(f"  measuring {name} {case.label} ...", flush=True)
        results.append(run_case(case, cfg, args.torch_compile))

    if args.check_only:
        return 0

    print()
    print_check_table(results)

    if args.highlight:
        if len(results) == 1:
            print_highlight(results[0], env)
        else:
            print_results(results)
            print("  --highlight matched more than one case; add --shape to pick "
                  "exactly one.")
            return 0
    else:
        print_results(results)
        print_summary(results)

    if args.markdown:
        write_markdown(args.markdown, env, results)
    if args.json:
        write_json(args.json, env, results)

    failed = [r for r in results if r.verdict == "invalid"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
