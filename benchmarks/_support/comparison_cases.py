"""Deterministic comparison cases, framework conversion and correctness oracles."""
import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import orthon as ortho
import orthon._diagnostics as diag
import torch
import torch.nn.functional as F

ORTHON_DEVICE = "cuda:0"
TORCH_DEVICE = "cuda:0"

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
        return {"ok": False, "max_abs": None, "rel": None,
                "tol": tol, "note": f"shape mismatch {ref.shape} vs {got.shape}"}
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        return dict(ok=False, max_abs=None, rel=None, tol=tol, note="nonfinite output")
    diff = np.abs(ref - got)
    max_abs = float(diff.max()) if diff.size else 0.0
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    rel = max_abs / max(scale, 1e-12)
    return {"ok": bool(rel <= tol), "max_abs": max_abs, "rel": rel,
            "tol": tol, "note": ""}


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


def _seeded(*key) -> np.random.Generator:
    """One deterministic generator per case, derived from its identity."""
    seed_text = "|".join(str(k) for k in key).encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], "big")
    return np.random.default_rng(digest)


def _orthon_tensor(array: np.ndarray, dtype: Optional[str] = None):
    if dtype is None:
        return ortho.to(ortho.from_numpy(np.ascontiguousarray(array)), ORTHON_DEVICE)
    return ortho.to(ortho.from_numpy(np.ascontiguousarray(array), dtype=dtype),
                    ORTHON_DEVICE)


def _matmul_path(a, b) -> str:
    reporter = getattr(diag, "matmul_dispatch_path", None)
    if reporter is None:
        return "unknown"
    try:
        return reporter(a, b)
    except Exception:
        return "unknown"


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


def build_fp32_matmul(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("fp32_matmul", n)
    a32 = rng.standard_normal((n, n), dtype=np.float32)
    b32 = rng.standard_normal((n, n), dtype=np.float32)

    ao, bo = _orthon_tensor(a32), _orthon_tensor(b32)
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


def build_softmax(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("softmax", rows, cols)
    x32 = rng.standard_normal((rows, cols), dtype=np.float32)

    xo = _orthon_tensor(x32)
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
        note="Orthon row-wise softmax vs PyTorch eager softmax",
        orthon_fn=lambda: ortho.softmax(xo, -1),
        torch_fn=lambda: F.softmax(xt, dim=-1),
        check=check,
    )


def build_cross_entropy(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("cross_entropy", rows, cols)
    logits32 = rng.standard_normal((rows, cols), dtype=np.float32)
    targets_np = rng.integers(0, cols, size=rows, dtype=np.int64)
    targets_list = targets_np.tolist()

    xo = _orthon_tensor(logits32)
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
        note=("Orthon takes host class indices (list[int]) and places them per "
              "call; PyTorch reads a pre-placed int64 tensor"),
        orthon_fn=lambda: ortho.cross_entropy(xo, targets_list),
        torch_fn=lambda: F.cross_entropy(xt, tt),
        check=check,
    )


def build_cross_entropy_backward(shape: tuple) -> Case:
    rows, cols = shape
    rng = _seeded("cross_entropy_backward", rows, cols)
    logits32 = rng.standard_normal((rows, cols), dtype=np.float32)
    targets_np = rng.integers(0, cols, size=rows, dtype=np.int64)
    targets_list = targets_np.tolist()

    xo = _orthon_tensor(logits32)
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
        co = _orthon_tensor(logits32)
        co.requires_grad_(True)
        ct = torch.from_numpy(logits32).to(TORCH_DEVICE).requires_grad_(True)
        loss_o = ortho.cross_entropy(co, targets_list)
        loss_t = F.cross_entropy(ct, tt)
        loss_o.backward()
        loss_t.backward()
        forward = compare(np.array([float(loss_t.item())]),
                          np.array([float(loss_o.item())]), tol=1e-5)
        grads = compare(torch_to_numpy(ct.grad), ortho_to_numpy(co.grad), tol=1e-5)
        merged = (
            grads
            if (math.inf if grads["rel"] is None else grads["rel"])
            > (math.inf if forward["rel"] is None else forward["rel"])
            else forward
        )
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


def build_fused_linear_relu(shape: tuple) -> Case:
    m, k, n = shape
    rng = _seeded("fused_linear_relu", m, k, n)
    x32 = rng.standard_normal((m, k), dtype=np.float32)
    w32 = rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.05)
    b32 = rng.standard_normal((n,), dtype=np.float32)

    xo, wo, bo = _orthon_tensor(x32), _orthon_tensor(w32), _orthon_tensor(b32)
    xt = torch.from_numpy(x32).to(TORCH_DEVICE)
    wt = torch.from_numpy(w32).to(TORCH_DEVICE)
    bt = torch.from_numpy(b32).to(TORCH_DEVICE)

    # F.linear consumes the transposed weight layout; prepare that copy outside
    # timing. Its eager linear+ReLU call is the baseline; the composed Orthon
    # path is reported only as a diagnostic.
    wt_linear = wt.t().contiguous()

    def torch_eager():
        return F.relu(F.linear(xt, wt_linear, bt))

    # The explicit matmul + bias + ReLU composition is a diagnostic, not the
    # baseline used for speedup or verdicts.
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
        """Construct and warm a compiled variant before measurement.

        The driver synchronizes setup and reports this separately from eager.
        """
        compiled = torch.compile(torch_eager)
        compiled()
        return compiled

    return Case(
        benchmark="fused_linear_relu",
        shape=(m, k, n),
        label=f"M={m} K={k} N={n}",
        dtype="float32",
        mode="forward",
        note=("Orthon runs a fused linear/ReLU operation; PyTorch eager runs "
              "F.relu(F.linear(x, W, b)) on its native N x K weight, prepared "
              "before timing"),
        orthon_fn=lambda: ortho.fused_linear_relu(xo, wo, bo),
        torch_fn=torch_eager,
        check=check,
        torch_compile_factory=compile_factory,
    )


def build_conv2d(shape: tuple) -> Case:
    n, c_in, hw, c_out = shape
    k, pad = 3, 1
    rng = _seeded("conv2d", n, c_in, hw, c_out)
    x32 = rng.standard_normal((n, c_in, hw, hw), dtype=np.float32)
    w32 = rng.standard_normal((c_out, c_in, k, k), dtype=np.float32) * np.float32(0.1)
    b32 = rng.standard_normal((c_out,), dtype=np.float32)

    xo, wo, bo = _orthon_tensor(x32), _orthon_tensor(w32), _orthon_tensor(b32)
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
        note="Orthon convolution vs PyTorch eager on the selected device",
        orthon_fn=lambda: ortho.conv2d(xo, wo, bo, 1, 1, pad, pad),
        torch_fn=lambda: F.conv2d(xt, wt, bt, stride=1, padding=pad),
        check=check,
    )


def build_elementwise_add(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("elementwise_add", n)
    a32 = rng.standard_normal((n,), dtype=np.float32)
    b32 = rng.standard_normal((n,), dtype=np.float32)

    ao, bo = _orthon_tensor(a32), _orthon_tensor(b32)
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


def build_reduce_sum(shape: tuple) -> Case:
    n = shape[0]
    rng = _seeded("reduce_sum", n)
    # Uniform positive values: a near-zero sum would make the relative
    # tolerance meaningless without saying anything about the kernel.
    a32 = rng.random((n,), dtype=np.float32)

    ao = _orthon_tensor(a32)
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
