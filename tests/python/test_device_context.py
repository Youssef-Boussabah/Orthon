"""CUDA calls select and restore the Tensor's exact device.

Multi-device cases require two GPUs; source fences protect unconditional
DeviceGuard usage even when hardware gates skip those cases."""
from tests.support.capabilities import orthon_cuda_available
import os
import re

import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
requires_2_gpus = pytest.mark.skipif(
    N_GPUS < 2, reason="needs >=2 CUDA devices (this machine has %d)" % N_GPUS)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CUDA_SRC_DIR = os.path.join(REPO, "src", "ops", "cuda")


def _vals(shape, ph, sc=1.0):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * ph) * sc + np.sin(i * 0.37) * 0.5 + 0.2 * (i % 4)).reshape(shape)


X34 = _vals((3, 4), 1.7)
X44 = _vals((4, 4), 1.1)
UP34 = _vals((3, 4), 2.3, 1.4)


def _d(a, dev="cpu"):
    t = ortho.from_numpy(np.ascontiguousarray(np.array(a, dtype=np.float32)))
    return t.to(dev) if dev != "cpu" else t


def _current():
    """Observe the process's current CUDA device."""
    return torch.cuda.current_device()


def _read(rel):
    rel = rel.replace("src/ops/cuda/", "src/backends/cuda/").replace("src/ops/cpu/", "src/ops/")
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def _strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


# exact output/gradient device is preserved
OPS = {
    "unary_relu": lambda x: ortho.relu(x),
    "unary_tanh": lambda x: ortho.tanh(x),
    "unary_neg": lambda x: ortho.neg(x),
    "unary_exp": lambda x: ortho.exp(x),
    "binary_add": lambda x: ortho.add(x, x),
    "binary_mul": lambda x: ortho.mul(x, x),
    "reduce_sum": lambda x: ortho.sum(x, 1, True),
    "reduce_mean": lambda x: ortho.mean(x, 0, True),
    "reduce_max": lambda x: ortho.max(x, 1, True),
    "softmax": lambda x: ortho.softmax(x, -1),
    "log_softmax": lambda x: ortho.log_softmax(x, -1),
    "reshape": lambda x: ortho.reshape(x, [12]),
    "transpose": lambda x: ortho.transpose(x, 0, 1),
    "contiguous": lambda x: ortho.contiguous(ortho.transpose(x, 0, 1)),
    "index": lambda x: x[1],
    "cast_roundtrip": lambda x: ortho.cast(ortho.cast(x, "float16"), "float32"),
}


@requires_cuda
@pytest.mark.parametrize("name", sorted(OPS))
def test_output_lands_on_the_exact_input_device(name):
    x = _d(X34, "cuda")
    y = OPS[name](x)
    assert y.device == x.device
    assert str(y.device) == "device('cuda:0')"   # exact, not merely is_cuda


@requires_cuda
def test_matmul_output_device_is_exact():
    a = _d(X44, "cuda")
    y = ortho.matmul(a, a)
    assert y.device == a.device
    assert str(y.device) == "device('cuda:0')"


@requires_cuda
@pytest.mark.parametrize("name", ["unary_relu", "unary_tanh", "binary_mul",
                                  "reduce_sum", "softmax", "log_softmax", "index"])
def test_gradient_lands_on_the_exact_input_device(name):
    x = _d(X34, "cuda")
    x.requires_grad_(True)
    ortho.sum(OPS[name](x) * _d(np.ones(OPS[name](_d(X34, "cuda")).shape), "cuda")).backward()
    assert x.grad.device == x.device
    assert str(x.grad.device) == "device('cuda:0')"


@requires_cuda
def test_conv2d_and_pooling_output_device_is_exact():
    x = _d(_vals((1, 1, 6, 6), 0.9), "cuda")
    w = _d(_vals((2, 1, 3, 3), 1.3), "cuda")
    b = _d(np.zeros(2), "cuda")
    y = ortho.conv2d(x, w, b, 1, 1, 0, 0)
    p = ortho.maxpool2d(y, 2, 2, 2, 2, 0, 0)
    assert str(y.device) == "device('cuda:0')"
    assert str(p.device) == "device('cuda:0')"


@requires_cuda
def test_tensor_to_roundtrip_preserves_exact_devices():
    cpu = _d(X34)
    gpu = cpu.to("cuda")
    assert str(gpu.device) == "device('cuda:0')"
    back = gpu.to("cpu")
    assert str(back.device) == "device('cpu')"
    assert back.tolist() == pytest.approx(cpu.tolist(), abs=0)
    same = gpu.to("cuda")
    assert str(same.device) == "device('cuda:0')"
    # Public to() is differentiable - a tracked source yields a tracked
    # ToNode output. Internal snapshot paths get their detachment from an
    # explicit no_grad rather than from to().
    rg = _d(X34, "cuda")
    rg.requires_grad_(True)
    tracked = rg.to("cuda")
    assert tracked.requires_grad is True
    assert tracked._has_grad_fn


# Current-device restoration; multi-device cases distinguish ordinals.
@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("name", sorted(OPS))
def test_operation_does_not_leak_device_context(name):
    before = _current()
    OPS[name](_d(X34, "cuda"))
    assert _current() == before


@requires_cuda
@requires_torch_cuda
def test_backward_does_not_leak_device_context():
    before = _current()
    x = _d(X34, "cuda")
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(ortho.relu(x), -1) * _d(UP34, "cuda")).backward()
    assert _current() == before


@requires_cuda
@requires_torch_cuda
def test_a_throwing_cuda_op_does_not_leak_device_context():
    """Unwinding through a DeviceGuard must restore, not abandon, the context.

    The throw is provoked by shapes that cannot broadcast, not by a
    transposed view: cuda_mul normalises non-contiguous operands and would
    accept one. The unwinding path is the same either way - cuda_mul builds
    its DeviceGuard first and throws afterwards, from cuda_broadcast_shape.
    """
    before = _current()
    a = _d(X34, "cuda")           # (3, 4)
    b = _d(_vals((5,), 0.9), "cuda")
    with pytest.raises(RuntimeError):
        ortho.mul(a, b)              # (3,4) and (5,) cannot broadcast
    assert _current() == before
    # ...and the runtime is still usable afterwards.
    assert ortho.sum(ortho.relu(_d(X34, "cuda"))).tolist()[0] > 0.0


@requires_cuda
@requires_torch_cuda
def test_memory_pool_hit_and_miss_agree_on_device_context():
    """Cache hit and cache miss must leave the same ambient device.

    A pool MISS reaches raw_allocate, which selects a device; a pool HIT
    returns earlier and touches nothing. Allocating the same size repeatedly
    exercises both paths, and the ambient device must be identical after each.
    """
    before = _current()
    seen = set()
    for _ in range(8):
        t = _d(X34, "cuda")          # first iterations miss, later ones hit
        del t
        seen.add(_current())
    assert seen == {before}


@requires_cuda
@requires_torch_cuda
@requires_2_gpus
@pytest.mark.parametrize("name", ["unary_relu", "binary_add", "reduce_sum",
                                  "softmax", "index"])
def test_op_on_a_second_gpu_restores_the_first(name):
    torch.cuda.set_device(0)
    x = _d(X34, "cuda:1")
    y = OPS[name](x)
    assert str(y.device) == "device('cuda:1')"
    assert _current() == 0


@requires_cuda
@requires_torch_cuda
@requires_2_gpus
def test_tensor_to_second_gpu_restores_the_first():
    torch.cuda.set_device(0)
    t = _d(X34).to("cuda:1")
    assert str(t.device) == "device('cuda:1')"
    assert _current() == 0
    back = t.to("cpu")
    assert _current() == 0
    assert back.tolist() == pytest.approx(_d(X34).tolist(), abs=0)


@requires_cuda
@requires_torch_cuda
@requires_2_gpus
def test_backward_on_a_second_gpu_restores_the_first():
    torch.cuda.set_device(0)
    x = _d(X34, "cuda:1")
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1) * _d(UP34, "cuda:1")).backward()
    assert str(x.grad.device) == "device('cuda:1')"
    assert _current() == 0


# DeviceGuard ownership across asynchronous CUDA call sites
# Sites that legitimately keep an explicit cudaSetDevice, with the reason.
ALLOWED_RAW_SETTERS = {
    # The guard itself: this is where the primitive is supposed to live.
    "src/device/device_guard.cpp",
    # NCCL work is asynchronous: wait(), is_completed() and the destructor can
    # run CUDA APIs long after the enqueuing call returned, so each method
    # selects its own device rather than relying on a scope that has exited.
    "bindings/nccl_bindings.cpp",
}


def _all_sources():
    out = []
    for sub in ("src", "bindings"):
        for root, _dirs, files in os.walk(os.path.join(REPO, sub)):
            for f in files:
                if f.endswith((".cpp", ".cu", ".h")):
                    p = os.path.join(root, f)
                    out.append(os.path.relpath(p, REPO).replace("\\", "/"))
    return sorted(out)


def test_no_unaudited_raw_cuda_set_device_remains():
    offenders = []
    for rel in _all_sources():
        if rel in ALLOWED_RAW_SETTERS:
            continue
        if "cudaSetDevice" in _strip_comments(_read(rel)):
            offenders.append(rel)
    assert offenders == [], (
        "raw cudaSetDevice outside the audited set: %s" % offenders)


def test_every_cuda_kernel_file_uses_the_device_guard():
    """Each .cu file that launches kernels or issues memory ops must include
    and use the guard, rather than inheriting whatever device was current."""
    missing = []
    for rel in _all_sources():
        if not rel.endswith(".cu"):
            continue
        src = _read(rel)
        body = _strip_comments(src)
        does_cuda_work = ("<<<" in body or "cudaMemcpy" in body or
                          "cudaMemset" in body or "cublas" in body)
        if does_cuda_work and "DeviceGuard guard" not in body:
            missing.append(rel)
    assert missing == [], "CUDA source without a DeviceGuard: %s" % missing


@pytest.mark.parametrize("rel,fn,dev", [
    ("src/ops/cuda/binary_kernels.cu", "cuda_add", "a.device()"),
    ("src/ops/cuda/binary_kernels.cu", "cuda_unbroadcast", "grad.device()"),
    ("src/ops/cuda/unary_kernels.cu", "cuda_relu", "t.device()"),
    ("src/ops/cuda/unary_kernels.cu", "cuda_make_contiguous", "t.device()"),
    ("src/ops/cuda/reduce_kernels.cu", "cuda_sum", "t.device()"),
    ("src/ops/cuda/softmax_kernels.cu", "cuda_softmax", "t.device()"),
    ("src/ops/cuda/cast_kernels.cu", "cuda_cast", "src.device()"),
    ("src/ops/cuda/matmul_kernels.cu", "cuda_matmul_tiled", "a.device()"),
    ("src/ops/cuda/conv2d_cuda.cu", "cuda_conv2d_forward", "input.device()"),
    ("src/ops/cuda/maxpool2d_kernels.cu", "cuda_maxpool2d_forward", "input.device()"),
    ("src/ops/cuda/embedding_kernels.cu", "cuda_embedding_forward", "weight.device()"),
    ("src/ops/cuda/nll_loss_kernels.cu", "cuda_nll_loss_forward", "log_probs.device()"),
    ("src/ops/cuda/cross_entropy_kernels.cu", "cuda_cross_entropy_loss", "logits.device()"),
    ("src/ops/cuda/fusion_kernels.cu", "cuda_fused_linear_relu", "x.device()"),
])
def test_entry_point_guards_its_authoritative_device(rel, fn, dev):
    """Spot-check that representative entry points guard on the operand the
    audit table names as authoritative, not on something incidental."""
    src = _read(rel)
    if fn == "cuda_matmul_tiled":
        # The diagnostic frontend delegates to
        # the plan executor. Device selection belongs to the actual launchers.
        from tests.support.source import function
        for executor in ("execute_custom_2d", "execute_custom_batched",
                         "cublas_gemm_2d", "cublas_gemm_batched"):
            assert "DeviceGuard guard(%s)" % dev in function(src, executor)
        return
    # The lookbehind matters: validate_cuda_fused_linear_relu CONTAINS
    # cuda_fused_linear_relu, so its own definition would otherwise match first.
    for m in re.finditer(r"(?<!\w)" + re.escape(fn) + r"\s*\([^;{]*\)\s*\{",
                         src, flags=re.S):
        head = src[m.end():m.end() + 400]
        assert "DeviceGuard guard(%s)" % dev in head, (
            "%s::%s does not guard on %s" % (rel, fn, dev))
        return
    pytest.fail("definition of %s not found in %s" % (fn, rel))


def test_memory_pool_raw_paths_use_the_guard():
    # The pool delegates raw CUDA ownership to the explicit CUDA allocator.
    body = _strip_comments(_read("src/memory/cuda_allocator.cpp"))
    alloc = body[body.index("cuda_raw_allocate("):body.index("cuda_raw_free(")]
    free = body[body.index("cuda_raw_free("):]
    assert "DeviceGuard guard(device)" in alloc
    assert "DeviceGuard guard(device)" in free
    assert "cudaSetDevice" not in alloc and "cudaSetDevice" not in free


def test_tensor_to_guards_every_cuda_branch():
    # Tensor::to uses registered pure transfer adapters.
    # DeviceGuard obligations stay in the selected execution adapter.
    body = _strip_comments(_read("src/backends/cuda/adapters.cpp"))
    to_body = body[body.index("Tensor host_to_cuda(const Tensor& self, Device target)"):]
    to_body = to_body[:to_body.index("Tensor cuda_add_adapter")]
    # One guard for the same-device copy, and one for each cross-device branch.
    assert to_body.count("DeviceGuard guard") >= 4
    # The cuda:N -> cuda:M branch uses cudaMemcpyPeer, which names both
    # device ordinals explicitly, rather than an ambient-context
    # cudaMemcpy(..., cudaMemcpyDeviceToDevice). The guard obligation is
    # asserted for whichever transfer call each branch uses.
    for kind in ("copy_h2d_snapshot", "copy_d2h_blocking",
                 "peer_copy_blocking"):
        i = to_body.index(kind)
        assert "DeviceGuard guard" in to_body[max(0, i - 400):i], (
            "%s is not preceded by a DeviceGuard" % kind)
    assert "cudaMemcpyDeviceToDevice" not in to_body, (
        "cross-GPU transfer must not fall back to an ambient-context "
        "device-to-device copy")


def test_device_guard_header_has_no_cuda_dependency():
    """CPU-only consumers must not acquire a CUDA include through this header."""
    header = _read("include/orthon/device/device_guard.h")
    assert "cuda_runtime" not in header
    assert "#include <cuda" not in header


def test_device_guard_restores_and_never_throws_from_the_destructor():
    impl = _read("src/device/device_guard.cpp")
    assert "~DeviceGuard() noexcept" in impl
    dtor = _strip_comments(impl)
    dtor = dtor[dtor.index("DeviceGuard::~DeviceGuard"):]
    assert "throw" not in dtor
    assert "cudaSetDevice(previous_device_)" in dtor
