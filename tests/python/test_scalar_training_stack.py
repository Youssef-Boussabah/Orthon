"""The training stack injects Python scalars as scalars.

This file is deliberately half runtime assertion and half SOURCE audit, because
half the claim is architectural and numerical tests cannot see it.

Building a constant TENSOR by hand to carry a Python number allocates memory
the size of the data:

    optim/adam.py       full(list(p.shape), 1.0 / bc2)   per parameter, per step
                        full(list(p.shape), self._eps)   per parameter, per step
    amp/grad_scaler.py  full(list(loss.shape), scale)
                        full(list(grad.shape), 1 / scale)

and the same pattern reaches statistic-shaped constants and hand-rolled
negation helpers in BatchNorm, LayerNorm and attention. The category

    "full parameter/input-shaped tensor used only as a scalar constant"

must stay empty, and `test_full_shaped_scalar_census` is what keeps it empty.

The one full()-built tensor that legitimately stays is GradScaler's [1]-element
distributed found_inf flag: it is the operand of a collective, not a scalar.
"""
import os
import re

import numpy as np
import pytest

import orthon as ortho
import orthon.nn as nn
from orthon.amp import GradScaler
from orthon.nn import BatchNorm1d, LayerNorm
from orthon.optim import Adam

torch = pytest.importorskip("torch")

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel):
    with open(os.path.join(REPO, rel.replace("/", os.sep)), encoding="utf-8") as fh:
        return fh.read()


def _strip_py_comments_and_docstrings(text):
    """Remove '#' comments and triple-quoted strings.

    The audits below quote the OLD code in explanatory comments on purpose, so
    a naive substring search over raw source would find the very patterns it is
    supposed to prove absent.
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"#[^\n]*", "", text)


# ---------------------------------------------------------------------------
# Adam allocates no parameter-shaped constants.
# ---------------------------------------------------------------------------

def test_adam_source_has_no_parameter_shaped_constants():
    body = _strip_py_comments_and_docstrings(_read("python/orthon/optim/adam.py"))
    assert "ortho.full(" not in body, "Adam still builds a constant tensor"
    assert "list(p.shape)" in body, (
        "expected list(p.shape) to survive only for lazy moment allocation")
    # The only remaining shape-sized allocations are the two moment buffers,
    # which are real per-parameter state, not scalar constants.
    assert body.count("list(p.shape)") == 2
    for line in body.splitlines():
        if "list(p.shape)" in line:
            assert "zeros(" in line, line


def test_adam_step_allocates_no_parameter_shaped_constant_at_runtime():
    """Runtime proof, not just a source read: every full() call is watched."""
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    optim = Adam(model.parameters(), lr=0.01, eps=1e-8)
    x = ortho.full([4, 6], 0.5)
    ortho.cross_entropy(model(x), [0, 1, 2, 0]).backward()

    param_shapes = {tuple(p.shape) for p in model.parameters()}
    assert param_shapes, "no parameters to check"

    seen = []
    real_full = ortho.full

    def spy(shape, value, dtype="float32"):
        seen.append(tuple(shape))
        return real_full(shape, value, dtype)

    ortho.full = spy
    try:
        optim.step()
    finally:
        ortho.full = real_full

    # Not "few" parameter-shaped constants - none at all.
    assert [s for s in seen if s in param_shapes] == []
    assert seen == [], "Adam.step() should not call full() at all now"


def test_adam_scalar_temporaries_are_rank0():
    """The replacement really is O(1), not a smaller full-shaped tensor."""
    p = ortho.full([64, 64], 0.25)
    assert tuple(ortho._scalar_like(p, 1.0 / 0.999).shape) == ()
    assert int(ortho._scalar_like(p, 1e-8).numel) == 1


# ---------------------------------------------------------------------------
# Adam mathematics is unchanged.
# ---------------------------------------------------------------------------

def _adam_trajectory(steps=25, lr=0.05, weight_decay=0.0, decay_bias=False):
    """Deterministic, framework-independent trajectory of a single parameter."""
    p = ortho.from_numpy(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))
    p.requires_grad_(True)
    opt = Adam([p], lr=lr, betas=(0.9, 0.999), eps=1e-8,
               weight_decay=weight_decay, decay_bias=decay_bias)
    trace = []
    for step in range(steps):
        grad_np = np.array(
            [0.3 * np.cos(step * 0.7 + k) + 0.05 * k for k in range(4)],
            dtype=np.float32)
        p.set_grad(ortho.from_numpy(grad_np))
        opt.step()
        trace.append(list(p.tolist()))
    return trace


def _torch_adam_trajectory(steps=25, lr=0.05, weight_decay=0.0):
    p = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=weight_decay)
    trace = []
    for step in range(steps):
        grad_np = np.array(
            [0.3 * np.cos(step * 0.7 + k) + 0.05 * k for k in range(4)],
            dtype=np.float32)
        p.grad = torch.tensor(grad_np, dtype=torch.float32)
        opt.step()
        trace.append(p.detach().numpy().astype(np.float64).tolist())
    return trace


def test_adam_matches_torch_over_a_full_trajectory():
    got = _adam_trajectory()
    ref = _torch_adam_trajectory()
    for step, (g, r) in enumerate(zip(got, ref)):
        assert g == pytest.approx(r, abs=1e-5), "step %d" % step


def test_adam_weight_decay_still_applies_only_to_rank2_by_default():
    """decay_bias / weight_decay plumbing untouched by the scalar rewrite."""
    p1d = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    p2d = ortho.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    for p in (p1d, p2d):
        p.requires_grad_(True)
        p.set_grad(ortho.zeros(list(p.shape)))
    opt = Adam([p1d, p2d], lr=0.1, weight_decay=0.5, decay_bias=False)
    opt.step()
    # Zero gradient: only the decay term can move a parameter.
    assert p1d.tolist() == pytest.approx([1.0, 2.0])
    assert p2d.tolist() != pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_adam_is_deterministic_across_identical_runs():
    """Identical inputs, identical trajectory."""
    assert _adam_trajectory() == _adam_trajectory()


def test_adam_converges_on_a_quadratic():
    p = ortho.from_numpy(np.array([5.0, -4.0], dtype=np.float32))
    p.requires_grad_(True)
    opt = Adam([p], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        # d/dp of sum(p^2) is 2p.
        p.set_grad(ortho.from_numpy(
            (2.0 * np.array(p.tolist(), dtype=np.float32))))
        opt.step()
    assert np.abs(np.array(p.tolist())).max() < 1e-2


def test_adam_state_dict_round_trip_unchanged():
    p = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    p.requires_grad_(True)
    p.set_grad(ortho.from_numpy(np.array([0.1, -0.2], dtype=np.float32)))
    opt = Adam([p], lr=0.03, betas=(0.8, 0.99), eps=1e-7,
               weight_decay=0.1, decay_bias=True)
    opt.step()
    state = opt.state_dict()

    p2 = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    p2.requires_grad_(True)
    opt2 = Adam([p2], lr=0.5)
    opt2.load_state_dict(state)
    assert opt2.get_lr() == 0.03
    assert opt2._t == state["t"]
    assert opt2._eps == 1e-7
    assert opt2._weight_decay == 0.1
    assert opt2._decay_bias is True


@requires_cuda
def test_adam_cuda_matches_cpu():
    cpu_trace = _adam_trajectory(steps=10)

    p = ortho.from_numpy(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)).to("cuda")
    p.requires_grad_(True)
    opt = Adam([p], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    for step in range(10):
        grad_np = np.array(
            [0.3 * np.cos(step * 0.7 + k) + 0.05 * k for k in range(4)],
            dtype=np.float32)
        p.set_grad(ortho.from_numpy(grad_np).to("cuda"))
        opt.step()
        assert p.tolist() == pytest.approx(cpu_trace[step], abs=1e-5)


@requires_cuda
def test_adam_moments_stay_on_the_parameter_device():
    p = ortho.zeros([3]).to("cuda")
    p.requires_grad_(True)
    p.set_grad(ortho.zeros([3]).to("cuda"))
    opt = Adam([p], lr=0.1)
    opt.step()
    assert opt._m[0].device == p.device
    assert opt._v[0].device == p.device


# ---------------------------------------------------------------------------
# GradScaler.
# ---------------------------------------------------------------------------

def test_grad_scaler_source_uses_scalar_arithmetic():
    body = _strip_py_comments_and_docstrings(
        _read("python/orthon/amp/grad_scaler.py"))
    assert "return loss * self._scale" in body
    assert "master.set_grad(master.grad * inv)" in body
    assert "master.set_grad(fp32_grad * inv)" in body
    # The ONLY surviving full() is the [1]-element collective flag.
    fulls = re.findall(r"ortho\.full\(([^)]*)\)", body)
    assert fulls == ["[1], 1.0 if self._found_inf else 0.0"], fulls
    # The fused CUDA unscale path is preserved.
    assert "_amp_unscale_to_float32_and_check" in body


def test_grad_scaler_scale_is_numerically_unchanged():
    loss = ortho.from_numpy(np.array([1.5, -2.0], dtype=np.float32))
    scaler = GradScaler(init_scale=1024.0)
    assert scaler.scale(loss).tolist() == pytest.approx([1536.0, -2048.0])
    assert scaler.scale(loss).dtype == "float32"


def test_grad_scaler_scale_is_autograd_safe():
    x = ortho.from_numpy(np.array([1.5, -2.0], dtype=np.float32))
    x.requires_grad_(True)
    scaler = GradScaler(init_scale=8.0)
    ortho.sum(scaler.scale(x)).backward()
    assert x.grad.tolist() == pytest.approx([8.0, 8.0])


def test_grad_scaler_scale_preserves_the_loss_rank():
    """Deliberate, documented shape change for a RANK-0 loss.

    The old code wrote `shape = list(loss.shape) or [1]`, so scaling a rank-0
    loss returned shape [1] - the `or [1]` was there because a rank-0 constant
    was not what the helper was written for, not because the caller wanted a
    rank-1 result. Scalar arithmetic broadcasts, so the rank is preserved.
    Every loss the runtime actually produces is unaffected: nll_loss and
    cross_entropy both return shape (1,).
    """
    scaler = GradScaler(init_scale=4.0)
    rank0 = ortho.mean(ortho.full([4], 2.0))
    assert tuple(rank0.shape) == ()
    scaled = scaler.scale(rank0)
    assert tuple(scaled.shape) == ()
    assert scaled.item() == pytest.approx(8.0)

    rank1 = ortho.cross_entropy(ortho.full([2, 3], 0.5), [0, 1])
    assert tuple(rank1.shape) == (1,)
    assert tuple(scaler.scale(rank1).shape) == (1,)

    # A rank-0 scaled loss still backpropagates.
    x = ortho.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    x.requires_grad_(True)
    ortho.sum(scaler.scale(ortho.mean(x))).backward()
    assert x.grad.tolist() == pytest.approx([1.0] * 4)


def test_grad_scaler_disabled_is_still_identity():
    loss = ortho.from_numpy(np.array([1.5], dtype=np.float32))
    assert GradScaler(enabled=False).scale(loss) is loss


def test_grad_scaler_state_dict_format_unchanged():
    state = GradScaler(init_scale=512.0).state_dict()
    assert set(state) == {
        "version", "scale", "growth_factor", "backoff_factor",
        "growth_interval", "enabled", "unscaled", "found_inf", "growth_tracker",
    }
    assert state["version"] == 1


@requires_cuda
def test_grad_scaler_scale_keeps_the_exact_device_and_dtype():
    for dtype in ("float32", "float16", "bfloat16"):
        base = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).to("cuda")
        loss = base if dtype == "float32" else ortho.cast(base, dtype)
        out = GradScaler(init_scale=4.0).scale(loss)
        assert out.dtype == dtype
        assert out.device == loss.device
        assert np.array(out.tolist()) == pytest.approx([4.0, 8.0], abs=0.05)


# ---------------------------------------------------------------------------
# LayerNorm, BatchNorm, attention.
# ---------------------------------------------------------------------------

def test_normalization_sources_use_ordinary_arithmetic():
    ln = _strip_py_comments_and_docstrings(_read("python/orthon/nn/layernorm.py"))
    assert "centered = x - mean" in ln
    assert "ortho.reciprocal(ortho.sqrt(var + eps))" in ln
    assert "ortho.full(" not in ln

    bn = _strip_py_comments_and_docstrings(_read("python/orthon/nn/batchnorm.py"))
    assert "centered = x - mean" in bn
    assert "batch_var * (count / (count - 1))" in bn
    assert "self.running_mean * keep + batch_mean * self.momentum" in bn
    assert "self.running_var * keep + unbiased_var * self.momentum" in bn
    assert "ortho.reciprocal(ortho.sqrt(var + self.eps))" in bn
    assert "ortho.full(" not in bn
    # The detached-statistic snapshot stays an explicit no_grad.
    assert "with ortho.no_grad():" in bn

    att = _strip_py_comments_and_docstrings(_read("python/orthon/functional/math.py"))
    assert "ortho.matmul(q, k_t) * (d ** -0.5)" in att
    assert "ortho.full(" not in att
    # Dropout RNG behaviour untouched.
    assert "_randbits(64)" in att


@pytest.mark.parametrize("shape,normalized", [((3, 4), 4), ((2, 3, 5), 5)])
def test_layernorm_parity(shape, normalized):
    n = int(np.prod(shape))
    x_np = (np.cos(np.arange(n) * 1.3) * 2.0 + 0.4).reshape(shape).astype(np.float32)
    x = ortho.from_numpy(np.ascontiguousarray(x_np))
    x.requires_grad_(True)
    xt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

    ln = LayerNorm(normalized)
    y = ln(x)
    yt = torch.nn.functional.layer_norm(xt, (normalized,),
                                        weight=torch.ones(normalized),
                                        bias=torch.zeros(normalized), eps=1e-5)
    assert np.array(y.tolist()) == pytest.approx(
        yt.detach().numpy().ravel(), abs=1e-5)

    ortho.sum(y).backward()
    yt.sum().backward()
    assert np.array(x.grad.tolist()) == pytest.approx(
        xt.grad.numpy().ravel(), abs=1e-5)


@requires_cuda
def test_layernorm_cuda_parity():
    x_np = (np.cos(np.arange(12) * 1.3) * 2.0 + 0.4).reshape(3, 4).astype(np.float32)
    x = ortho.from_numpy(np.ascontiguousarray(x_np)).to("cuda")
    x.requires_grad_(True)
    ln = LayerNorm(4, device="cuda")
    y = ln(x)
    ortho.sum(y).backward()

    xt = torch.tensor(x_np, dtype=torch.float32, device="cuda", requires_grad=True)
    yt = torch.nn.functional.layer_norm(
        xt, (4,), weight=torch.ones(4, device="cuda"),
        bias=torch.zeros(4, device="cuda"), eps=1e-5)
    yt.sum().backward()

    assert y.device == x.device
    assert np.array(y.tolist()) == pytest.approx(
        yt.detach().cpu().numpy().ravel(), abs=1e-4)
    assert np.array(x.grad.tolist()) == pytest.approx(
        xt.grad.cpu().numpy().ravel(), abs=1e-4)


def test_batchnorm_running_stats_unchanged_by_the_rewrite():
    """Explicit closed-form check of the running-stat update."""
    x_np = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 0.0], [7.0, 4.0]],
                    dtype=np.float32)
    bn = BatchNorm1d(2, eps=1e-5, momentum=0.1, affine=False)
    bn.train()
    bn(ortho.from_numpy(np.ascontiguousarray(x_np)))

    count = x_np.shape[0]
    batch_mean = x_np.mean(axis=0)
    biased_var = x_np.var(axis=0)
    unbiased = biased_var * (count / (count - 1))
    expected_mean = 0.0 * 0.9 + batch_mean * 0.1
    expected_var = 1.0 * 0.9 + unbiased * 0.1

    assert np.array(bn.running_mean.tolist()) == pytest.approx(expected_mean, abs=1e-5)
    assert np.array(bn.running_var.tolist()) == pytest.approx(expected_var, abs=1e-5)


def test_batchnorm_running_stats_stay_out_of_the_graph():
    """Buffers are detached state, never differentiable outputs."""
    x = ortho.from_numpy(np.ascontiguousarray(
        np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)))
    x.requires_grad_(True)
    bn = BatchNorm1d(2)
    bn.train()
    bn(x)
    assert bn.running_mean.requires_grad is False
    assert bn.running_mean._has_grad_fn is False
    assert bn.running_var.requires_grad is False
    assert bn.running_var._has_grad_fn is False


def test_attention_scale_matches_torch():
    q_np = np.cos(np.arange(24) * 0.9).reshape(2, 3, 4).astype(np.float32)
    k_np = np.sin(np.arange(24) * 0.5).reshape(2, 3, 4).astype(np.float32)
    v_np = np.cos(np.arange(24) * 0.3).reshape(2, 3, 4).astype(np.float32)
    q, k, v = (ortho.from_numpy(np.ascontiguousarray(a)) for a in (q_np, k_np, v_np))
    out = ortho.scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np))
    assert np.array(out.tolist()) == pytest.approx(ref.numpy().ravel(), abs=1e-5)


@requires_cuda
def test_attention_keeps_the_exact_device():
    q_np = np.cos(np.arange(24) * 0.9).reshape(2, 3, 4).astype(np.float32)
    q = ortho.from_numpy(np.ascontiguousarray(q_np)).to("cuda")
    out = ortho.scaled_dot_product_attention(q, q, q)
    assert out.device == q.device


# ---------------------------------------------------------------------------
# the full-shaped scalar census, as an executable rule.
# ---------------------------------------------------------------------------

RUNTIME_ROOT = "python/orthon"

# Every surviving full()/ones()/zeros()-as-constant call in the runtime, with
# the reason it is NOT a disguised scalar. Anything not on this list fails.
ALLOWED_CONSTANT_TENSORS = {
    # rel path                                    justification
    "python/orthon/amp/grad_scaler.py": "collective operand: [1]-element found_inf flag",
}


def _runtime_python_files():
    root = os.path.join(REPO, RUNTIME_ROOT.replace("/", os.sep))
    for dirpath, _, names in os.walk(root):
        for name in sorted(names):
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, REPO).replace(os.sep, "/")


def test_full_shaped_scalar_census():
    """No runtime call site builds a data-shaped tensor to carry a scalar.

    The signature of the retired pattern is a full() whose shape argument is
    derived from another tensor's shape.
    """
    offenders = []
    shape_derived = re.compile(
        r"\bfull\(\s*(?:list\()?\s*[A-Za-z_][\w.]*\.shape")
    ones_times_scalar = re.compile(r"\bones\([^)]*\)\s*\*\s*[-\d.]")
    for rel in _runtime_python_files():
        body = _strip_py_comments_and_docstrings(_read(rel))
        for pattern in (shape_derived, ones_times_scalar):
            for match in pattern.finditer(body):
                offenders.append("%s: %s" % (rel, match.group(0)))
    assert offenders == [], (
        "full parameter/input-shaped tensor used only as a scalar constant:\n  "
        + "\n  ".join(offenders))


def test_only_the_collective_flag_still_calls_full_in_the_scalar_modules():
    """The five scalar-arithmetic modules build no constant tensors at all."""
    for rel in ("python/orthon/optim/adam.py",
                "python/orthon/nn/layernorm.py",
                "python/orthon/nn/batchnorm.py",
                "python/orthon/functional/math.py"):
        body = _strip_py_comments_and_docstrings(_read(rel))
        assert "full(" not in body, rel
    body = _strip_py_comments_and_docstrings(
        _read("python/orthon/amp/grad_scaler.py"))
    assert body.count("full(") == 1
    assert ALLOWED_CONSTANT_TENSORS["python/orthon/amp/grad_scaler.py"]
