"""End-to-end AMP smoke tests: FP16 + BF16 training loops on CUDA."""
import pytest
import numpy as np

import orthon as ortho
from orthon.nn import Linear
from orthon.nn.module import Module
from orthon.optim import Adam, SGD
from orthon.amp import MixedPrecisionOptimizer, GradScaler

CUDA = ortho.cuda_enabled()


# Tiny model fixture.

class SmallMLP(Module):
    """Three-layer MLP: wide enough that gradients flow through all params."""
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 8)
        self.fc2 = Linear(8, 4)
        self.fc3 = Linear(4, 2)

    def forward(self, x):
        return self.fc3(ortho.relu(self.fc2(ortho.relu(self.fc1(x)))))


# Helpers.

def _make_input(dtype):
    """Batch of 4 × 8 random inputs, on CUDA, cast to dtype."""
    x = ortho.from_numpy(np.random.randn(4, 8).astype(np.float32)).to("cuda")
    return ortho.cast(x, dtype)


def _sq_loss(out):
    """Sum-of-squares loss. Always ≥ 0; gradient pushes outputs toward 0."""
    return ortho.sum(ortho.mul(out, out))


def _scalar(t):
    """Pull a scalar CUDA tensor to a Python float."""
    cpu = t.to("cpu")
    vals = _flatten(ortho.cast(cpu, "float32").tolist())
    return vals[0]


def _param_snapshot(model):
    """Flat list of all parameter values as float32 Python floats."""
    out = []
    for p in model.parameters():
        out.extend(_flatten(ortho.cast(p.to("cpu"), "float32").tolist()))
    return out


def _flatten(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    result = []
    for v in x:
        result.extend(_flatten(v))
    return result


def _params_changed(before, after):
    return any(a != b for a, b in zip(before, after))


# FP16 smoke tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestAmpSmokeFp16:
    DTYPE = "float16"

    def _build(self, lr=0.01, scale=256.0):
        np.random.seed(0)
        model = SmallMLP().to("cuda").half()
        opt = MixedPrecisionOptimizer(model, Adam, lr=lr)
        scaler = GradScaler(init_scale=scale)
        return model, opt, scaler

    def _x(self):
        return _make_input(self.DTYPE)

    # Single step.

    def test_params_update_after_one_step(self):
        model, opt, scaler = self._build()
        x = self._x()
        before = _param_snapshot(model)
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        assert _params_changed(before, _param_snapshot(model))

    def test_dtype_fp16_preserved_after_step(self):
        model, opt, scaler = self._build()
        x = self._x()
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE, f"{name}: expected {self.DTYPE}, got {p.dtype}"

    def test_requires_grad_preserved_after_step(self):
        model, opt, scaler = self._build()
        x = self._x()
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for p in model.parameters():
            assert p.requires_grad

    def test_params_on_cuda_after_step(self):
        model, opt, scaler = self._build()
        x = self._x()
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for p in model.parameters():
            assert "cuda" in str(p.device)

    # Multi-step.

    def test_loss_decreases_over_steps(self):
        model, opt, scaler = self._build(lr=0.05)
        x = self._x()
        losses = []
        for _ in range(10):
            opt.zero_grad()
            out = model(x)
            loss = _sq_loss(out)
            losses.append(_scalar(loss))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        assert losses[-1] < losses[0], \
            f"FP16 loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_five_iterations_no_crash(self):
        model, opt, scaler = self._build()
        x = self._x()
        for _ in range(5):
            opt.zero_grad()
            scaler.scale(_sq_loss(model(x))).backward()
            scaler.step(opt)
            scaler.update()
        # Sanity: all params still fp16 on cuda
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE
            assert "cuda" in str(p.device)

    def test_explicit_unscale_then_step(self):
        """scale → backward → unscale_ → step → update preserves dtype."""
        model, opt, scaler = self._build()
        x = self._x()
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.unscale_(opt)          # explicit unscale
        scaler.step(opt)              # must not double-unscale
        scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE

    def test_sgd_fp16_smoke(self):
        np.random.seed(0)
        model = SmallMLP().to("cuda").half()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=256.0)
        x = self._x()
        before = _param_snapshot(model)
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        assert _params_changed(before, _param_snapshot(model))
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE


# BF16 smoke tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestAmpSmokeBf16:
    DTYPE = "bfloat16"

    def _build_no_scaler(self, lr=0.01):
        np.random.seed(0)
        model = SmallMLP().to("cuda").bfloat16()
        opt = MixedPrecisionOptimizer(model, Adam, lr=lr)
        scaler = GradScaler(enabled=False)     # BF16 has FP32 exponent range
        return model, opt, scaler

    def _build_scale1(self, lr=0.01):
        np.random.seed(0)
        model = SmallMLP().to("cuda").bfloat16()
        opt = MixedPrecisionOptimizer(model, Adam, lr=lr)
        scaler = GradScaler(init_scale=1.0, enabled=True)
        return model, opt, scaler

    def _x(self):
        return _make_input(self.DTYPE)

    # Disabled scaler, the typical BF16 path.

    def test_params_update_no_scaler(self):
        model, opt, scaler = self._build_no_scaler()
        x = self._x()
        before = _param_snapshot(model)
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        scaler.step(opt)
        scaler.update()
        assert _params_changed(before, _param_snapshot(model))

    def test_dtype_bf16_preserved_no_scaler(self):
        model, opt, scaler = self._build_no_scaler()
        x = self._x()
        for _ in range(3):
            opt.zero_grad()
            ortho.sum(model(x)).backward()
            scaler.step(opt)
            scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE, f"{name}: expected {self.DTYPE}, got {p.dtype}"

    def test_requires_grad_preserved_no_scaler(self):
        model, opt, scaler = self._build_no_scaler()
        x = self._x()
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        scaler.step(opt)
        scaler.update()
        for p in model.parameters():
            assert p.requires_grad

    def test_loss_decreases_no_scaler(self):
        model, opt, scaler = self._build_no_scaler(lr=0.05)
        x = self._x()
        losses = []
        for _ in range(10):
            opt.zero_grad()
            out = model(x)
            loss = _sq_loss(out)
            losses.append(_scalar(loss))
            loss.backward()
            scaler.step(opt)
            scaler.update()
        assert losses[-1] < losses[0], \
            f"BF16 loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    # scale=1.0 path.

    def test_bf16_scale_1_params_update(self):
        model, opt, scaler = self._build_scale1()
        x = self._x()
        before = _param_snapshot(model)
        opt.zero_grad()
        scaler.scale(_sq_loss(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        assert _params_changed(before, _param_snapshot(model))

    def test_bf16_scale_1_dtype_preserved(self):
        model, opt, scaler = self._build_scale1()
        x = self._x()
        for _ in range(3):
            opt.zero_grad()
            scaler.scale(_sq_loss(model(x))).backward()
            scaler.step(opt)
            scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE, f"{name}: expected {self.DTYPE}, got {p.dtype}"

    def test_bf16_five_iterations_no_crash(self):
        model, opt, scaler = self._build_no_scaler()
        x = self._x()
        for _ in range(5):
            opt.zero_grad()
            ortho.sum(model(x)).backward()
            scaler.step(opt)
            scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == self.DTYPE
            assert "cuda" in str(p.device)
