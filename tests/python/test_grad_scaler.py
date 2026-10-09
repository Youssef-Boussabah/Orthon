"""Tests for GradScaler (static loss scaling)."""
from tests.support.capabilities import orthon_cuda_available
import pytest
import numpy as np

import orthon as ortho
from orthon.nn import Linear
from orthon.nn.module import Module
from orthon.optim import SGD, Adam
from orthon.amp import MixedPrecisionOptimizer, GradScaler

CUDA = orthon_cuda_available()


# Tiny model fixture.

class TinyNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 4)
        self.fc2 = Linear(4, 2)

    def forward(self, x):
        return self.fc2(ortho.relu(self.fc1(x)))


def _flatten(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(_flatten(v))
    return out


def _param_values_f32(model):
    out = {}
    for name, p in model.named_parameters():
        cpu = p.to("cpu") if CUDA else p
        out[name] = _flatten(ortho.cast(cpu, "float32").tolist())
    return out


# Basic property tests that do not need autograd.

class TestGradScalerProperties:
    def test_default_scale(self):
        s = GradScaler()
        assert s.scale_factor == 1024.0

    def test_custom_scale(self):
        s = GradScaler(init_scale=256.0)
        assert s.scale_factor == 256.0

    def test_enabled_by_default(self):
        assert GradScaler().enabled is True

    def test_disabled(self):
        assert GradScaler(enabled=False).enabled is False

    def test_update_is_noop(self):
        s = GradScaler(init_scale=512.0)
        s.update()
        assert s.scale_factor == 512.0

    def test_update_resets_unscaled_flag(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.0)
        s = GradScaler(init_scale=8.0)
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.unscale_(opt)
        assert s._unscaled is True
        s.update()
        assert s._unscaled is False

    def test_scale_disabled_returns_same_tensor(self):
        s = GradScaler(enabled=False)
        model = TinyNet()
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        loss = ortho.sum(model(x))
        scaled = s.scale(loss)
        assert scaled is loss

    def test_state_dict_roundtrip_preserves_dynamic_state(self):
        s = GradScaler(
            init_scale=32.0,
            growth_factor=3.0,
            backoff_factor=0.25,
            growth_interval=4,
            enabled=True,
        )
        s._growth_tracker = 2
        s._found_inf = True
        restored = GradScaler()

        restored.load_state_dict(s.state_dict())

        assert restored.scale_factor == pytest.approx(32.0)
        assert restored._growth_factor == pytest.approx(3.0)
        assert restored._backoff_factor == pytest.approx(0.25)
        assert restored._growth_interval == 4
        assert restored.enabled is True
        assert restored.found_inf is True
        assert restored._growth_tracker == 2


# CPU scale() tests.

class TestGradScalerDynamic:
    def _first_param(self, model):
        return next(model.parameters())

    def _set_first_grad(self, model, value):
        p = self._first_param(model)
        p.set_grad(ortho.full(list(p.shape), value))
        return p

    def test_finite_gradients_allow_optimizer_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=2.0)
        p = self._set_first_grad(model, 1.0)
        before = p.tolist()[:]

        scaler.step(opt)
        scaler.update()

        assert p.tolist() != pytest.approx(before)
        assert scaler.scale_factor == pytest.approx(2.0)

    def test_inf_gradients_skip_optimizer_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=2.0)
        p = self._set_first_grad(model, float("inf"))
        before = p.tolist()[:]

        scaler.step(opt)
        scaler.update()

        assert p.tolist() == pytest.approx(before)

    def test_nan_gradients_skip_optimizer_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=2.0)
        p = self._set_first_grad(model, float("nan"))
        before = p.tolist()[:]

        scaler.step(opt)
        scaler.update()

        assert p.tolist() == pytest.approx(before)

    def test_scale_backs_off_after_nonfinite_gradients(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=8.0, backoff_factor=0.25)
        self._set_first_grad(model, float("inf"))

        scaler.step(opt)
        scaler.update()

        assert scaler.scale_factor == pytest.approx(2.0)

    def test_scale_grows_after_growth_interval_successes(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=2.0, growth_factor=4.0, growth_interval=2)

        self._set_first_grad(model, 1.0)
        scaler.step(opt)
        scaler.update()
        assert scaler.scale_factor == pytest.approx(2.0)

        opt.zero_grad()
        self._set_first_grad(model, 1.0)
        scaler.step(opt)
        scaler.update()
        assert scaler.scale_factor == pytest.approx(8.0)

    def test_disabled_scaler_path_behaves_like_normal_training(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=128.0, enabled=False)
        p = self._set_first_grad(model, 1.0)
        before = p.tolist()[:]

        scaler.step(opt)
        scaler.update()

        assert p.tolist() != pytest.approx(before)
        assert scaler.scale_factor == pytest.approx(128.0)


class TestGradScalerScale:
    def test_scale_multiplies_loss_value(self):
        SCALE = 16.0
        s = GradScaler(init_scale=SCALE)
        model = TinyNet()
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        loss = ortho.sum(model(x))
        scaled = s.scale(loss)
        loss_val = _flatten(loss.tolist())[0]
        scaled_val = _flatten(scaled.tolist())[0]
        assert abs(scaled_val - loss_val * SCALE) < abs(loss_val * SCALE) * 1e-5 + 1e-6

    def test_scale_backward_produces_grads(self):
        s = GradScaler(init_scale=8.0)
        model = TinyNet()
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        s.scale(ortho.sum(model(x))).backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_scale_grads_are_proportional(self):
        """Grads with scale=S should be S× larger than without scaling."""
        SCALE = 4.0
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        model = TinyNet()

        # Reference: unscaled backward on the same model
        ortho.sum(model(x)).backward()
        ref_grads = {id(p): _flatten(p.grad.tolist())
                     for p in model.parameters() if p.grad is not None}

        # Zero and re-run with scaling on the same model / same weights
        for p in model.parameters():
            if p.grad is not None:
                p.zero_grad()
        GradScaler(init_scale=SCALE).scale(ortho.sum(model(x))).backward()

        for p in model.parameters():
            pid = id(p)
            if p.grad is not None and pid in ref_grads:
                sg = _flatten(p.grad.tolist())
                rg = ref_grads[pid]
                for sv, rv in zip(sg, rg):
                    if abs(rv) > 1e-8:
                        assert abs(sv / rv - SCALE) < 0.01 * SCALE


# CPU unscale_() tests with FP32 params.

class TestGradScalerUnscale:
    def test_unscale_sets_master_grads_nonnone(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.0)
        s = GradScaler(init_scale=8.0)
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.unscale_(opt)
        assert any(p.grad is not None for p in model.parameters())
        for p, master in zip(model.parameters(), opt.state_dict()["masters"]):
            assert (master.grad is None) == (p.grad is None)

    def test_unscale_divides_grad_by_scale(self):
        """Master grad after unscale_ should match direct (unscaled) backward."""
        SCALE = 8.0
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))

        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.0)

        # Reference: direct backward without scaling. Use entry index as key to
        # avoid name collisions ("weight" appears in both fc1 and fc2).
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        ref = {}
        for idx, master in enumerate(opt.state_dict()["masters"]):
            if master.grad is not None:
                ref[idx] = _flatten(ortho.cast(master.grad, "float32").tolist())

        # Scaled backward then unscale_
        opt.zero_grad()
        GradScaler(init_scale=SCALE).scale(ortho.sum(model(x))).backward()
        scaler2 = GradScaler(init_scale=SCALE)
        scaler2.unscale_(opt)

        for idx, master in enumerate(opt.state_dict()["masters"]):
            if master.grad is not None and idx in ref:
                got = _flatten(ortho.cast(master.grad, "float32").tolist())
                exp = ref[idx]
                for g, e in zip(got, exp):
                    assert abs(g - e) < abs(e) * 0.01 + 1e-6, \
                        f"parameter {idx}: got {g:.6f}, expected {e:.6f}"

    def test_unscale_idempotent(self):
        """Calling unscale_ twice should not double-divide."""
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.0)
        s = GradScaler(init_scale=4.0)
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.unscale_(opt)
        # Capture grad values after first unscale
        grad1 = {name: _flatten(ortho.cast(master.grad, "float32").tolist())
                 for name, master in enumerate(opt.state_dict()["masters"]) if master.grad is not None}
        s.unscale_(opt)  # second call must be ignored
        grad2 = {name: _flatten(ortho.cast(master.grad, "float32").tolist())
                 for name, master in enumerate(opt.state_dict()["masters"]) if master.grad is not None}
        for name in grad1:
            for a, b in zip(grad1[name], grad2[name]):
                assert abs(a - b) < 1e-9

    def test_unscale_disabled_noop(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.0)
        s = GradScaler(enabled=False)
        x = ortho.from_numpy(np.ones((1, 4), dtype=np.float32))
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        s.unscale_(opt)  # should not modify grads or crash


# Full CPU FP32 cycle.

class TestGradScalerFullCycleCpu:
    def test_step_changes_params(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        s = GradScaler(init_scale=8.0)
        before = _param_values_f32(model)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.step(opt)
        s.update()
        after = _param_values_f32(model)
        changed = any(a != b for k in before for a, b in zip(after[k], before[k]))
        assert changed

    def test_step_disabled_delegates_to_mp_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        s = GradScaler(enabled=False)
        before = _param_values_f32(model)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        s.step(opt)
        s.update()
        after = _param_values_f32(model)
        changed = any(a != b for k in before for a, b in zip(after[k], before[k]))
        assert changed

    def test_explicit_unscale_then_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        s = GradScaler(init_scale=16.0)
        before = _param_values_f32(model)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.unscale_(opt)        # explicit unscale
        s.step(opt)            # should not double-unscale
        s.update()
        after = _param_values_f32(model)
        changed = any(a != b for k in before for a, b in zip(after[k], before[k]))
        assert changed

    def test_fp32_params_stay_fp32_after_cycle(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        s = GradScaler(init_scale=8.0)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        s.scale(ortho.sum(model(x))).backward()
        s.step(opt)
        s.update()
        for name, p in model.named_parameters():
            assert p.dtype == "float32", f"{name} is {p.dtype}"

    def test_multiple_iterations(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, Adam, lr=1e-3)
        s = GradScaler(init_scale=8.0)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        for _ in range(3):
            opt.zero_grad()
            s.scale(ortho.sum(model(x))).backward()
            s.step(opt)
            s.update()
        for p in model.parameters():
            assert p.requires_grad


# CUDA FP16 tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestGradScalerCudaFp16:
    def _setup(self, dtype_method="half", scale=1024.0):
        model = TinyNet().to("cuda")
        getattr(model, dtype_method)()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        scaler = GradScaler(init_scale=scale)
        return model, opt, scaler

    def _input(self, dtype):
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32)).to("cuda")
        return ortho.cast(x, dtype)

    def test_fp16_full_cycle_no_crash(self):
        model, opt, scaler = self._setup("half")
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.step(opt)
        scaler.update()

    def test_fp16_params_remain_fp16_after_cycle(self):
        model, opt, scaler = self._setup("half")
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == "float16", f"{name} is {p.dtype}"

    def test_fp16_params_change_after_step(self):
        model, opt, scaler = self._setup("half")
        before = _param_values_f32(model)
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        after = _param_values_f32(model)
        changed = any(a != b for k in before for a, b in zip(after[k], before[k]))
        assert changed

    def test_fp16_requires_grad_preserved(self):
        model, opt, scaler = self._setup("half")
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for p in model.parameters():
            assert p.requires_grad

    def test_fp16_unscale_sets_fp32_master_grads(self):
        model, opt, scaler = self._setup("half", scale=32.0)
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.unscale_(opt)
        for orig_dtype, master in zip(opt.state_dict()["orig_dtypes"], opt.state_dict()["masters"]):
            if orig_dtype == "float16" and master.grad is not None:
                assert master.grad.dtype == "float32"

    def test_bf16_enabled_false(self):
        """BF16 model with disabled scaler still trains correctly."""
        model, opt, scaler = self._setup("bfloat16", scale=1.0)
        scaler = GradScaler(enabled=False)
        x = self._input("bfloat16")
        before = _param_values_f32(model)
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        scaler.step(opt)
        scaler.update()
        after = _param_values_f32(model)
        for name, p in model.named_parameters():
            assert p.dtype == "bfloat16"
        changed = any(a != b for k in before for a, b in zip(after[k], before[k]))
        assert changed

    def test_multiple_fp16_iterations(self):
        model, opt, scaler = self._setup("half")
        x = self._input("float16")
        for _ in range(3):
            opt.zero_grad()
            scaler.scale(ortho.sum(model(x))).backward()
            scaler.step(opt)
            scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_explicit_unscale_then_step_fp16(self):
        model, opt, scaler = self._setup("half")
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.unscale_(opt)
        scaler.step(opt)   # must not double-unscale
        scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_with_adam_fp16(self):
        model = TinyNet().to("cuda").half()
        opt = MixedPrecisionOptimizer(model, Adam, lr=1e-3)
        scaler = GradScaler(init_scale=256.0)
        x = self._input("float16")
        opt.zero_grad()
        scaler.scale(ortho.sum(model(x))).backward()
        scaler.step(opt)
        scaler.update()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_fp16_nonfinite_grad_skips_step_and_backs_off_scale(self):
        model, opt, scaler = self._setup("half", scale=8.0)
        scaler = GradScaler(init_scale=8.0, backoff_factor=0.25)
        master = opt.state_dict()["masters"][0]
        p = next(model.parameters())
        orig_dtype = p.dtype
        before = _param_values_f32(model)
        grad = ortho.cast(ortho.full(list(p.shape), float("inf")).to("cuda"), "float16")
        p.set_grad(grad)

        scaler.step(opt)
        scaler.update()

        assert scaler.scale_factor == pytest.approx(2.0)
        assert _param_values_f32(model) == before
        assert orig_dtype == "float16"
        assert master.grad is not None

    def test_fp16_finite_grad_steps_and_grows_scale(self):
        model, opt, scaler = self._setup("half", scale=2.0)
        scaler = GradScaler(init_scale=2.0, growth_factor=4.0, growth_interval=2)
        before = _param_values_f32(model)

        for expected_scale in (2.0, 8.0):
            opt.zero_grad()
            p = next(model.parameters())
            grad = ortho.cast(ortho.full(list(p.shape), 1.0).to("cuda"), "float16")
            p.set_grad(grad)
            scaler.step(opt)
            scaler.update()
            assert scaler.scale_factor == pytest.approx(expected_scale)

        assert _param_values_f32(model) != before
