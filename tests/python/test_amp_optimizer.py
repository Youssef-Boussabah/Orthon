"""Tests for MixedPrecisionOptimizer."""
import pytest
import numpy as np

import orthon as ortho
from orthon.nn import Linear
from orthon.nn.module import Module
from orthon.optim import SGD, Adam
from orthon.amp import MixedPrecisionOptimizer

CUDA = ortho.cuda_enabled()

# Tiny model fixture.

class TinyNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 4)
        self.fc2 = Linear(4, 2)

    def forward(self, x):
        return self.fc2(ortho.relu(self.fc1(x)))


def _param_values_f32(model):
    """Collect all param values as FP32 numpy arrays for comparison."""
    out = {}
    for name, p in model.named_parameters():
        cpu = p.to("cpu") if ortho.cuda_enabled() else p
        f32 = ortho.cast(cpu, "float32")
        out[name] = f32.tolist()
    return out


# CPU and FP32 infrastructure tests.

class TestMixedPrecisionOptimizerCpuFp32:
    """CPU FP32 model where the master is the model param."""

    def test_creates_without_error(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        assert opt is not None

    def test_fp32_master_is_same_object(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        # For FP32 params, master should be the same tensor (no copy)
        for sub_mod, name, orig_dtype, master in opt._entries:
            if orig_dtype == "float32":
                assert master is sub_mod._parameters[name]

    def test_zero_grad_clears_model_param_grads(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        out = ortho.sum(model(x))
        out.backward()
        # At least some params should have grads now
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        opt.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_step_changes_params(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        before = _param_values_f32(model)

        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        out = ortho.sum(model(x))
        out.backward()
        opt.step()

        after = _param_values_f32(model)
        changed = any(
            any(a != b for a, b in zip(_flatten(after[k]), _flatten(before[k])))
            for k in before
        )
        assert changed, "No parameters changed after optimizer step"

    def test_fp32_params_remain_fp32_after_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for name, p in model.named_parameters():
            assert p.dtype == "float32", f"{name} is {p.dtype}"

    def test_requires_grad_preserved_after_step(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for p in model.parameters():
            assert p.requires_grad

    def test_multiple_steps(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        for _ in range(3):
            opt.zero_grad()
            ortho.sum(model(x)).backward()
            opt.step()
        # Just verify it doesn't crash and params are still FP32
        for name, p in model.named_parameters():
            assert p.dtype == "float32"

    def test_with_adam_optimizer(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, Adam, lr=1e-3)
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32))
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for p in model.parameters():
            assert p.requires_grad

    def test_get_lr_set_lr(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        assert abs(opt.get_lr() - 0.1) < 1e-9
        opt.set_lr(0.01)
        assert abs(opt.get_lr() - 0.01) < 1e-9

    def test_sync_to_model_is_noop_for_fp32(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        before_ids = {id(p) for p in model.parameters()}
        opt.sync_to_model()
        after_ids = {id(p) for p in model.parameters()}
        # FP32 params: sync_to_model does nothing, same tensor objects
        assert before_ids == after_ids

    def test_state_dict_roundtrip_restores_inner_optimizer_config(self):
        model = TinyNet()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.05, weight_decay=0.1, decay_bias=True)
        state = opt.state_dict()

        restored = MixedPrecisionOptimizer(TinyNet(), SGD, lr=1.0)
        restored.load_state_dict(state)

        assert restored.get_lr() == pytest.approx(0.05)
        assert restored._inner._weight_decay == pytest.approx(0.1)
        assert restored._inner._decay_bias is True


# CUDA low-precision tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestMixedPrecisionOptimizerCudaFp16:
    def _make_opt(self, dtype_method="half"):
        model = TinyNet().to("cuda")
        getattr(model, dtype_method)()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.1)
        return model, opt

    def _input(self, dtype):
        x = ortho.from_numpy(np.ones((2, 4), dtype=np.float32)).to("cuda")
        return ortho.cast(x, dtype)

    def test_fp16_masters_are_fp32(self):
        model, opt = self._make_opt("half")
        for sub_mod, name, orig_dtype, master in opt._entries:
            if orig_dtype == "float16":
                assert master.dtype == "float32", f"Master for {name} is {master.dtype}"

    def test_bf16_masters_are_fp32(self):
        model, opt = self._make_opt("bfloat16")
        for sub_mod, name, orig_dtype, master in opt._entries:
            if orig_dtype == "bfloat16":
                assert master.dtype == "float32", f"Master for {name} is {master.dtype}"

    def test_step_fp16_model_params_remain_fp16(self):
        model, opt = self._make_opt("half")
        x = self._input("float16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for name, p in model.named_parameters():
            assert p.dtype == "float16", f"{name} is {p.dtype}"

    def test_step_bf16_model_params_remain_bfloat16(self):
        model, opt = self._make_opt("bfloat16")
        x = self._input("bfloat16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for name, p in model.named_parameters():
            assert p.dtype == "bfloat16", f"{name} is {p.dtype}"

    def test_step_changes_fp16_params(self):
        model, opt = self._make_opt("half")
        before = _param_values_f32(model)
        x = self._input("float16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        after = _param_values_f32(model)
        changed = any(
            any(a != b for a, b in zip(_flatten(after[k]), _flatten(before[k])))
            for k in before
        )
        assert changed, "No FP16 parameters changed after optimizer step"

    def test_step_changes_bf16_params(self):
        model, opt = self._make_opt("bfloat16")
        before = _param_values_f32(model)
        x = self._input("bfloat16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        after = _param_values_f32(model)
        changed = any(
            any(a != b for a, b in zip(_flatten(after[k]), _flatten(before[k])))
            for k in before
        )
        assert changed, "No BF16 parameters changed after optimizer step"

    def test_requires_grad_preserved_after_fp16_step(self):
        model, opt = self._make_opt("half")
        x = self._input("float16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for p in model.parameters():
            assert p.requires_grad

    def test_multiple_steps_fp16(self):
        model, opt = self._make_opt("half")
        x = self._input("float16")
        for _ in range(3):
            opt.zero_grad()
            ortho.sum(model(x)).backward()
            opt.step()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_zero_grad_clears_fp16_model_param_grads(self):
        model, opt = self._make_opt("half")
        x = self._input("float16")
        ortho.sum(model(x)).backward()
        opt.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_with_adam_fp16(self):
        model, opt = self._make_opt("half")
        opt = MixedPrecisionOptimizer(model, Adam, lr=1e-3)
        x = self._input("float16")
        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_sync_to_model_restores_fp16(self):
        model, opt = self._make_opt("half")
        # Manually update a master and sync
        _, _, _, master = opt._entries[0]
        with ortho.no_grad():
            master.add_(master, alpha=-0.001)   # small in-place tweak on FP32 master
        opt.sync_to_model()
        for name, p in model.named_parameters():
            assert p.dtype == "float16"

    def test_sync_to_model_preserves_fp16_parameter_objects(self):
        model, opt = self._make_opt("half")
        before_ids = {name: p._impl_id for name, p in model.named_parameters()}

        opt.sync_to_model()

        after_ids = {name: p._impl_id for name, p in model.named_parameters()}
        assert after_ids == before_ids

    def test_step_preserves_fp16_parameter_objects(self):
        model, opt = self._make_opt("half")
        before_ids = {name: p._impl_id for name, p in model.named_parameters()}
        x = self._input("float16")

        opt.zero_grad()
        ortho.sum(model(x)).backward()
        opt.step()

        after_ids = {name: p._impl_id for name, p in model.named_parameters()}
        assert after_ids == before_ids

    def test_state_dict_roundtrip_restores_fp32_masters(self):
        model, opt = self._make_opt("half")
        for _, _, orig_dtype, master in opt._entries:
            if orig_dtype == "float16":
                with ortho.no_grad():
                    master.mul_(0.0)
                    master.add_(ortho.full(list(master.shape), 0.75).to(master.device))
        state = opt.state_dict()

        restored_model, restored_opt = self._make_opt("half")
        restored_opt.load_state_dict(state)

        for _, _, orig_dtype, master in restored_opt._entries:
            if orig_dtype == "float16":
                assert master.dtype == "float32"
                assert _flatten(master.to("cpu").tolist()) == pytest.approx(
                    [0.75] * int(master.numel)
                )
        for name, param in restored_model.named_parameters():
            assert param.dtype == "float16", name


# Helper.

def _flatten(x):
    if isinstance(x, (int, float)):
        return [x]
    out = []
    for v in x:
        out.extend(_flatten(v))
    return out
