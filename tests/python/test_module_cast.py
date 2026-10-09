"""Tests for Module.cast(), Module.half(), Module.bfloat16()."""
from tests.support.capabilities import orthon_cuda_available
import pytest

import orthon as ortho
from orthon.nn import Linear, Sequential
from orthon.nn.module import Module

CUDA = orthon_cuda_available()

# Small nested model fixture.

class TwoLayer(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class NestedModel(Module):
    def __init__(self):
        super().__init__()
        self.encoder = TwoLayer()
        self.head = Linear(2, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


# Helpers.

def all_param_dtypes(model):
    return {name: p.dtype for name, p in model.named_parameters()}


# CPU cast tests.

class TestModuleCastCpu:
    def test_cast_to_float16_all_params(self):
        m = TwoLayer()
        m.cast("float16")
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "float16", f"{name} is {dtype}"

    def test_cast_to_bfloat16_all_params(self):
        m = TwoLayer()
        m.cast("bfloat16")
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "bfloat16", f"{name} is {dtype}"

    def test_half_is_float16(self):
        m = TwoLayer()
        m.half()
        for dtype in all_param_dtypes(m).values():
            assert dtype == "float16"

    def test_bfloat16_method(self):
        m = TwoLayer()
        m.bfloat16()
        for dtype in all_param_dtypes(m).values():
            assert dtype == "bfloat16"

    def test_cast_returns_self(self):
        m = TwoLayer()
        assert m.cast("float16") is m

    def test_half_returns_self(self):
        m = TwoLayer()
        assert m.half() is m

    def test_bfloat16_returns_self(self):
        m = TwoLayer()
        assert m.bfloat16() is m

    def test_requires_grad_preserved_after_cast(self):
        m = TwoLayer()
        m.cast("float16")
        for p in m.parameters():
            assert p.requires_grad

    def test_requires_grad_preserved_after_half(self):
        m = TwoLayer()
        m.half()
        for p in m.parameters():
            assert p.requires_grad

    def test_requires_grad_preserved_after_bfloat16(self):
        m = TwoLayer()
        m.bfloat16()
        for p in m.parameters():
            assert p.requires_grad

    def test_nested_model_cast_recurses(self):
        m = NestedModel()
        m.cast("float16")
        dtypes = all_param_dtypes(m)
        assert len(dtypes) > 0
        for name, dtype in dtypes.items():
            assert dtype == "float16", f"{name} is {dtype}"

    def test_nested_model_bfloat16_recurses(self):
        m = NestedModel()
        m.bfloat16()
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "bfloat16", f"{name} is {dtype}"

    def test_cast_then_cast_back(self):
        m = TwoLayer()
        m.half().cast("float32")
        for dtype in all_param_dtypes(m).values():
            assert dtype == "float32"

    def test_sequential_cast(self):
        m = Sequential(Linear(4, 4), Linear(4, 2))
        m.half()
        for dtype in all_param_dtypes(m).values():
            assert dtype == "float16"

    def test_to_device_still_works_after_cast(self):
        # Ensures cast() doesn't break the to() path
        m = TwoLayer()
        m.cast("float32")
        # Should not raise
        m.to("cpu")
        for p in m.parameters():
            assert p.requires_grad

    def test_param_shapes_unchanged_after_cast(self):
        m = TwoLayer()
        shapes_before = {n: tuple(p.shape) for n, p in m.named_parameters()}
        m.half()
        shapes_after = {n: tuple(p.shape) for n, p in m.named_parameters()}
        assert shapes_before == shapes_after


# CUDA cast tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestModuleCastCuda:
    def test_cast_to_float16_on_cuda(self):
        m = TwoLayer().to("cuda")
        m.cast("float16")
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "float16", f"{name} is {dtype}"

    def test_cast_to_bfloat16_on_cuda(self):
        m = TwoLayer().to("cuda")
        m.cast("bfloat16")
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "bfloat16", f"{name} is {dtype}"

    def test_half_on_cuda(self):
        m = TwoLayer().to("cuda").half()
        for dtype in all_param_dtypes(m).values():
            assert dtype == "float16"

    def test_bfloat16_on_cuda(self):
        m = TwoLayer().to("cuda").bfloat16()
        for dtype in all_param_dtypes(m).values():
            assert dtype == "bfloat16"

    def test_requires_grad_preserved_cuda(self):
        m = TwoLayer().to("cuda").half()
        for p in m.parameters():
            assert p.requires_grad

    def test_nested_cast_cuda(self):
        m = NestedModel().to("cuda").bfloat16()
        for name, dtype in all_param_dtypes(m).items():
            assert dtype == "bfloat16", f"{name} is {dtype}"

    def test_cast_returns_self_cuda(self):
        m = TwoLayer().to("cuda")
        assert m.cast("float16") is m
