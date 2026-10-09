"""Tests for ortho.cast() and Tensor.cast()."""
from tests.support.capabilities import orthon_cuda_available
import math
import pytest
import numpy as np

import orthon as ortho

CUDA = orthon_cuda_available()

# Helpers.

def make_f32(values):
    return ortho.from_numpy(np.array(values, dtype=np.float32))

# CPU cast tests.

class TestCpuCast:
    VALUES = [1.0, -2.5, 0.0, 3.14]

    def test_f32_to_f32_same_dtype(self):
        t = make_f32(self.VALUES)
        out = ortho.cast(t, "float32")
        assert out.dtype == "float32"
        assert tuple(out.shape) == (4,)

    def test_f32_to_f64(self):
        t = make_f32(self.VALUES)
        out = ortho.cast(t, "float64")
        assert out.dtype == "float64"
        assert tuple(out.shape) == (4,)
        for i, expected in enumerate(self.VALUES):
            assert abs(out[i].item() - expected) < 1e-5

    def test_f32_to_f16(self):
        t = make_f32([1.0, 2.0, -3.0])
        out = ortho.cast(t, "float16")
        assert out.dtype == "float16"
        assert tuple(out.shape) == (3,)

    def test_f32_to_bf16(self):
        t = make_f32([1.0, 2.0, -3.0])
        out = ortho.cast(t, "bfloat16")
        assert out.dtype == "bfloat16"
        assert tuple(out.shape) == (3,)

    def test_f32_to_int32(self):
        t = make_f32([1.7, -2.9, 0.0, 3.0])
        out = ortho.cast(t, "int32")
        assert out.dtype == "int32"
        vals = [out[i].item() for i in range(4)]
        assert vals == [1, -2, 0, 3]

    def test_f32_to_int64(self):
        t = make_f32([10.0, -20.0])
        out = ortho.cast(t, "int64")
        assert out.dtype == "int64"
        assert [out[i].item() for i in range(2)] == [10, -20]

    def test_f32_to_uint8(self):
        t = make_f32([0.0, 1.0, 255.0])
        out = ortho.cast(t, "uint8")
        assert out.dtype == "uint8"
        assert [out[i].item() for i in range(3)] == [0, 1, 255]

    def test_f32_to_int8(self):
        t = make_f32([0.0, 1.0, -128.0, 127.0])
        out = ortho.cast(t, "int8")
        assert out.dtype == "int8"

    def test_f32_to_bool(self):
        t = make_f32([0.0, 1.0, -3.5, 0.0])
        out = ortho.cast(t, "bool")
        assert out.dtype == "bool"
        vals = [out[i].item() for i in range(4)]
        assert vals == [0, 1, 1, 0]

    def test_f16_to_f32_roundtrip(self):
        t = make_f32([1.0, -2.0, 0.5])
        f16 = ortho.cast(t, "float16")
        back = ortho.cast(f16, "float32")
        assert back.dtype == "float32"
        for i, expected in enumerate([1.0, -2.0, 0.5]):
            assert abs(back[i].item() - expected) < 0.01

    def test_bf16_to_f32_roundtrip(self):
        t = make_f32([1.0, -2.0, 0.5])
        bf16 = ortho.cast(t, "bfloat16")
        back = ortho.cast(bf16, "float32")
        assert back.dtype == "float32"
        for i, expected in enumerate([1.0, -2.0, 0.5]):
            assert abs(back[i].item() - expected) < 0.01

    def test_int32_to_f32(self):
        arr = np.array([1, 2, -3], dtype=np.int32)
        t = ortho.from_numpy(arr)
        assert t.dtype == "int32"
        out = ortho.cast(t, "float32")
        assert out.dtype == "float32"
        for i, expected in enumerate([1.0, 2.0, -3.0]):
            assert abs(out[i].item() - expected) < 1e-6

    def test_bool_to_f32(self):
        arr = np.array([True, False, True], dtype=np.uint8)
        t = ortho.from_numpy(arr).cast("bool")
        out = ortho.cast(t, "float32")
        assert out.dtype == "float32"
        vals = [out[i].item() for i in range(3)]
        assert vals == [1.0, 0.0, 1.0]

    def test_preserves_shape_2d(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        t = ortho.from_numpy(arr)
        out = ortho.cast(t, "float64")
        assert tuple(out.shape) == (3, 4)

    def test_tensor_method_cast(self):
        t = make_f32([1.0, 2.0])
        out = t.cast("float64")
        assert out.dtype == "float64"

    def test_module_level_cast(self):
        t = make_f32([1.0])
        out = ortho.cast(t, "int32")
        assert out.dtype == "int32"

    def test_noncontiguous_input(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = ortho.from_numpy(arr)
        t_T = ortho.transpose(t, 0, 1)
        out = ortho.cast(t_T, "float64")
        assert tuple(out.shape) == (2, 2)
        assert out.dtype == "float64"

    def test_nan_inf_passthrough_f32_f64(self):
        arr = np.array([float('nan'), float('inf'), float('-inf')], dtype=np.float32)
        t = ortho.from_numpy(arr)
        out = ortho.cast(t, "float64")
        assert out.dtype == "float64"
        assert math.isnan(out[0].item())
        assert math.isinf(out[1].item()) and out[1].item() > 0
        assert math.isinf(out[2].item()) and out[2].item() < 0


# CUDA cast tests.

@pytest.mark.skipif(not CUDA, reason="CUDA not available")
class TestCudaCast:
    def _cuda_f32(self, values):
        t = ortho.from_numpy(np.array(values, dtype=np.float32))
        return t.to("cuda")

    def test_f32_to_f16_cuda(self):
        t = self._cuda_f32([1.0, -2.0, 0.5])
        out = ortho.cast(t, "float16")
        assert out.dtype == "float16"
        assert tuple(out.shape) == (3,)
        out_cpu = out.to("cpu")
        back = ortho.cast(out_cpu, "float32")
        for i, expected in enumerate([1.0, -2.0, 0.5]):
            assert abs(back[i].item() - expected) < 0.01

    def test_f16_to_f32_cuda(self):
        t = self._cuda_f32([1.0, -2.0, 0.5])
        f16 = ortho.cast(t, "float16")
        back = ortho.cast(f16, "float32")
        assert back.dtype == "float32"
        back_cpu = back.to("cpu")
        for i, expected in enumerate([1.0, -2.0, 0.5]):
            assert abs(back_cpu[i].item() - expected) < 0.01

    def test_f32_to_bf16_cuda(self):
        t = self._cuda_f32([1.0, -2.0, 0.5])
        out = ortho.cast(t, "bfloat16")
        assert out.dtype == "bfloat16"
        assert tuple(out.shape) == (3,)

    def test_bf16_to_f32_cuda(self):
        t = self._cuda_f32([1.0, -2.0, 0.5])
        bf16 = ortho.cast(t, "bfloat16")
        back = ortho.cast(bf16, "float32")
        assert back.dtype == "float32"
        back_cpu = back.to("cpu")
        for i, expected in enumerate([1.0, -2.0, 0.5]):
            assert abs(back_cpu[i].item() - expected) < 0.02

    def test_same_dtype_cuda_copy(self):
        t = self._cuda_f32([1.0, 2.0])
        out = ortho.cast(t, "float32")
        assert out.dtype == "float32"
        assert tuple(out.shape) == (2,)

    def test_2d_shape_preserved_cuda(self):
        arr = np.ones((2, 3), dtype=np.float32)
        t = ortho.from_numpy(arr).to("cuda")
        out = ortho.cast(t, "float16")
        assert tuple(out.shape) == (2, 3)

    def test_tensor_method_cast_cuda(self):
        t = self._cuda_f32([1.0, 2.0])
        out = t.cast("float16")
        assert out.dtype == "float16"

    def test_f16_to_bf16_cuda(self):
        t = self._cuda_f32([1.0, -2.0])
        f16 = ortho.cast(t, "float16")
        bf16 = ortho.cast(f16, "bfloat16")
        assert bf16.dtype == "bfloat16"

    def test_bf16_to_f16_cuda(self):
        t = self._cuda_f32([1.0, -2.0])
        bf16 = ortho.cast(t, "bfloat16")
        f16 = ortho.cast(bf16, "float16")
        assert f16.dtype == "float16"

    def test_unsupported_pair_raises(self):
        t = self._cuda_f32([1.0])
        t_int = ortho.cast(t.to("cpu"), "int32").to("cuda")
        with pytest.raises(RuntimeError):
            ortho.cast(t_int, "float32")
