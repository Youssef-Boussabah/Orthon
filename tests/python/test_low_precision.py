"""Float16/BFloat16 ties-to-even, subnormal and host/device conversion parity.

full(), cast() and from_numpy() share conversions; checkpoints preserve raw bits."""
from tests.support.capabilities import orthon_cuda_available
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


# Values chosen to hit every branch of the conversion: signed zero, the
# smallest subnormal, the exact halfway point below it, several subnormals, the
# subnormal/normal boundary from both sides, ordinary values, exact rounding
# ties in the normal range, the largest finite half, the overflow boundary
# (65504 stays finite, 65520 rounds to infinity), and the infinities.
VALUES = [
    0.0, -0.0,
    5.960464477539063e-08,     # 2^-24, smallest float16 subnormal
    2.980232238769531e-08,     # 2^-25 exactly: a tie, ties-to-even -> 0
    2.9802327e-08,             # just above 2^-25 -> rounds up to 2^-24
    -2.9802327e-08,
    1e-8, 6e-8, 1e-7, 1e-6, 1e-5, -1e-5,
    6.0975552e-05,             # largest float16 subnormal region
    6.103515625e-05,           # 2^-14, smallest float16 normal
    0.3333333333333333, 0.5, 1.0, -1.0, 3.141592653589793,
    2049.0, 2050.0, 2051.0,    # ties and near-ties in the normal range
    65504.0,                   # float16 max, must stay finite
    65519.0,                   # rounds down to float16 max
    65520.0,                   # rounds up to infinity
    65535.0, 70000.0, 1e30,
    float("inf"), float("-inf"),
]

# bfloat16 shares float32's exponent, so it needs its own tie cases: these two
# float32 values sit exactly halfway between adjacent bfloat16 values, which is
# where round-half-up and round-half-even disagree.
BF16_EXTRA = [
    1.00390625,                # 0x3F808000 -> tie, ties-to-even rounds DOWN
    1.01171875,                # 0x3F818000 -> tie, ties-to-even rounds UP
    3.4028234663852886e38,     # float32 max -> bfloat16 infinity
]


def _f32(value):
    return np.ascontiguousarray(np.array([value], dtype=np.float32))


def _bits(tensor):
    """Raw 16-bit pattern of a one-element low-precision tensor."""
    return int.from_bytes(ortho._tensor_to_bytes(tensor), "little")


def _numpy_f16_bits(value):
    # errstate: overflowing to float16 infinity is the behaviour under test,
    # not a problem to warn about.
    with np.errstate(over="ignore"):
        return int(_f32(value).astype(np.float16).view(np.uint16)[0])


def _torch_bf16_bits(value):
    return int(torch.tensor(_f32(value)).bfloat16().view(torch.uint16).item())


# float16 parity with NumPy, at every entry point.
@pytest.mark.parametrize("value", VALUES)
def test_full_float16_matches_numpy(value):
    assert _bits(ortho.full([1], value, dtype="float16")) == _numpy_f16_bits(value)


@pytest.mark.parametrize("value", VALUES)
def test_cast_float16_matches_numpy(value):
    tensor = ortho.cast(ortho.from_numpy(_f32(value)), "float16")
    assert _bits(tensor) == _numpy_f16_bits(value)


@pytest.mark.parametrize("value", VALUES)
def test_full_and_cast_float16_agree(value):
    assert _bits(ortho.full([1], value, dtype="float16")) == \
        _bits(ortho.cast(ortho.from_numpy(_f32(value)), "float16"))


def test_float16_specific_defects_are_fixed():
    """The values a truncating or mis-shifted converter gets wrong."""
    assert ortho.full([1], 1e-5, dtype="float16").tolist()[0] == \
        pytest.approx(1.0013580322265625e-05)
    assert ortho.cast(ortho.from_numpy(_f32(1e-5)), "float16").tolist()[0] == \
        pytest.approx(1.0013580322265625e-05)
    # Used to underflow to zero: 6e-8 is above 2^-25 and must reach 2^-24.
    assert ortho.full([1], 6e-8, dtype="float16").tolist()[0] == \
        pytest.approx(5.960464477539063e-08)


def test_float16_rounds_to_nearest_even_not_toward_zero():
    """Normals used to TRUNCATE the mantissa. 2051 must round up to 2052."""
    assert ortho.full([1], 2051.0, dtype="float16").tolist()[0] == 2052.0
    assert ortho.full([1], 2050.0, dtype="float16").tolist()[0] == 2050.0
    # Exact tie: 2049 sits between 2048 and 2050, ties-to-even picks 2048.
    assert ortho.full([1], 2049.0, dtype="float16").tolist()[0] == 2048.0


def test_float16_nan_stays_a_nan():
    tensor = ortho.cast(ortho.from_numpy(_f32(float("nan"))), "float16")
    bits = _bits(tensor)
    assert (bits & 0x7C00) == 0x7C00        # exponent all ones
    assert (bits & 0x03FF) != 0             # non-zero payload: NaN, not Inf


def test_float16_signed_zero_preserved():
    assert _bits(ortho.full([1], -0.0, dtype="float16")) == 0x8000
    assert _bits(ortho.full([1], 0.0, dtype="float16")) == 0x0000


# bfloat16 parity with PyTorch, at every entry point.
@pytest.mark.parametrize("value", VALUES + BF16_EXTRA)
def test_full_bfloat16_matches_torch(value):
    assert _bits(ortho.full([1], value, dtype="bfloat16")) == _torch_bf16_bits(value)


@pytest.mark.parametrize("value", VALUES + BF16_EXTRA)
def test_cast_bfloat16_matches_torch(value):
    tensor = ortho.cast(ortho.from_numpy(_f32(value)), "bfloat16")
    assert _bits(tensor) == _torch_bf16_bits(value)


@pytest.mark.parametrize("value", VALUES + BF16_EXTRA)
def test_from_numpy_bfloat16_matches_torch(value):
    tensor = ortho.from_numpy(_f32(value), dtype="bfloat16")
    assert _bits(tensor) == _torch_bf16_bits(value)


@pytest.mark.parametrize("value", VALUES + BF16_EXTRA)
def test_all_three_bfloat16_entry_points_agree(value):
    """One value, one bit pattern, regardless of entry point."""
    from_full = _bits(ortho.full([1], value, dtype="bfloat16"))
    from_cast = _bits(ortho.cast(ortho.from_numpy(_f32(value)), "bfloat16"))
    from_np = _bits(ortho.from_numpy(_f32(value), dtype="bfloat16"))
    assert from_full == from_cast == from_np


def test_bfloat16_specific_defects_are_fixed():
    """65504 used to be 65280 via full() and 65536 via cast()."""
    assert ortho.full([1], 65504.0, dtype="bfloat16").tolist()[0] == 65536.0
    assert ortho.cast(ortho.from_numpy(_f32(65504.0)), "bfloat16").tolist()[0] == 65536.0
    assert ortho.from_numpy(_f32(65504.0), dtype="bfloat16").tolist()[0] == 65536.0


def test_bfloat16_ties_go_to_even_not_up():
    """from_numpy used (bits + 0x8000) >> 16, which rounds ties AWAY from zero.

    0x3F808000 is exactly halfway between 1.0 (0x3F80) and 1.0078125 (0x3F81).
    Ties-to-even picks the even mantissa, 0x3F80.
    """
    assert _bits(ortho.from_numpy(_f32(1.00390625), dtype="bfloat16")) == 0x3F80
    assert _bits(ortho.full([1], 1.00390625, dtype="bfloat16")) == 0x3F80
    # And the neighbouring tie, where even happens to be upward.
    assert _bits(ortho.from_numpy(_f32(1.01171875), dtype="bfloat16")) == 0x3F82


def test_bfloat16_nan_stays_a_nan():
    tensor = ortho.cast(ortho.from_numpy(_f32(float("nan"))), "bfloat16")
    bits = _bits(tensor)
    assert (bits & 0x7F80) == 0x7F80
    assert (bits & 0x007F) != 0


def test_bfloat16_signed_zero_preserved():
    assert _bits(ortho.full([1], -0.0, dtype="bfloat16")) == 0x8000
    assert _bits(ortho.full([1], 0.0, dtype="bfloat16")) == 0x0000


# Round trips through the decode direction.
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("value", VALUES)
def test_low_precision_readback_matches_reference(dtype, value):
    """tolist()/item() decode must agree with the reference framework."""
    tensor = ortho.cast(ortho.from_numpy(_f32(value)), dtype)
    got = tensor.tolist()[0]
    if dtype == "float16":
        with np.errstate(over="ignore"):
            expected = float(_f32(value).astype(np.float16)[0])
    else:
        expected = float(torch.tensor(_f32(value)).bfloat16().float().item())
    if expected != expected:
        assert got != got
    else:
        assert got == expected


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_widening_is_exact(dtype):
    """Every low-precision value is exactly representable in float32."""
    source = np.ascontiguousarray(
        np.array([1.0, -2.5, 0.25, 1e-5, 65504.0 if dtype == "float16" else 1e30],
                 dtype=np.float32))
    low = ortho.cast(ortho.from_numpy(source), dtype)
    wide = ortho.cast(low, "float32")
    assert wide.tolist() == low.tolist()


# the host must agree with the CUDA intrinsics.
@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("value", VALUES)
def test_cpu_cuda_cast_parity(dtype, value):
    """CUDA kernels keep using __float2half / __float2bfloat16.

    Those implement round-to-nearest-even, which is exactly the rule the host
    helpers now implement, so this asserts agreement rather than coupling the
    two implementations together.
    """
    source = _f32(value)
    on_cpu = ortho.cast(ortho.from_numpy(source), dtype)
    on_gpu = ortho.cast(ortho.from_numpy(source).to("cuda"), dtype)
    assert ortho._tensor_to_bytes(on_cpu) == ortho._tensor_to_bytes(on_gpu.to("cpu"))


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_then_transfer_matches_cuda_cast(dtype):
    """full() builds on the host, so its bits must survive the device move."""
    for value in (1e-5, 1.0 / 3.0, 65504.0, -2.25):
        host = ortho.full([1], value, dtype=dtype)
        moved = host.to("cuda")
        assert ortho._tensor_to_bytes(host) == ortho._tensor_to_bytes(moved.to("cpu"))
