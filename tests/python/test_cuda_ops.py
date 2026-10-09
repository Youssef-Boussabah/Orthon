"""
CUDA kernel tests: add and relu.

CUDA tests are skipped automatically when CUDA is not available.
CPU tests at the bottom verify the existing paths are untouched.
"""
import orthon._diagnostics as diag
from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest
import numpy as np
import orthon as ortho



cuda_available = _cuda_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


# Finite checks.

def test_isfinite_all_float32_cpu_true():
    t = ortho.from_numpy(np.array([1.0, -2.5, 0.0], dtype=np.float32))
    assert ortho.isfinite_all(t) is True
    assert t.isfinite_all() is True


def test_isfinite_all_float32_cpu_false_for_inf_nan():
    inf_t = ortho.from_numpy(np.array([1.0, np.inf], dtype=np.float32))
    nan_t = ortho.from_numpy(np.array([1.0, np.nan], dtype=np.float32))
    assert ortho.isfinite_all(inf_t) is False
    assert ortho.isfinite_all(nan_t) is False


@requires_cuda
def test_isfinite_all_float32_cuda_true():
    t = ortho.from_numpy(np.array([1.0, -2.5, 0.0], dtype=np.float32)).to("cuda")
    assert ortho.isfinite_all(t) is True
    assert t.isfinite_all() is True


@requires_cuda
def test_isfinite_all_float32_cuda_false_for_inf_nan():
    inf_t = ortho.from_numpy(np.array([1.0, np.inf], dtype=np.float32)).to("cuda")
    nan_t = ortho.from_numpy(np.array([1.0, np.nan], dtype=np.float32)).to("cuda")
    assert ortho.isfinite_all(inf_t) is False
    assert ortho.isfinite_all(nan_t) is False


@requires_cuda
def test_isfinite_all_float64_cuda_false_for_inf_nan():
    finite = ortho.from_numpy(np.array([1.0, -2.5], dtype=np.float64)).to("cuda")
    nonfinite = ortho.from_numpy(np.array([1.0, np.inf], dtype=np.float64)).to("cuda")
    assert ortho.isfinite_all(finite) is True
    assert ortho.isfinite_all(nonfinite) is False


@requires_cuda
def test_isfinite_all_cuda_does_not_host_materialize(monkeypatch, capfd):
    monkeypatch.setenv("ORTHON_DEBUG_DEVICE", "1")
    t = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).to("cuda")

    assert ortho.isfinite_all(t) is True

    captured = capfd.readouterr()
    assert "isfinite_all" not in captured.err


@requires_cuda
def test_isfinite_all_float16_cuda_true():
    t = ortho.from_numpy(np.array([1.0, -2.0], dtype=np.float16)).to("cuda")
    assert ortho.isfinite_all(t) is True


@requires_cuda
def test_isfinite_all_float16_cuda_false_for_inf_nan():
    inf_t = ortho.from_numpy(np.array([1.0, np.inf], dtype=np.float16)).to("cuda")
    nan_t = ortho.from_numpy(np.array([1.0, np.nan], dtype=np.float16)).to("cuda")
    assert ortho.isfinite_all(inf_t) is False
    assert ortho.isfinite_all(nan_t) is False


@requires_cuda
def test_isfinite_all_bfloat16_cuda_if_supported():
    try:
        finite = ortho.from_numpy(
            np.array([1.0, -2.0], dtype=np.float32),
            dtype="bfloat16",
        ).to("cuda")
        nonfinite = ortho.from_numpy(
            np.array([1.0, np.inf], dtype=np.float32),
            dtype="bfloat16",
        ).to("cuda")
    except RuntimeError as exc:
        pytest.skip(f"bfloat16 CUDA not available: {exc}")
    assert ortho.isfinite_all(finite) is True
    assert ortho.isfinite_all(nonfinite) is False


@requires_cuda
def test_amp_unscale_to_float32_and_check_cuda_float16():
    grad = ortho.from_numpy(np.array([2.0, -4.0], dtype=np.float16)).to("cuda")
    unscaled, is_finite = ortho._amp_unscale_to_float32_and_check(grad, 0.5)

    assert is_finite is True
    assert unscaled.dtype == "float32"
    assert unscaled.device.is_cuda()
    assert unscaled.to("cpu").tolist() == pytest.approx([1.0, -2.0])


@requires_cuda
def test_amp_unscale_to_float32_and_check_cuda_detects_nonfinite():
    grad = ortho.from_numpy(np.array([2.0, np.inf], dtype=np.float16)).to("cuda")
    _, is_finite = ortho._amp_unscale_to_float32_and_check(grad, 0.5)

    assert is_finite is False


@requires_cuda
def test_amp_copy_master_to_model_inplace_float16():
    master = ortho.from_numpy(np.array([1.25, -2.5], dtype=np.float32)).to("cuda")
    destination = ortho.from_numpy(np.array([0.0, 0.0], dtype=np.float16)).to("cuda")
    before_id = destination._impl_id

    ortho._amp_copy_master_to_model_(destination, master)

    assert destination._impl_id == before_id
    assert destination.dtype == "float16"
    assert destination.to("cpu").tolist() == pytest.approx([1.25, -2.5])


@requires_cuda
def test_amp_copy_master_to_model_inplace_bfloat16_if_supported():
    try:
        destination = ortho.from_numpy(
            np.array([0.0, 0.0], dtype=np.float32),
            dtype="bfloat16",
        ).to("cuda")
    except RuntimeError as exc:
        pytest.skip(f"bfloat16 CUDA not available: {exc}")
    master = ortho.from_numpy(np.array([1.25, -2.5], dtype=np.float32)).to("cuda")
    before_id = destination._impl_id

    ortho._amp_copy_master_to_model_(destination, master)

    assert destination._impl_id == before_id
    assert destination.dtype == "bfloat16"
    assert destination.to("cpu").tolist() == pytest.approx([1.25, -2.5], rel=1e-2)


# Mixed-device error handling.
# These tests work whether CUDA is available or not:
# if CUDA is unavailable the inner try/except skips the test early,
# otherwise the mixed-device op must raise.

def test_mixed_device_add_raises():
    t_cpu = ortho.zeros([4])
    try:
        t_gpu = t_cpu.to("cuda")
    except RuntimeError:
        pytest.skip("CUDA not available")
    with pytest.raises(RuntimeError):
        ortho.add(t_cpu, t_gpu)


def test_mixed_device_add_operator_raises():
    t_cpu = ortho.zeros([4])
    try:
        t_gpu = t_cpu.to("cuda")
    except RuntimeError:
        pytest.skip("CUDA not available")
    with pytest.raises(RuntimeError):
        _ = t_cpu + t_gpu


# CUDA add.

@requires_cuda
def test_cuda_add_matches_cpu():
    a_cpu = ortho.full([1024], 3.0)
    b_cpu = ortho.full([1024], 2.0)
    expected = ortho.add(a_cpu, b_cpu).tolist()

    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    result = ortho.add(a_gpu, b_gpu).to("cpu").tolist()

    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_add_operator_matches_cpu():
    a_cpu = ortho.full([64], 1.0)
    b_cpu = ortho.full([64], 4.0)
    expected = (a_cpu + b_cpu).tolist()

    result = (a_cpu.to("cuda") + b_cpu.to("cuda")).to("cpu").tolist()

    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_add_result_lives_on_cuda():
    a = ortho.full([8], 1.0).to("cuda")
    b = ortho.full([8], 1.0).to("cuda")
    c = a + b
    assert c.device.is_cuda()


@requires_cuda
def test_cuda_add_grid_stride_loop():
    """n > threads*max_blocks to exercise the grid-stride loop."""
    n = 1 << 16  # 65 536 elements, more than fits in one wave
    a = ortho.full([n], 1.0).to("cuda")
    b = ortho.full([n], 2.0).to("cuda")
    result = ortho.add(a, b).to("cpu").tolist()
    assert result == pytest.approx([3.0] * n)


# CUDA relu.

@requires_cuda
def test_cuda_relu_matches_cpu():
    t_cpu = ortho.zeros([5])
    t_cpu[0] = -3.0
    t_cpu[1] = -1.0
    t_cpu[2] = 0.0
    t_cpu[3] = 2.0
    t_cpu[4] = 5.0
    expected = ortho.relu(t_cpu).tolist()

    result = ortho.relu(t_cpu.to("cuda")).to("cpu").tolist()

    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_relu_all_negative():
    t = ortho.full([64], -2.0).to("cuda")
    assert ortho.relu(t).to("cpu").tolist() == pytest.approx([0.0] * 64)


@requires_cuda
def test_cuda_relu_all_positive():
    t = ortho.full([64], 3.0).to("cuda")
    assert ortho.relu(t).to("cpu").tolist() == pytest.approx([3.0] * 64)


@requires_cuda
def test_cuda_relu_result_lives_on_cuda():
    t = ortho.full([8], 1.0).to("cuda")
    assert ortho.relu(t).device.is_cuda()


@requires_cuda
def test_cuda_relu_grid_stride_loop():
    n = 1 << 16
    t = ortho.full([n], -1.0).to("cuda")
    assert ortho.relu(t).to("cpu").tolist() == pytest.approx([0.0] * n)


@requires_cuda
def test_cuda_sqrt_large_linear_weight_shape():
    x = ortho.full([512, 64], 4.0).to("cuda")

    result = ortho.sqrt(x).to("cpu").tolist()

    assert result == pytest.approx([2.0] * (512 * 64))


@requires_cuda
def test_cuda_reciprocal_large_linear_weight_shape():
    x = ortho.full([512, 64], 4.0).to("cuda")

    result = ortho.reciprocal(x).to("cpu").tolist()

    assert result == pytest.approx([0.25] * (512 * 64))


# CUDA broadcast add.

@requires_cuda
def test_cuda_add_broadcast_row():
    """(2, 3) + (3,): row vector broadcast."""
    import numpy as np
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = (a_np + b_np).flatten().tolist()

    # Build tensors element-wise to exercise indexed assignment and broadcasting.
    a_cpu = ortho.zeros([2, 3])
    vals_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for i in range(2):
        for j in range(3):
            a_cpu[[i, j]] = vals_a[i * 3 + j]
    b_cpu = ortho.zeros([3])
    for j in range(3):
        b_cpu[j] = [10.0, 20.0, 30.0][j]

    result = ortho.add(a_cpu.to("cuda"), b_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_add_broadcast_3d():
    """(4, 1, 5) + (1, 3, 5): broadcast over two dimensions."""
    import numpy as np
    rng = np.random.default_rng(0)
    a_np = rng.random((4, 1, 5)).astype(np.float32)
    b_np = rng.random((1, 3, 5)).astype(np.float32)
    expected = (a_np + b_np).flatten().tolist()

    # Build as flat tensors and reshape
    a_cpu = ortho.zeros([4, 1, 5])
    for i, v in enumerate(a_np.flatten().tolist()):
        a_cpu[[i // 5, 0, i % 5]] = v
    b_cpu = ortho.zeros([1, 3, 5])
    b_flat = b_np.flatten().tolist()
    for i, v in enumerate(b_flat):
        b_cpu[[0, i // 5, i % 5]] = v

    result = ortho.add(a_cpu.to("cuda"), b_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected, rel=1e-5)


@requires_cuda
def test_cuda_add_broadcast_scalar_left():
    """(1,) + (n,): scalar-like tensor broadcast."""
    n = 64
    a_cpu = ortho.full([1], 5.0)
    b_cpu = ortho.full([n], 3.0)
    expected = [8.0] * n

    result = ortho.add(a_cpu.to("cuda"), b_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_add_broadcast_scalar_right():
    """(n,) + (1,): scalar-like tensor broadcast."""
    n = 64
    a_cpu = ortho.full([n], 7.0)
    b_cpu = ortho.full([1], 2.0)
    expected = [9.0] * n

    result = ortho.add(a_cpu.to("cuda"), b_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_add_broadcast_output_shape():
    """Output shape is the broadcasted shape."""
    a_cpu = ortho.full([2, 1, 4], 1.0)
    b_cpu = ortho.full([3, 4], 1.0)
    result = ortho.add(a_cpu.to("cuda"), b_cpu.to("cuda"))
    assert tuple(result.shape) == (2, 3, 4)


@requires_cuda
def test_cuda_add_broadcast_output_on_cuda():
    """Broadcasting result lives on CUDA."""
    a = ortho.full([2, 3], 1.0).to("cuda")
    b = ortho.full([3], 1.0).to("cuda")
    assert ortho.add(a, b).device.is_cuda()


@requires_cuda
def test_cuda_add_same_shape_still_works():
    """Same-shape add is unaffected by broadcasting changes."""
    a = ortho.full([4, 4], 2.0).to("cuda")
    b = ortho.full([4, 4], 3.0).to("cuda")
    result = ortho.add(a, b).to("cpu").tolist()
    assert result == pytest.approx([5.0] * 16)


# CUDA sum.

@requires_cuda
def test_cuda_sum_small():
    t = ortho.full([8], 2.0).to("cuda")
    assert ortho.sum(t).item() == pytest.approx(16.0)


@requires_cuda
def test_cuda_sum_large():
    """Large tensor exercises grid-stride loop (n > kThreads * kMaxBlocks)."""
    n = 1 << 17  # 131 072, multiple grid-stride rounds
    t = ortho.full([n], 1.0).to("cuda")
    assert ortho.sum(t).item() == pytest.approx(float(n), rel=1e-4)


@requires_cuda
def test_cuda_sum_non_power_of_two():
    n = 100
    t_cpu = ortho.zeros([n])
    for i in range(n):
        t_cpu[i] = float(i + 1)   # 1 + 2 + ... + 100 = 5050
    assert ortho.sum(t_cpu.to("cuda")).item() == pytest.approx(5050.0, rel=1e-4)


@requires_cuda
def test_cuda_sum_2d():
    """2D contiguous tensor reduces to same value as CPU."""
    t_cpu = ortho.full([4, 8], 3.0)
    expected = ortho.sum(t_cpu).item()
    assert ortho.sum(t_cpu.to("cuda")).item() == pytest.approx(expected, rel=1e-5)


@requires_cuda
def test_cuda_sum_matches_cpu():
    """Random-like values: CUDA sum matches CPU sum."""
    t_cpu = ortho.zeros([256])
    for i in range(256):
        t_cpu[i] = float((i * 7 + 3) % 17) - 8.0   # values in [-8, 8]
    expected = ortho.sum(t_cpu).item()
    assert ortho.sum(t_cpu.to("cuda")).item() == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_sum_empty_raises():
    t_cpu = ortho.empty([0])
    try:
        t_gpu = t_cpu.to("cuda")
    except RuntimeError:
        pytest.skip("CUDA not available")
    with pytest.raises(RuntimeError, match="empty"):
        ortho.sum(t_gpu)


# CUDA mean.

@requires_cuda
def test_cuda_mean_small():
    t = ortho.full([4], 3.0).to("cuda")
    assert ortho.mean(t).item() == pytest.approx(3.0)


@requires_cuda
def test_cuda_mean_large():
    n = 1 << 17
    t = ortho.full([n], 2.5).to("cuda")
    assert ortho.mean(t).item() == pytest.approx(2.5, rel=1e-4)


@requires_cuda
def test_cuda_mean_mixed():
    """Positive and negative values; CUDA mean matches CPU mean."""
    t_cpu = ortho.zeros([128])
    for i in range(128):
        t_cpu[i] = float(i) - 63.5   # symmetric around 0, mean ≈ 0
    expected = ortho.mean(t_cpu).item()
    assert ortho.mean(t_cpu.to("cuda")).item() == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_mean_2d():
    t_cpu = ortho.full([8, 8], 5.0)
    expected = ortho.mean(t_cpu).item()
    assert ortho.mean(t_cpu.to("cuda")).item() == pytest.approx(expected)


@requires_cuda
def test_cuda_mean_empty_raises():
    t_cpu = ortho.empty([0])
    try:
        t_gpu = t_cpu.to("cuda")
    except RuntimeError:
        pytest.skip("CUDA not available")
    with pytest.raises(RuntimeError, match="empty"):
        ortho.mean(t_gpu)


# CUDA matmul.

@requires_cuda
def test_cuda_matmul_square():
    """ones([4,4]) @ ones([4,4]) = full([4,4], 4.0)."""
    A = ortho.ones([4, 4]).to("cuda")
    B = ortho.ones([4, 4]).to("cuda")
    C = ortho.matmul(A, B)
    assert C.to("cpu").tolist() == pytest.approx([4.0] * 16)


@requires_cuda
def test_cuda_matmul_rectangular():
    """ones([2,3]) @ ones([3,4]) = full([2,4], 3.0)."""
    A = ortho.ones([2, 3]).to("cuda")
    B = ortho.ones([3, 4]).to("cuda")
    C = ortho.matmul(A, B)
    assert C.to("cpu").tolist() == pytest.approx([3.0] * 8)


@requires_cuda
def test_cuda_matmul_output_shape():
    A = ortho.ones([5, 3]).to("cuda")
    B = ortho.ones([3, 7]).to("cuda")
    assert tuple(ortho.matmul(A, B).shape) == (5, 7)


@requires_cuda
def test_cuda_matmul_output_device():
    A = ortho.ones([3, 3]).to("cuda")
    B = ortho.ones([3, 3]).to("cuda")
    assert ortho.matmul(A, B).device.is_cuda()


@requires_cuda
def test_cuda_matmul_values_vs_cpu():
    """Non-trivial values: CUDA result matches CPU result."""
    A_cpu = ortho.zeros([4, 3])
    B_cpu = ortho.zeros([3, 4])
    for i in range(4):
        for j in range(3):
            A_cpu[[i, j]] = float(i * 3 + j + 1)   # 1..12
    for i in range(3):
        for j in range(4):
            B_cpu[[i, j]] = float(i * 4 + j + 1)   # 1..12
    expected = ortho.matmul(A_cpu, B_cpu).tolist()
    result   = ortho.matmul(A_cpu.to("cuda"), B_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_matmul_larger():
    """32x64 @ 64x16 exercises multiple thread blocks in both dims."""
    A = ortho.full([32, 64], 1.0).to("cuda")
    B = ortho.full([64, 16], 1.0).to("cuda")
    C = ortho.matmul(A, B)
    assert tuple(C.shape) == (32, 16)
    assert C.to("cpu").tolist() == pytest.approx([64.0] * (32 * 16), rel=1e-4)


@requires_cuda
def test_cuda_matmul_non_multiple_tile_17x19x23():
    """(17,19) @ (19,23): all dims cross a 16-tile boundary."""
    A_cpu = ortho.zeros([17, 19])
    B_cpu = ortho.zeros([19, 23])
    for i in range(17):
        for j in range(19):
            A_cpu[[i, j]] = float((i * 19 + j) % 7 + 1)
    for i in range(19):
        for j in range(23):
            B_cpu[[i, j]] = float((i * 23 + j) % 5 + 1)
    expected = ortho.matmul(A_cpu, B_cpu).tolist()
    result   = ortho.matmul(A_cpu.to("cuda"), B_cpu.to("cuda")).to("cpu").tolist()
    assert tuple(ortho.matmul(A_cpu.to("cuda"), B_cpu.to("cuda")).shape) == (17, 23)
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_matmul_non_multiple_tile_31x33x7():
    """(31,33) @ (33,7): M and K cross tile boundaries, N < tile."""
    A_cpu = ortho.zeros([31, 33])
    B_cpu = ortho.zeros([33, 7])
    for i in range(31):
        for j in range(33):
            A_cpu[[i, j]] = float((i + j) % 9 + 1)
    for i in range(33):
        for j in range(7):
            B_cpu[[i, j]] = float((i * 7 + j) % 11 + 1)
    expected = ortho.matmul(A_cpu, B_cpu).tolist()
    result   = ortho.matmul(A_cpu.to("cuda"), B_cpu.to("cuda")).to("cpu").tolist()
    assert tuple(ortho.matmul(A_cpu.to("cuda"), B_cpu.to("cuda")).shape) == (31, 7)
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_matmul_tiled_output_device():
    """Tiled kernel result lives on CUDA."""
    A = ortho.ones([17, 19]).to("cuda")
    B = ortho.ones([19, 23]).to("cuda")
    assert ortho.matmul(A, B).device.is_cuda()


@requires_cuda
def test_cuda_matmul_tiled_output_shape():
    """Tiled kernel output shape is correct for non-multiple tile sizes."""
    A = ortho.ones([31, 33]).to("cuda")
    B = ortho.ones([33, 7]).to("cuda")
    assert tuple(ortho.matmul(A, B).shape) == (31, 7)


@requires_cuda
def test_cuda_matmul_non_contiguous_matches_cpu():
    """Transposed CUDA inputs are materialized internally for matmul."""
    A = ortho.ones([4, 4]).to("cuda")
    A_T = ortho.transpose(A, 0, 1)   # non-contiguous view of CUDA storage
    B = ortho.ones([4, 4]).to("cuda")
    result = ortho.matmul(A_T, B).to("cpu").tolist()
    expected = ortho.matmul(
        ortho.transpose(ortho.ones([4, 4]), 0, 1),
        ortho.ones([4, 4]),
    ).tolist()
    assert result == pytest.approx(expected)


@requires_cuda
def test_cuda_matmul_mixed_device_raises():
    A_cpu  = ortho.ones([3, 3])
    A_cuda = A_cpu.to("cuda")
    with pytest.raises(RuntimeError):
        ortho.matmul(A_cpu, A_cuda)


@requires_cuda
def test_cuda_matmul_shape_mismatch_raises():
    """Shape mismatch is still caught before reaching the CUDA path."""
    A = ortho.ones([3, 4]).to("cuda")
    B = ortho.ones([3, 4]).to("cuda")
    with pytest.raises(RuntimeError):
        ortho.matmul(A, B)


# CUDA softmax.

@requires_cuda
def test_cuda_softmax_basic():
    """[0, 1, 2] row: values increase monotonically after softmax."""
    t_cpu = ortho.zeros([1, 3])
    t_cpu[[0, 0]] = 0.0
    t_cpu[[0, 1]] = 1.0
    t_cpu[[0, 2]] = 2.0
    result = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    assert len(result) == 3
    assert result[0] < result[1] < result[2]
    assert sum(result) == pytest.approx(1.0, abs=1e-5)


@requires_cuda
def test_cuda_softmax_rows_sum_to_one():
    """Every row of the output sums to 1.0."""
    M, N = 8, 16
    t_cpu = ortho.zeros([M, N])
    for i in range(M):
        for j in range(N):
            t_cpu[[i, j]] = float(i * N + j)
    result = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    # Reshape flat list into rows
    for i in range(M):
        row_sum = sum(result[i * N: (i + 1) * N])
        assert row_sum == pytest.approx(1.0, abs=1e-5)


@requires_cuda
def test_cuda_softmax_numerical_stability():
    """Large values [1000, 1001, 1002] must not overflow to inf/nan."""
    t_cpu = ortho.zeros([1, 3])
    t_cpu[[0, 0]] = 1000.0
    t_cpu[[0, 1]] = 1001.0
    t_cpu[[0, 2]] = 1002.0
    result = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    for v in result:
        assert v == v        # nan check
        assert v != float("inf")
    assert sum(result) == pytest.approx(1.0, abs=1e-5)


@requires_cuda
def test_cuda_softmax_negative_values():
    """All-negative row still produces a valid probability distribution."""
    t_cpu = ortho.zeros([1, 4])
    for j in range(4):
        t_cpu[[0, j]] = float(-10 - j)
    result = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    for v in result:
        assert v > 0.0
    assert sum(result) == pytest.approx(1.0, abs=1e-5)


@requires_cuda
def test_cuda_softmax_random_vs_cpu():
    """CUDA softmax matches CPU reference on non-trivial values."""
    M, N = 5, 7
    t_cpu = ortho.zeros([M, N])
    for i in range(M):
        for j in range(N):
            t_cpu[[i, j]] = float((i * 7 + j * 3 + 1) % 13) - 6.0
    expected = ortho.softmax(t_cpu).tolist()
    result   = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected, rel=1e-4, abs=1e-6)


@requires_cuda
def test_cuda_softmax_output_shape():
    """Output shape matches input shape."""
    t = ortho.full([3, 5], 1.0).to("cuda")
    assert tuple(ortho.softmax(t).shape) == (3, 5)


@requires_cuda
def test_cuda_softmax_output_device():
    """Output lives on CUDA."""
    t = ortho.full([2, 4], 1.0).to("cuda")
    assert ortho.softmax(t).device.is_cuda()


@requires_cuda
def test_cuda_softmax_non_multiple_block_size():
    """N = 300 > 256 exercises the grid-stride loop inside each block."""
    M, N = 4, 300
    t_cpu = ortho.zeros([M, N])
    for i in range(M):
        for j in range(N):
            t_cpu[[i, j]] = float((i + j) % 17) - 8.0
    expected = ortho.softmax(t_cpu).tolist()
    result   = ortho.softmax(t_cpu.to("cuda")).to("cpu").tolist()
    assert result == pytest.approx(expected, rel=1e-4, abs=1e-6)
    for i in range(M):
        row_sum = sum(result[i * N: (i + 1) * N])
        assert row_sum == pytest.approx(1.0, abs=1e-4)


@requires_cuda
def test_cuda_softmax_empty_raises():
    """Empty tensor raises RuntimeError."""
    t_cpu = ortho.empty([0])
    try:
        t_gpu = t_cpu.to("cuda")
    except RuntimeError:
        pytest.skip("CUDA not available")
    # empty([0]) is 1D; softmax needs 2D, so either check can fire.
    with pytest.raises(RuntimeError):
        ortho.softmax(t_gpu)


# CUDA cross entropy.

@requires_cuda
def test_cuda_cross_entropy_simple():
    """logits=[0,1,0], target=1 → loss = log(e^0+e^1+e^0) - 1."""
    import math
    logits_cpu = ortho.zeros([1, 3])
    logits_cpu[[0, 1]] = 1.0
    expected = math.log(math.exp(0) + math.exp(1) + math.exp(0)) - 1.0
    result = ortho.cross_entropy(logits_cpu.to("cuda"), [1])
    assert result.item() == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_cross_entropy_vs_cpu():
    """CUDA result matches CPU reference on non-trivial values."""
    B, C = 6, 5
    logits_cpu = ortho.zeros([B, C])
    targets = [(i * 3 + 1) % C for i in range(B)]
    for i in range(B):
        for j in range(C):
            logits_cpu[[i, j]] = float((i * C + j * 2 + 1) % 11) - 5.0
    expected = ortho.cross_entropy(logits_cpu, targets).item()
    result   = ortho.cross_entropy(logits_cpu.to("cuda"), targets).item()
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_cross_entropy_numerical_stability():
    """Large logits [1000,1001,1002], target=2 must not produce NaN/inf."""
    import math
    logits_cpu = ortho.zeros([1, 3])
    logits_cpu[[0, 0]] = 1000.0
    logits_cpu[[0, 1]] = 1001.0
    logits_cpu[[0, 2]] = 1002.0
    result = ortho.cross_entropy(logits_cpu.to("cuda"), [2]).item()
    assert result == result              # nan check
    assert result != float("inf")
    # With subtract-max: -1002 + 1002 + log(exp(-2)+exp(-1)+exp(0))
    expected = math.log(math.exp(-2) + math.exp(-1) + math.exp(0))
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_cross_entropy_batch_size_1():
    """Single example: result equals the per-row loss."""
    import math
    logits_cpu = ortho.zeros([1, 4])
    for j in range(4):
        logits_cpu[[0, j]] = float(j)   # [0, 1, 2, 3]
    rmax = 3.0
    sum_exp = sum(math.exp(v - rmax) for v in [0, 1, 2, 3])
    expected = -2.0 + rmax + math.log(sum_exp)   # target=2
    result = ortho.cross_entropy(logits_cpu.to("cuda"), [2]).item()
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_cross_entropy_many_classes():
    """C=300 > 256 exercises the grid-stride loop inside each block."""
    B, C = 4, 300
    logits_cpu = ortho.zeros([B, C])
    targets = [j % C for j in range(B)]
    for i in range(B):
        for j in range(C):
            logits_cpu[[i, j]] = float((i + j) % 13) - 6.0
    expected = ortho.cross_entropy(logits_cpu, targets).item()
    result   = ortho.cross_entropy(logits_cpu.to("cuda"), targets).item()
    assert result == pytest.approx(expected, rel=1e-4)


@requires_cuda
def test_cuda_cross_entropy_output_is_scalar():
    """Return value is a scalar Tensor (shape {1})."""
    logits = ortho.full([3, 4], 1.0).to("cuda")
    result = ortho.cross_entropy(logits, [0, 1, 2])
    assert isinstance(result, ortho.Tensor)
    assert result.numel() == 1
    assert result.device.is_cuda()


@requires_cuda
def test_debug_device_logs_cuda_host_materialization(monkeypatch, capfd):
    monkeypatch.setenv("ORTHON_DEBUG_DEVICE", "1")
    loss = ortho.cross_entropy(ortho.full([2, 3], 1.0).to("cuda"), [0, 1])

    assert loss.device.is_cuda()
    _ = loss.item()

    captured = capfd.readouterr()
    assert "Tensor.item" in captured.err
    assert "copying CUDA scalar to CPU" in captured.err


# Validation tests run before GPU dispatch; no CUDA required.

def test_cross_entropy_invalid_target_raises():
    logits = ortho.ones([2, 3])
    with pytest.raises(RuntimeError):
        ortho.cross_entropy(logits, [0, 3])   # 3 >= num_classes=3
    with pytest.raises(RuntimeError):
        ortho.cross_entropy(logits, [0, -1])  # negative index


def test_cross_entropy_shape_mismatch_raises():
    logits = ortho.ones([3, 4])
    with pytest.raises(RuntimeError):
        ortho.cross_entropy(logits, [0, 1])   # 2 targets for batch_size=3


def test_cross_entropy_empty_batch_raises():
    """B=0 raises before any GPU work."""
    logits = ortho.empty([0, 4])             # 2D, 0 rows
    with pytest.raises(RuntimeError):
        ortho.cross_entropy(logits, [])


# CPU smoke tests that must pass regardless of CUDA.

def test_cpu_add_unchanged():
    a = ortho.full([3], 2.0)
    b = ortho.full([3], 3.0)
    assert ortho.add(a, b).tolist() == pytest.approx([5.0, 5.0, 5.0])


def test_cpu_add_broadcast_unchanged():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3], 1.0)
    assert ortho.add(a, b).tolist() == pytest.approx([3.0] * 6)


def test_cpu_relu_unchanged():
    t = ortho.zeros([4])
    t[0] = -2.0
    t[1] = -1.0
    t[2] = 0.0
    t[3] = 3.0
    assert ortho.relu(t).tolist() == pytest.approx([0.0, 0.0, 0.0, 3.0])


def test_cpu_add_autograd_unchanged():
    a = ortho.full([2], 1.0)
    b = ortho.full([2], 2.0)
    a.requires_grad_(True)
    b.requires_grad_(True)
    out = a + b
    out.backward()
    assert a.grad.tolist() == pytest.approx([1.0, 1.0])
    assert b.grad.tolist() == pytest.approx([1.0, 1.0])


# Zero-element tensors: valid empty inputs behave the same on CPU and CUDA
# (no zero-block kernel launches).

def test_empty_tensor_elementwise_cpu():
    x = ortho.zeros([0, 3])
    assert tuple(ortho.relu(x).shape) == (0, 3)
    assert tuple(ortho.neg(x).shape) == (0, 3)
    assert tuple(ortho.sigmoid(x).shape) == (0, 3)
    assert tuple(ortho.add(x, x).shape) == (0, 3)
    assert tuple(ortho.mul(x, x).shape) == (0, 3)
    assert ortho.zeros([0]).tolist() == []


def test_empty_tensor_broadcast_add_cpu():
    # A size-1 dim stretches to 0, never the other way: [0,3] + [3] is [0,3].
    a = ortho.zeros([0, 3])
    b = ortho.ones([3])
    assert tuple(ortho.add(a, b).shape) == (0, 3)
    assert tuple(ortho.mul(a, b).shape) == (0, 3)


@requires_cuda
def test_empty_tensor_elementwise_cuda_matches_cpu():
    x = ortho.zeros([0, 3]).to("cuda")
    for op in (ortho.relu, ortho.neg, ortho.sigmoid, ortho.tanh,
               ortho.gelu, ortho.exp, ortho.sqrt, ortho.reciprocal):
        out = op(x)
        assert tuple(out.shape) == (0, 3)
        assert out.device.is_cuda()
    assert ortho.zeros([0]).to("cuda").to("cpu").tolist() == []


@requires_cuda
def test_empty_tensor_add_mul_cuda():
    x = ortho.zeros([0, 3]).to("cuda")
    assert tuple(ortho.add(x, x).shape) == (0, 3)
    assert tuple(ortho.mul(x, x).shape) == (0, 3)


@requires_cuda
def test_empty_tensor_broadcast_add_cuda():
    a = ortho.zeros([0, 3]).to("cuda")
    b = ortho.ones([3]).to("cuda")
    out = ortho.add(a, b)
    assert tuple(out.shape) == (0, 3)
    assert out.device.is_cuda()


@requires_cuda
def test_empty_tensor_cast_cuda():
    x = ortho.zeros([0]).to("cuda")
    y = ortho.cast(x, "float16")
    assert tuple(y.shape) == (0,)
    assert y.device.is_cuda()


# Private matmul benchmark diagnostics: not public, and validated.

def test_matmul_benchmark_names_are_not_public():
    assert not hasattr(ortho, "matmul_naive")
    assert not hasattr(ortho, "matmul_tiled")


@requires_cuda
def test_matmul_diagnostics_reject_invalid_rank():
    a1 = ortho.ones([4]).to("cuda")          # 1-D: previously undefined behavior
    b  = ortho.ones([4, 4]).to("cuda")
    with pytest.raises(RuntimeError, match="2-D"):
        diag.matmul_naive(a1, b)
    with pytest.raises(RuntimeError, match="2-D"):
        diag.matmul_custom(a1, b)


@requires_cuda
def test_matmul_diagnostics_reject_cpu_operands():
    a_cpu = ortho.ones([4, 4])
    b_gpu = ortho.ones([4, 4]).to("cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        diag.matmul_naive(a_cpu, b_gpu)
    with pytest.raises(RuntimeError, match="CUDA"):
        diag.matmul_custom(a_cpu, b_gpu)


@requires_cuda
def test_matmul_diagnostics_reject_inner_dim_mismatch():
    a = ortho.ones([4, 3]).to("cuda")
    b = ortho.ones([4, 4]).to("cuda")
    with pytest.raises(RuntimeError, match="inner dimensions"):
        diag.matmul_naive(a, b)


@requires_cuda
def test_matmul_diagnostics_valid_call_still_works():
    a = ortho.ones([4, 5]).to("cuda")
    b = ortho.ones([5, 6]).to("cuda")
    out = diag.matmul_naive(a, b)
    assert tuple(out.shape) == (4, 6)
    assert out.to("cpu").tolist() == [5.0] * 24
    out_t = diag.matmul_custom(a, b)
    assert out_t.to("cpu").tolist() == [5.0] * 24
