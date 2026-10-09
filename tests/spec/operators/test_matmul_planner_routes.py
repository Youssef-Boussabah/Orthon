import orthon._diagnostics as diag
import numpy as np
import pytest
import orthon as ortho


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("batch,m,k,n", [
    (None, 17, 19, 23), (None, 255, 255, 255), (None, 256, 256, 256),
    (None, 257, 257, 257), (2, 17, 19, 23), (4, 160, 160, 160), (4, 164, 164, 164),
])
def test_forced_algorithms_and_automatic_route_agree_numerically(cuda_device, typed_tensor, dtype, batch, m, k, n):
    rng = np.random.default_rng(1300)
    shape_a = (m, k) if batch is None else (batch, m, k)
    shape_b = (k, n) if batch is None else (batch, k, n)
    a = typed_tensor(rng.normal(0, .05, shape_a), dtype=dtype, device=cuda_device)
    b = typed_tensor(rng.normal(0, .05, shape_b), dtype=dtype, device=cuda_device)
    custom = diag.matmul_custom if batch is None else diag.batched_matmul_custom
    library = diag.matmul_cublas if batch is None else diag.batched_matmul_cublas
    outputs = [ortho.matmul(a, b), custom(a, b), library(a, b)]
    expected = np.array(a.tolist()).reshape(shape_a) @ np.array(b.tolist()).reshape(shape_b)
    tolerance = .003 if dtype == "float16" else .025
    for out in outputs:
        assert out.dtype == dtype and out.device == a.device
        np.testing.assert_allclose(np.array(out.tolist()).reshape(expected.shape), expected,
                                   atol=tolerance, rtol=tolerance)
        assert out.requires_grad is False
    work = (1 if batch is None else batch) * m * k * n
    selected = (dtype == "float16" and batch is None) or work >= 16 << 20
    assert diag.matmul_dispatch_path(a, b).startswith("cublas-") == selected
    assert diag.matmul_dispatch_path(a, b).startswith("cublas-" if selected else "custom-tiled-")
    report = diag.matmul_guard_report(a, b)
    assert f"work_items={work}" in report
    assert "cublas_capability_eligible=true" in report
    assert "cuda_ordinal=0" in report
    if batch is None:
        np.testing.assert_allclose(ortho.matmul(a, b).tolist(), outputs[0].tolist(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("batched", [False, True])
def test_cublas_timing_is_genuinely_forced_below_automatic_threshold(cuda_device, typed_tensor, dtype, batched):
    a = typed_tensor(np.ones((2, 3, 4) if batched else (3, 4)), dtype=dtype, device=cuda_device)
    b = typed_tensor(np.ones((2, 4, 5) if batched else (4, 5)), dtype=dtype, device=cuda_device)
    # Forced execution preserves dispatch and shared timing ownership.
    from benchmarks._support.timing import TimingConfig, measure
    from benchmarks._support.environment import synchronizer
    library = diag.batched_matmul_cublas if batched else diag.matmul_cublas
    out = library(a, b)
    np.testing.assert_allclose(out.tolist(), [4.] * (30 if batched else 15))
    stats = measure(lambda: library(a, b), synchronizer(str(a.device)),
                    TimingConfig(warmup=1, samples=3, fixed_inner=1))
    assert stats.sample_count == 3 and stats.inner == 1
    assert "cublas_capability_eligible=true" in diag.matmul_guard_report(a, b)


@pytest.mark.parametrize("batched", [False, True])
def test_force_cublas_rejects_float32_and_forced_rank_mismatch(cuda_device, typed_tensor, batched):
    a = typed_tensor(np.ones((2, 3, 4) if batched else (3, 4)), device=cuda_device)
    b = typed_tensor(np.ones((2, 4, 5) if batched else (4, 5)), device=cuda_device)
    library = diag.batched_matmul_cublas if batched else diag.matmul_cublas
    with pytest.raises(ValueError, match="eligible dtype"):
        library(a, b)
    wrong_rank = diag.matmul_custom if batched else diag.batched_matmul_custom
    with pytest.raises(RuntimeError, match="operands must be"):
        wrong_rank(a, b)
