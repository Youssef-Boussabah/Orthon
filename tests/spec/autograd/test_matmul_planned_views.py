import orthon._diagnostics as diag
import numpy as np
import pytest
import orthon as ortho


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "offset-slice"])
def test_planned_views_preserve_backward_accumulation_and_saved_versions(operator_profile, typed_tensor, batched, layout):
    device, dtype = operator_profile
    raw = np.arange(1, 13, dtype=np.float32).reshape(3, 4) / 16
    if batched:
        raw = np.stack((raw, raw / 2))
    if layout == "transpose":
        owner = typed_tensor(np.swapaxes(raw, -1, -2), dtype=dtype, device=device).requires_grad_(True)
        a = owner.transpose(-1, -2)
    else:
        padded = np.pad(raw, [(0, 0)] * (raw.ndim - 1) + [(1, 1)])
        owner = typed_tensor(padded, dtype=dtype, device=device).requires_grad_(True)
        a = owner.slice(-1, 1, 5)
    b_raw = np.arange(1, 9, dtype=np.float32).reshape(4, 2) / 16
    if batched:
        b_raw = np.stack((b_raw, b_raw / 2))
    b = typed_tensor(b_raw, dtype=dtype, device=device).requires_grad_(True)
    product = a @ b
    assert product._has_grad_fn and product.requires_grad
    if device != "cpu":
        report = diag.matmul_guard_report(a, b)
        assert "packs_lhs=true" in report and "execution_contiguous=true" in report
        assert diag.matmul_dispatch_path(a, b).startswith("cublas-") == (dtype == "float16" and not batched)
        assert diag.matmul_dispatch_path(a, b).startswith("cublas-" if dtype == "float16" and not batched else "custom-tiled-")
    product.sum().backward(retain_graph=True)
    product.sum().backward(retain_graph=True)
    expected_a = np.ones(product.shape) @ np.swapaxes(b_raw, -1, -2) * 2
    expected_owner = np.swapaxes(expected_a, -1, -2) if layout == "transpose" else np.pad(expected_a, [(0, 0)] * (expected_a.ndim - 1) + [(1, 1)])
    tolerance = {"float32": 2e-6, "float64": 1e-12, "float16": .005, "bfloat16": .04}[dtype]
    np.testing.assert_allclose(np.array(owner.grad.tolist()).reshape(expected_owner.shape), expected_owner, atol=tolerance, rtol=tolerance)
    expected_b = np.swapaxes(raw, -1, -2) @ np.ones(product.shape) * 2
    np.testing.assert_allclose(np.array(b.grad.tolist()).reshape(expected_b.shape), expected_b, atol=tolerance, rtol=tolerance)
    with ortho.no_grad():
        if dtype in ("float16", "bfloat16"):
            ortho._amp_copy_master_to_model_(owner, ortho.cast(owner, "float32") * 2)
        else:
            owner.mul_(2)
    with pytest.raises(RuntimeError, match="MatMulNode"):
        product.sum().backward()
