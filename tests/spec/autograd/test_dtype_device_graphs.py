import numpy as np
import pytest

import orthon as ortho


def test_supported_arithmetic_profiles_preserve_graph_free_leaf_gradient_metadata(
    operator_profile, typed_tensor, assert_output,
):
    device, dtype = operator_profile
    leaf = typed_tensor([[.5, 1.], [1.5, 2.]], dtype=dtype, device=device).requires_grad_(True)
    shared = leaf * leaf + leaf * 2
    shared.mean().backward(retain_graph=True)
    shared.mean().backward()
    assert_output(leaf.grad, [[1.5, 2.], [2.5, 3.]], dtype=dtype, device=device)
    assert leaf.grad.requires_grad is False and leaf.grad.is_leaf is True
    assert shared.grad is None


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8", "int8", "bool"])
def test_integer_conversion_cuts_one_deep_path_without_blocking_other_trainable_paths(
    tensor, assert_grad, dtype,
):
    source = tensor([.25, 1.5, 2.25], requires_grad=True)
    interior = source * 2 + 1
    encoded = interior.cast(dtype)
    constant = encoded.cast(source.dtype)
    other = tensor([3., 5., 7.], requires_grad=True)
    root = (constant * other + source * 2).sum()
    with pytest.raises(RuntimeError):
        encoded.backward()
    root.backward()
    assert_grad(source, [2., 2., 2.])
    assert_grad(other, constant.tolist())
    assert encoded.grad is None and constant.grad is None and interior.grad is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_floating_cast_inside_a_deep_cpu_graph_returns_the_gradient_to_the_source_dtype(
    dtype, typed_tensor, assert_output,
):
    leaf = typed_tensor([.5, 1., 2.], dtype=dtype).requires_grad_(True)
    changed = leaf.cast("float64" if dtype == "float32" else "float32")
    shared = changed * changed + 1
    (shared * shared).sum().backward()
    assert_output(leaf.grad, [2.5, 8., 40.], dtype=dtype, device="cpu")
    assert changed.grad is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"])
@pytest.mark.parametrize("source_dtype", ["float32", "float64"])
def test_unsupported_cpu_compute_does_not_consume_an_existing_differentiable_graph(
    typed_tensor, assert_grad, dtype, source_dtype,
):
    leaf = typed_tensor([1., 2.], dtype=source_dtype).requires_grad_(True)
    shared = leaf * leaf + 1
    root = (shared * shared).sum()
    with pytest.raises(RuntimeError):
        ortho.neg(ortho.ones([2], dtype))
    assert leaf.grad is None
    root.backward()
    assert_grad(leaf, [8., 40.])
