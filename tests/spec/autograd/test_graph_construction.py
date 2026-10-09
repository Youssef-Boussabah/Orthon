import numpy as np
import pytest

import orthon as ortho


def test_computed_no_grad_branch_contributes_values_without_a_gradient_path(tensor, assert_grad):
    raw = np.array([1., 2., 3.])
    source = tensor(raw, requires_grad=True)
    other = tensor([2., 4., 5.], requires_grad=True)
    with ortho.no_grad():
        constant = (source * source + 1) * 2
    result = constant * other + source * 3
    (result * result).mean().backward()
    r = 2 * (raw**2 + 1) * np.array([2., 4., 5.]) + 3 * raw
    assert_grad(source, 2 * r)
    assert_grad(other, 4 * r * (raw**2 + 1) / 3)
    assert constant.requires_grad is False and constant.is_leaf is True
    assert constant.grad is None and result.grad is None


def test_no_grad_copy_can_be_a_new_root_alongside_the_original_deep_path(tensor, assert_grad):
    source = tensor([1., 2.], requires_grad=True)
    interior = source * source + 2
    with ortho.no_grad():
        independent = interior.to(interior.device)
    independent.requires_grad_(True)
    root = (interior * 3 + independent * independent).sum()
    root.backward()
    assert_grad(source, [6., 12.])
    assert_grad(independent, [6., 12.])
    assert interior.grad is None


@pytest.mark.parametrize("passthrough", ["contiguous", "same-dtype-cast"])
def test_no_grad_passthrough_keeps_a_preexisting_deep_graph(tensor, assert_grad, passthrough):
    source = tensor([1., 2.], requires_grad=True)
    shared = source * source + 1
    with ortho.no_grad():
        forwarded = shared.contiguous() if passthrough == "contiguous" else shared.cast(shared.dtype)
    output = forwarded * forwarded + source * 5
    output.sum().backward()
    assert_grad(source, [13., 45.])
    assert forwarded.requires_grad and not forwarded.is_leaf
    assert forwarded.grad is None


def test_only_trainable_leaves_receive_grad_in_a_multi_operand_chain(tensor, assert_grad):
    source = tensor([1., 2., -3.], requires_grad=True)
    trainable = tensor([2., 3., 4.], requires_grad=True)
    frozen = tensor([5., 7., 11.])
    intermediate = source * frozen + trainable
    (intermediate * intermediate).sum().backward()
    r = np.array([7., 17., -29.])
    assert_grad(source, 2 * r * [5., 7., 11.])
    assert_grad(trainable, 2 * r)
    assert frozen.grad is None and intermediate.grad is None
