import pytest


@pytest.mark.parametrize("writer", ["base", "reshape", "transpose", "slice", "index"])
def test_saved_values_reject_mutation_through_every_alias_family(tensor, writer):
    leaf = tensor([[1, 2], [3, 4]], requires_grad=True)
    output = leaf * leaf
    alias = {
        "base": lambda: leaf,
        "reshape": lambda: leaf.reshape([4]),
        "transpose": lambda: leaf.transpose(0, 1),
        "slice": lambda: leaf.slice(0, 0, 1),
        "index": lambda: leaf[0],
    }[writer]()
    coordinate = (0,) if alias.ndim == 1 else (0, 0)
    alias[coordinate].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert leaf.grad is None


def test_a_saved_view_is_protected_against_a_write_through_the_base(tensor):
    leaf = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    view = leaf.transpose(0, 1)
    output = view * view
    leaf[0, 1].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert leaf.grad is None


def test_even_a_disjoint_sibling_write_invalidates_saved_values_in_shared_storage(tensor):
    leaf = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    selected = leaf.slice(1, 0, 2)
    output = selected * selected
    leaf[1, 2].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert leaf.grad is None


def test_a_saved_constant_is_protected_even_when_it_does_not_require_grad(tensor):
    leaf = tensor([2, 3], requires_grad=True)
    constant = tensor([5, 7])
    output = leaf * constant
    constant[0].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert leaf.grad is None
    assert constant.grad is None


@pytest.mark.parametrize("write", ["same_values", "change_then_restore"])
def test_successful_writes_invalidate_saved_values_even_if_the_final_bytes_are_equal(tensor, assert_tensor, write):
    leaf = tensor([2, 3], requires_grad=True)
    output = leaf * leaf
    if write == "same_values":
        leaf.mul_(1)
    else:
        leaf.mul_(2)
        leaf.mul_(0.5)
    assert_tensor(leaf, [2, 3])
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert leaf.grad is None


def test_stale_saved_values_fail_before_changing_any_old_or_new_leaf_gradient(tensor, assert_grad):
    first = tensor([2, 3], requires_grad=True)
    second = tensor([5, 7], requires_grad=True)
    new_leaf = tensor([11, 13], requires_grad=True)
    (first * 3).backward()
    (second * 5).backward()
    output = first * first + second + new_leaf
    first[0].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert_grad(first, [3, 3])
    assert_grad(second, [5, 5])
    assert new_leaf.grad is None


def test_value_independent_derivatives_remain_valid_after_source_mutation(tensor, assert_grad):
    leaf = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    transformed = (-leaf).transpose(0, 1).slice(0, 1, 3).contiguous().reshape([4])
    output = (transformed * 3 + 7).sum()
    leaf.mul_(10)
    assert output.item() == -20
    output.backward()
    assert_grad(leaf, [[0, -3, -3], [0, -3, -3]])


@pytest.mark.parametrize("operation", ["tensor_add", "sum", "mean", "scalar_scale", "view"])
def test_payload_independent_backward_allows_writes_to_its_direct_inputs(tensor, assert_grad, operation):
    leaf = tensor([2, 3], requires_grad=True)
    constant = tensor([5, 7])
    output, gradient = {
        "tensor_add": lambda: (leaf + constant, [1, 1]),
        "sum": lambda: (leaf.sum(), [1, 1]),
        "mean": lambda: (leaf.mean(), [0.5, 0.5]),
        "scalar_scale": lambda: (leaf * 3, [3, 3]),
        "view": lambda: (leaf.reshape([2, 1]), [1, 1]),
    }[operation]()
    leaf.mul_(2)
    constant.mul_(2)
    output.backward()
    assert_grad(leaf, gradient)


def test_a_rejected_write_preserves_saved_values_and_the_graph_can_still_run(tensor, assert_grad):
    leaf = tensor([2, 3], requires_grad=True)
    output = leaf * leaf
    with pytest.raises(RuntimeError):
        leaf.add_(tensor([1, 2, 3]))
    output.backward()
    assert_grad(leaf, [4, 6])


def test_a_retained_graph_rejects_an_update_but_a_new_forward_uses_the_updated_values(tensor, assert_grad):
    leaf = tensor([2, -3], requires_grad=True)
    old_output = leaf * leaf
    old_output.backward(retain_graph=True)
    leaf.mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        old_output.backward()
    assert_grad(leaf, [4, -6])
    (leaf * leaf).backward()
    assert_grad(leaf, [12, -18])


def test_creating_and_reading_views_is_not_a_payload_mutation(tensor, assert_grad):
    leaf = tensor([[1, 2], [3, 4]], requires_grad=True)
    output = leaf * leaf
    leaf.reshape([4]).tolist()
    leaf.transpose(0, 1).slice(0, 0, 1)[0].tolist()
    leaf.requires_grad_(True)
    output.backward()
    assert_grad(leaf, [[2, 4], [6, 8]])
