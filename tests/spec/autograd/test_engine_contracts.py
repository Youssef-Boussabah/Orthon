"""Public autograd contracts: captured history, complete DAGs and leaf preflight."""
import pytest
import orthon as ortho


def sibling_roots(tensor):
    x = tensor([2., 3.], requires_grad=True)
    shared = x * x
    return x, shared, (shared * 2 + 1).sum(), (shared * 3 + 2).sum()


def test_preexisting_siblings_survive_ordinary_release(tensor, assert_grad):
    x, shared, first, second = sibling_roots(tensor)
    first.backward()
    assert_grad(x, [8., 12.])
    second.backward()
    assert_grad(x, [20., 30.])
    assert shared.grad is None and first.grad is None and second.grad is None
    for released in (first, second, shared):
        with pytest.raises(RuntimeError, match="freed graph"):
            released.backward()
    assert_grad(x, [20., 30.])


def test_clear_grad_between_siblings(tensor, assert_grad):
    x, _, first, second = sibling_roots(tensor)
    first.backward()
    x.zero_grad()
    second.backward()
    assert_grad(x, [12., 18.])


def test_surviving_sibling_combines_with_fresh_branch(tensor, assert_grad):
    x, _, first, second = sibling_roots(tensor)
    first.backward()
    (second + (x * 5).sum()).backward()
    assert_grad(x, [25., 35.])


def test_outer_root_survives_shared_intermediate_backward(tensor, assert_grad):
    x, shared, outer, _ = sibling_roots(tensor)
    shared.backward()
    assert_grad(x, [4., 6.])
    outer.backward()
    assert_grad(x, [12., 18.])


def test_retained_root_survives_sibling_release(tensor, assert_grad):
    x, _, first, second = sibling_roots(tensor)
    first.backward(retain_graph=True)
    second.backward()
    first.backward()
    assert_grad(x, [28., 42.])
    with pytest.raises(RuntimeError, match="freed graph"):
        first.backward(retain_graph=True)


def test_new_operation_on_released_history_rejects_before_writes(tensor, assert_grad):
    x, shared, first, surviving = sibling_roots(tensor)
    fresh_leaf = tensor([1., 4.], requires_grad=True)
    first.backward()
    invalid = (shared * 4 + fresh_leaf * 5).sum()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="freed graph"):
            invalid.backward()
        assert_grad(x, [8., 12.])
        assert fresh_leaf.grad is None
    surviving.backward()
    assert_grad(x, [20., 30.])


def test_released_history_rejection_preserves_none_after_clear(tensor):
    x, shared, first, _ = sibling_roots(tensor)
    first.backward()
    x.zero_grad()
    invalid = (shared + x * 5).sum()
    with ortho.no_grad():
        with pytest.raises(RuntimeError, match="freed graph"):
            invalid.backward()
        assert ortho.is_grad_enabled() is False
    assert x.grad is None


def test_diamond_with_five_consumers_and_duplicate_input(tensor, assert_grad):
    x = tensor([2., 3.], requires_grad=True)
    shared = x * x
    combined = shared + shared
    for scale in range(1, 6):
        combined = combined + shared * scale
    combined.backward()
    assert_grad(x, [68., 102.])  # (2 + 1 + 2 + 3 + 4 + 5) * 2*x


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"])
def test_unsupported_cpu_leaf_rejects_consistently(dtype):
    leaf = ortho.ones([2], dtype).requires_grad_(True)
    for retain in (False, True):
        with pytest.raises(RuntimeError, match="autograd.*unsupported"):
            leaf.backward(retain_graph=retain)
        assert leaf.grad is None


@pytest.mark.parametrize("dtype", ["float64", "int32", "int64", "uint8", "int8", "bool"])
def test_unsupported_cuda_leaf_rejects_consistently(dtype, cuda_device):
    leaf = ortho.ones([2], dtype).to(cuda_device).requires_grad_(True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="autograd.*unsupported"):
            leaf.backward()
        assert leaf.grad is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mixed_leaf_preflight_preserves_existing_and_absent_grad(dtype):
    unsupported = ortho.ones([2], dtype).requires_grad_(True)
    supported = ortho.ones([2]).requires_grad_(True)
    absent = ortho.ones([2]).requires_grad_(True)
    supported.backward()
    root = (unsupported.cast("float32") * 2 + supported * 3 + absent * 4).sum()
    with pytest.raises(RuntimeError, match="autograd.*unsupported"):
        root.backward()
    assert supported.grad.tolist() == [1., 1.]
    assert unsupported.grad is None and absent.grad is None


def test_supported_direct_leaf_profiles(operator_profile, typed_tensor, assert_output):
    device, dtype = operator_profile
    leaf = typed_tensor([2., 3.], dtype=dtype, device=device).requires_grad_(True)
    for expected, retain in (([1., 1.], False), ([2., 2.], True)):
        leaf.backward(retain_graph=retain)
        assert_output(leaf.grad, expected, dtype=dtype, device=device)
        assert leaf.grad.requires_grad is False and leaf.grad.is_leaf is True
    leaf.zero_grad()
    leaf.backward()
    assert_output(leaf.grad, [1., 1.], dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_chain_4096_ordinary_release(dtype):
    leaf = ortho.ones([2], dtype).requires_grad_(True)
    root = leaf
    for _ in range(4096):
        root = root + 1
    root.backward()
    assert leaf.grad.tolist() == [1., 1.]
    with pytest.raises(RuntimeError, match="freed graph"):
        root.backward()
