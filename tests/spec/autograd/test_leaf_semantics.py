import pytest

import orthon as ortho


def test_a_factory_result_is_a_leaf_regardless_of_its_gradient_flag(tensor):
    created = tensor([1, 2])
    assert created.is_leaf is True
    assert created.requires_grad is False
    assert created.grad is None
    assert created.requires_grad_(True) is created
    assert created.is_leaf is True
    created.requires_grad_(False)
    assert created.is_leaf is True


@pytest.mark.parametrize("operation", ["arithmetic", "view", "copy"])
def test_a_tracked_result_is_not_a_leaf_and_does_not_receive_grad(tensor, assert_grad, operation):
    source = tensor([[1, 2], [3, 4]], requires_grad=True)
    result = {
        "arithmetic": lambda: source * 3,
        "view": lambda: source.transpose(0, 1),
        "copy": lambda: source.to(source.device),
    }[operation]()
    assert result.requires_grad is True
    assert result.is_leaf is False
    result.backward()
    scale = 3 if operation == "arithmetic" else 1
    assert_grad(source, [[scale, scale], [scale, scale]])
    assert result.grad is None
    assert result.is_leaf is False


def test_a_view_created_without_recording_can_become_an_independent_gradient_leaf(tensor, assert_grad):
    source = tensor([[1, 2], [3, 4]], requires_grad=True)
    with ortho.no_grad():
        view = source.reshape([4])
    assert view.requires_grad is False
    assert view.is_leaf is True
    view.requires_grad_(True)
    (view * 3).backward()
    assert_grad(view, [3, 3, 3, 3])
    assert source.grad is None
    view[0].mul_(2)
    assert source[0, 0].item() == 2


def test_a_copy_created_without_recording_can_be_trained_without_reaching_its_source(tensor, assert_grad):
    source = tensor([2, 3], requires_grad=True)
    with ortho.no_grad():
        snapshot = source.to(source.device)
    assert snapshot.requires_grad is False
    assert snapshot.is_leaf is True
    snapshot.requires_grad_(True)
    (snapshot * 4).backward()
    assert_grad(snapshot, [4, 4])
    assert source.grad is None
    snapshot.mul_(3)
    assert source.tolist() == [2, 3]


@pytest.mark.parametrize("operation", ["contiguous", "same_dtype_cast"])
def test_no_grad_does_not_detach_an_existing_tensor_returned_by_a_noop(tensor, assert_grad, operation):
    source = tensor([2, 3], requires_grad=True)
    intermediate = source * 2
    with ortho.no_grad():
        result = intermediate.contiguous() if operation == "contiguous" else intermediate.cast(intermediate.dtype)
    assert result.requires_grad is True
    assert result.is_leaf is False
    result.backward()
    assert_grad(source, [2, 2])
    assert intermediate.grad is None


def test_integer_conversion_ends_gradient_tracking(tensor):
    source = tensor([2, 3], requires_grad=True)
    converted = source.cast("int32")
    assert converted.requires_grad is False
    assert converted.is_leaf is True
    with pytest.raises(RuntimeError):
        converted.backward()
    assert source.grad is None
