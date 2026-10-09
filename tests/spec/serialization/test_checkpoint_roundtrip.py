import math

import pytest

import orthon as ortho
from orthon.optim import Adam

from .workload import RegressionModel, advance, training_state, value_tree


@pytest.mark.parametrize("api", ["state", "checkpoint"])
def test_nested_model_roundtrip_preserves_destination_owners_flags_and_gradients(
    tensor_case, tmp_path, api,
):
    device, dtype = tensor_case
    source = RegressionModel(dtype, device).eval()
    source.spare.requires_grad_(False)
    source.register_buffer("count", ortho.full([1], 7, dtype="int64").to(device))
    path = tmp_path / "model.data"
    if api == "state":
        assert ortho.save_state_dict(source, path) is None
    else:
        assert ortho.save_checkpoint(path, model=source) is None
    target = RegressionModel(dtype, device)
    target.register_buffer("count", ortho.zeros([1], dtype="int64").to(device))
    target.cache.mul_(2.)
    target.ordinary.mul_(2.)
    (target.weight.sum() + target.head.bias.sum()).backward()
    parameters = list(target.parameters())
    gradients = [None if p.grad is None else p.grad.tolist() for p in parameters]
    old_offset = target.offset
    if api == "state":
        assert ortho.load_state_dict(target, path) is None
    else:
        assert ortho.load_checkpoint(path, model=target) == {}
    assert value_tree(target.state_dict()) == value_tree(source.state_dict())
    assert all(p is old for p, old in zip(target.parameters(), parameters))
    assert all(p.requires_grad and p.is_leaf for p in parameters)
    assert [None if p.grad is None else p.grad.tolist() for p in parameters] == gradients
    assert target.training and target.head.training
    assert target.offset is not old_offset
    assert target.cache.tolist() == [84.]
    assert target.ordinary.tolist() == [198.]
    source.weight.mul_(2.)
    assert target.weight.tolist() == [0.5, -0.25]
    assert old_offset.tolist() == [0.25]


@pytest.mark.parametrize("api", ["state", "checkpoint"])
def test_saving_leaves_existing_gradients_and_saved_value_graph_usable(
    tensor, tmp_path, api, assert_grad,
):
    from orthon.nn import Module, Parameter

    p = tensor([2., 3.], requires_grad=True)
    model = Module()
    model.weight = Parameter(p)
    p.sum().backward()
    captured_gradient = p.grad
    pending = (p * p).sum()
    path = tmp_path / "observational.ckpt"
    if api == "state":
        ortho.save_state_dict(model, path)
    else:
        ortho.save_checkpoint(path, model=model, metadata={"tracked": p})
    assert model.weight is p
    assert p.requires_grad and p.is_leaf
    assert captured_gradient.tolist() == [1., 1.]
    pending.backward()
    assert_grad(p, [5., 7.])


def test_optimizer_only_restore_preserves_parameter_data_and_restores_public_progress(
    training_pair, tmp_path,
):
    source, optimizer = training_pair()
    advance(source, optimizer, 2)
    path = tmp_path / "optimizer.ckpt"
    ortho.save_checkpoint(path, optimizer=optimizer, metadata={"steps": 2})
    expected = value_tree(optimizer.state_dict())
    destination, restored = training_pair(lr=0.5)
    before = value_tree(destination.state_dict())
    assert ortho.load_checkpoint(path, optimizer=restored) == {"steps": 2}
    assert value_tree(destination.state_dict()) == before
    assert value_tree(restored.state_dict()) == expected


def test_state_only_loader_can_apply_model_from_combined_checkpoint(training_pair, tmp_path):
    source, optimizer = training_pair()
    advance(source, optimizer, 1)
    path = tmp_path / "combined.ckpt"
    ortho.save_checkpoint(path, model=source, optimizer=optimizer, metadata={"step": 1})
    destination, restored = training_pair(lr=0.5)
    untouched = value_tree(restored.state_dict())
    assert ortho.load_state_dict(destination, path) is None
    assert value_tree(destination.state_dict()) == value_tree(source.state_dict())
    assert value_tree(restored.state_dict()) == untouched


def test_nested_training_metadata_preserves_types_and_is_a_detached_snapshot(
    tensor, tmp_path, assert_tensor,
):
    p = tensor([2., 3.], requires_grad=True)
    tracked = p * 2
    metadata = {"step": 2**62 + 1, "flags": [True, 1, None],
                "schedule": ([0.125, -0.0], ("é", "中")),
                "statistics": {"positive": float("inf"), "negative": float("-inf"),
                               "missing": float("nan")}, "prediction": tracked}
    path = tmp_path / "metadata.ckpt"
    ortho.save_checkpoint(path, metadata=metadata)
    tracked.mul_(3.)
    loaded = ortho.load_checkpoint(path)
    assert loaded["step"] == 2**62 + 1
    assert loaded["flags"][0] is True
    assert type(loaded["flags"][1]) is int
    assert loaded["flags"][2] is None
    assert type(loaded["schedule"]) is tuple
    assert type(loaded["schedule"][0]) is list
    assert loaded["schedule"][1] == ("é", "中")
    assert math.copysign(1., loaded["schedule"][0][1]) == -1.
    assert loaded["statistics"]["positive"] == float("inf")
    assert loaded["statistics"]["negative"] == float("-inf")
    assert math.isnan(loaded["statistics"]["missing"])
    restored = loaded["prediction"]
    assert_tensor(restored, [4., 6.])
    assert restored.dtype == tracked.dtype
    assert restored.device.is_cpu()
    assert restored.is_leaf and not restored.requires_grad
    assert restored.grad is None
    restored.mul_(2.)
    assert_tensor(ortho.load_checkpoint(path)["prediction"], [4., 6.])


def test_same_ordered_combined_state_and_restored_state_produce_identical_archives(
    training_pair, tmp_path,
):
    source, optimizer = training_pair()
    advance(source, optimizer, 2)
    metadata = {"step": 2, "plan": (3, [0.125, 0.0625])}
    a, b, c = (tmp_path / name for name in ("a.ckpt", "other-name.data", "restored.ckpt"))
    ortho.save_checkpoint(a, model=source, optimizer=optimizer, metadata=metadata)
    ortho.save_checkpoint(b, model=source, optimizer=optimizer, metadata=metadata)
    restored, resumed_optimizer = training_pair(lr=0.5)
    saved_metadata = ortho.load_checkpoint(a, model=restored, optimizer=resumed_optimizer)
    ortho.save_checkpoint(c, model=restored, optimizer=resumed_optimizer, metadata=saved_metadata)
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()
    assert training_state(restored, resumed_optimizer) == training_state(source, optimizer)


def test_model_registration_order_does_not_change_state_file_bytes(tensor, tmp_path):
    from orthon.nn import Module, Parameter

    left, right = Module(), Module()
    left.a, left.z = Parameter(tensor([2.])), Parameter(tensor([3.]))
    right.z, right.a = Parameter(tensor([3.])), Parameter(tensor([2.]))
    assert list(left.state_dict()) != list(right.state_dict())
    a, b = tmp_path / "left.ckpt", tmp_path / "right.ckpt"
    ortho.save_state_dict(left, a)
    ortho.save_state_dict(right, b)
    assert a.read_bytes() == b.read_bytes()
