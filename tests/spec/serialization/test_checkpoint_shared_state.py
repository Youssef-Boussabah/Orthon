import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def test_tied_destination_restores_values_without_losing_its_existing_tie(
    tensor, tmp_path, assert_tensor, assert_grad,
):
    p = tensor([2., 3.])
    source = Module()
    shared = Parameter(p)
    source.a = shared
    source.branch = Module()
    source.branch.b = shared
    path = tmp_path / "tied.ckpt"
    ortho.save_state_dict(source, path)
    restored = Module()
    q = tensor([0., 0.])
    tied = Parameter(q)
    restored.a = tied
    restored.branch = Module()
    restored.branch.b = tied
    ortho.load_state_dict(restored, path)
    assert restored.a is restored.branch.b is q
    assert list(restored.parameters()) == [q]
    assert_tensor(q, [2., 3.])
    (restored.a + restored.branch.b).sum().backward()
    assert_grad(q, [2., 2.])
    SGD(restored.parameters(), lr=0.25).step()
    assert_tensor(q, [1.5, 2.5])
    assert_tensor(p, [2., 3.])


def test_source_ties_do_not_create_new_ownership_ties_in_untied_destination(
    tensor, tmp_path, assert_tensor,
):
    source = Module()
    shared = Parameter(tensor([2., 3.]))
    source.a = shared
    source.b = shared
    destination = Module()
    destination.a, destination.b = Parameter(tensor([0., 0.])), Parameter(tensor([0., 0.]))
    a, b = destination.a, destination.b
    path = tmp_path / "logical-values.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(destination, path)
    assert destination.a is a and destination.b is b
    a.mul_(2.)
    assert_tensor(a, [4., 6.])
    assert_tensor(b, [2., 3.])


def test_shared_metadata_references_and_views_restore_independent_value_snapshots(
    tensor, tmp_path, assert_tensor,
):
    base = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    path = tmp_path / "aliases.ckpt"
    ortho.save_checkpoint(path, metadata={"same": (base, base), "view": base.transpose(0, 1)})
    loaded = ortho.load_checkpoint(path)
    a, b = loaded["same"]
    a.mul_(2.)
    assert_tensor(b, [[1., 2., 3.], [4., 5., 6.]])
    assert_tensor(loaded["view"], [[1., 4.], [2., 5.], [3., 6.]])
    assert loaded["view"].is_contiguous
    assert all(value.is_leaf and not value.requires_grad for value in (a, b, loaded["view"]))
    assert_tensor(base, [[1., 2., 3.], [4., 5., 6.]])


@pytest.mark.parametrize("view_kind", ["transpose", "offset-slice"])
def test_strided_source_state_loads_logical_values_into_contiguous_destination(
    tensor, tmp_path, assert_tensor, view_kind,
):
    base = tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    if view_kind == "transpose":
        view = base.transpose(0, 1)
        expected = [[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]
    else:
        view = base.slice(1, 1, 3)
        expected = [[2., 3.], [5., 6.], [8., 9.]]
    source, target = Module(), Module()
    source.weight = Parameter(view)
    target.weight = Parameter(tensor([[0.] * len(expected[0]) for _ in expected]))
    held = target.weight
    path = tmp_path / "strided.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(target, path)
    assert target.weight is held
    assert target.weight.is_contiguous
    assert_tensor(target.weight, expected)
    base.mul_(2.)
    assert_tensor(target.weight, expected)


def test_strided_buffer_destination_can_be_replaced_by_loaded_independent_values(
    tensor, tmp_path, assert_tensor,
):
    source, target = Module(), Module()
    source.register_buffer("buffer", tensor([[1., 2.], [3., 4.]]))
    original = tensor([[5., 6.], [7., 8.]]).transpose(0, 1)
    target.register_buffer("buffer", original)
    path = tmp_path / "buffer.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(target, path)
    assert target.buffer is not original
    assert target.buffer.is_contiguous
    assert_tensor(target.buffer, [[1., 2.], [3., 4.]])
    assert_tensor(original, [[5., 7.], [6., 8.]])


@pytest.mark.parametrize("kind", [SGD, Adam])
def test_equivalent_tied_model_and_optimizer_restore_can_continue_training(
    typed_tensor, operator_device, tmp_path, kind,
):
    def create():
        model = Module()
        shared = Parameter(typed_tensor([2., 3.], device=operator_device))
        model.a = shared
        model.branch = Module()
        model.branch.b = shared
        options = {"betas": (0.5, 0.75), "eps": 0.25} if kind is Adam else {}
        optimizer = kind(model.parameters(), lr=0.125, weight_decay=0.25, decay_bias=True, **options)
        return model, optimizer

    def step(model, optimizer):
        optimizer.zero_grad()
        loss = (model.a + model.branch.b).sum()
        loss.backward()
        optimizer.step()
        return loss.item()

    from .workload import value_tree

    source, optimizer = create()
    step(source, optimizer)
    path = tmp_path / "tied-training.ckpt"
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    expected = [step(source, optimizer) for _ in range(2)]
    restored, resumed = create()
    held = restored.a
    ortho.load_checkpoint(path, model=restored, optimizer=resumed)
    assert restored.a is restored.branch.b is held
    assert [step(restored, resumed) for _ in range(2)] == expected
    assert value_tree(restored.state_dict()) == value_tree(source.state_dict())
    assert value_tree(resumed.state_dict()) == value_tree(optimizer.state_dict())


def test_cpu_strided_metadata_snapshot_supports_every_storage_dtype(
    typed_tensor, storage_dtype, tmp_path,
):
    import numpy as np

    base = typed_tensor(np.arange(12).reshape(3, 4), dtype=storage_dtype)
    view = base.transpose(0, 1).slice(0, 1, 3)
    expected = view.tolist()
    path = tmp_path / "all-strided.ckpt"
    ortho.save_checkpoint(path, metadata={"view": view})
    restored = ortho.load_checkpoint(path)["view"]
    assert restored.dtype == storage_dtype
    assert restored.shape == (2, 3)
    assert restored.is_contiguous
    assert restored.tolist() == expected
