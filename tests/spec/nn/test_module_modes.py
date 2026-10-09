import numpy as np
import pytest

from orthon.nn import BatchNorm1d, BatchNorm2d, Dropout, Module, Parameter


def test_modes_propagate_recursively_are_repeatable_and_preserve_state_identity(module_factory):
    model = module_factory()
    model.branch.bias.requires_grad_(False)
    model.weight.sum().backward()
    model.branch.deep = Module()
    entries = model.state_dict()
    assert model.training is model.branch.training is model.branch.deep.training is True
    for mode in ["eval", "eval", "train", "train"]:
        assert getattr(model, mode)() is model
        expected = mode == "train"
        assert model.training is expected
        assert model.branch.training is expected
        assert model.branch.deep.training is expected
        assert all(model.state_dict()[name] is value for name, value in entries.items())
        assert model.weight.requires_grad is True
        assert model.branch.bias.requires_grad is False
        assert model.weight.grad.tolist() == [1, 1, 1, 1]


def test_eval_does_not_disable_autograd_or_freeze_parameters(tensor, assert_grad):
    class Weighted(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(tensor([2, 3]))

        def forward(self, x):
            return x * self.weight

    model = Weighted().eval()
    output = model(tensor([4, 5]))
    assert model.weight.requires_grad is True
    assert output.requires_grad is True
    output.sum().backward()
    assert_grad(model.weight, [4, 5])


@pytest.mark.parametrize("p", [0.0, .5])
def test_dropout_modes_preserve_input_or_apply_inverted_masks(tensor, assert_tensor, assert_grad, p):
    values = np.arange(1, 33, dtype=np.float64).reshape(4, 8)
    source = tensor(values, requires_grad=True)
    root = Module()
    root.drop = Dropout(p, seed=19)
    root.eval()
    assert root.drop(source) is source
    root.train()
    if p != 0 and source.dtype != "float32":
        with pytest.raises(RuntimeError):
            root.drop(source)
        root.eval()
        assert root.drop(source) is source
        assert_tensor(source, values)
        return
    output = root.drop(source)
    if p == 0:
        assert output is source
    else:
        observed = np.array(output.tolist()).reshape(values.shape)
        assert np.all((observed == 0) | (observed == values * 2))
        output.sum().backward()
        assert_grad(source, observed / values)
    root.eval()
    assert root.drop(source) is source
    assert_tensor(source, values)


@pytest.mark.parametrize("normalizer", [BatchNorm1d, BatchNorm2d])
def test_batchnorm_training_updates_state_and_eval_uses_it_without_updating(
    tensor, tensor_case, assert_tensor, normalizer,
):
    device, dtype = tensor_case
    root = Module()
    root.norm = normalizer(2, eps=.25, momentum=.5).cast(dtype).to(device)
    values = np.array([[1, 2], [5, 10]], dtype=np.float64)
    train_shape = (2, 2) if normalizer is BatchNorm1d else (2, 2, 1, 1)
    eval_shape = (1, 2) if normalizer is BatchNorm1d else (1, 2, 1, 1)
    output = root.norm(tensor(values.reshape(train_shape)))
    assert_tensor(output, ((values - [3, 6]) / np.sqrt([4.25, 16.25])).reshape(train_shape))
    assert_tensor(root.norm.running_mean, [1.5, 3])
    assert_tensor(root.norm.running_var, [4.5, 16.5])
    assert root.norm.running_mean.requires_grad is False
    assert root.norm.running_var.requires_grad is False
    root.eval()
    for _ in range(2):
        expected = np.array([[1.5 / np.sqrt(4.75), 3 / np.sqrt(16.75)]]).reshape(eval_shape)
        assert_tensor(root.norm(tensor(np.array([[3, 6]]).reshape(eval_shape))), expected)
        assert_tensor(root.norm.running_mean, [1.5, 3])
        assert_tensor(root.norm.running_var, [4.5, 16.5])
    root.train()
    with pytest.raises(ValueError):
        root.norm(tensor(np.array([[3, 6]]).reshape(eval_shape)))
    assert [name for name, _ in root.named_parameters()] == ["norm.weight", "norm.bias"]
    assert [name for name, _ in root.named_buffers()] == ["norm.running_mean", "norm.running_var"]
