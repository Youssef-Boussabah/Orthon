"""Module API tests.

cross_entropy() returns a scalar Tensor (shape {1}) with a grad_fn, so
loss.backward() correctly propagates dL/dlogits through the full graph.
"""
import pytest
import orthon
from orthon.nn import Linear, Module, Parameter, ReLU, Sequential
from orthon.optim import SGD


# Module and Parameter basics.

def test_parameter_sets_requires_grad():
    t = orthon.zeros([4])
    assert not t.requires_grad
    p = Parameter(t)
    assert p._data.requires_grad


def test_parameter_rejects_non_tensor():
    with pytest.raises(TypeError):
        Parameter(3.14)


def test_module_registers_parameter():
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.w = Parameter(orthon.zeros([3]))

        def forward(self, x):
            return x

    net = Net()
    assert 'w' in net._parameters
    params = list(net.parameters())
    assert len(params) == 1
    assert params[0].requires_grad


def test_module_registers_submodule():
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(2, 3)

        def forward(self, x):
            return self.layer(x)

    net = Net()
    assert 'layer' in net._modules
    assert len(list(net.parameters())) == 2  # weight + bias


def test_train_sets_training_true_recursively():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    model.eval()
    model.train()

    assert model.training is True
    for child in model._modules.values():
        assert child.training is True


def test_eval_sets_training_false_recursively():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    model.eval()

    assert model.training is False
    for child in model._modules.values():
        assert child.training is False


def test_train_eval_return_self():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    assert model.train() is model
    assert model.eval() is model


def test_named_parameters_linear_names():
    layer = Linear(4, 2)
    assert [name for name, _ in layer.named_parameters()] == ['weight', 'bias']


def test_named_parameters_sequential_names():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    assert [name for name, _ in model.named_parameters()] == [
        '0.weight',
        '0.bias',
        '2.weight',
        '2.bias',
    ]


def test_named_parameters_match_parameters_and_state_dict():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    named = list(model.named_parameters())
    params = list(model.parameters())
    state = model.state_dict()

    assert [p for _, p in named] == params
    assert [name for name, _ in named] == list(state.keys())
    for name, tensor in named:
        assert tensor is state[name]


# Linear layer.

def test_linear_output_shape():
    layer = Linear(4, 2)
    x = orthon.full([3, 4], 1.0)
    out = layer(x)
    assert out.shape == (3, 2)


def test_linear_param_shapes():
    layer = Linear(4, 6)
    params = list(layer.parameters())
    assert len(params) == 2
    assert params[0].shape == (4, 6)  # weight
    assert params[1].shape == (6,)    # bias


def test_linear_params_require_grad():
    layer = Linear(4, 6)
    for p in layer.parameters():
        assert p.requires_grad


def test_linear_bias_initialised_to_zero():
    layer = Linear(4, 6)
    bias = list(layer.parameters())[1]
    assert bias.tolist() == pytest.approx([0.0] * 6)


# ReLU.

def test_relu_positive_passthrough():
    layer = ReLU()
    x = orthon.full([2, 3], 1.0)
    out = layer(x)
    assert out.tolist() == pytest.approx([1.0] * 6)


def test_relu_kills_negatives():
    layer = ReLU()
    x = orthon.full([2, 3], -1.0)
    out = layer(x)
    assert out.tolist() == pytest.approx([0.0] * 6)


# Sequential.

def test_sequential_output_shape():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = orthon.full([3, 4], 1.0)
    assert model(x).shape == (3, 2)


def test_sequential_parameter_count():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    assert len(list(model.parameters())) == 4  # w1, b1, w2, b2


def test_sequential_parameter_shapes():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    params = list(model.parameters())
    assert params[0].shape == (4, 8)
    assert params[1].shape == (8,)
    assert params[2].shape == (8, 2)
    assert params[3].shape == (2,)


def test_sequential_no_duplicate_params():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    params = list(model.parameters())
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids))


# Autograd through the module stack.

def test_gradients_flow_to_all_params():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = orthon.full([3, 4], 1.0)
    model(x).backward()

    params = list(model.parameters())
    for i, p in enumerate(params):
        assert p.grad is not None, f"param {i} has no gradient"
        assert p.grad.shape == p.shape, f"param {i} grad shape mismatch"


def test_zero_grad_clears_all_gradients():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = orthon.full([3, 4], 1.0)
    model(x).backward()
    model.zero_grad()

    for p in model.parameters():
        assert p.grad is None


def test_output_requires_grad():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = orthon.full([3, 4], 1.0)
    out = model(x)
    assert out.requires_grad
    assert out._has_grad_fn


# SGD optimizer.

def test_sgd_step_changes_weights():
    # Two Linear layers without ReLU so gradients are always non-zero.
    model = Sequential(Linear(2, 3), Linear(3, 1))
    x = orthon.full([2, 2], 1.0)
    optimizer = SGD(model.parameters(), lr=0.1)

    w = list(model.parameters())[0]
    before = w.tolist()[:]

    model(x).backward()
    optimizer.step()

    assert w.tolist() != before, "SGD did not update weights"


def test_sgd_zero_grad():
    model = Sequential(Linear(2, 3), Linear(3, 1))
    x = orthon.full([2, 2], 1.0)
    optimizer = SGD(model.parameters(), lr=0.1)

    model(x).backward()
    optimizer.zero_grad()

    for p in model.parameters():
        assert p.grad is None


def test_sgd_params_remain_leaf_after_step():
    model = Sequential(Linear(2, 3), Linear(3, 1))
    x = orthon.full([2, 2], 1.0)
    optimizer = SGD(model.parameters(), lr=0.1)

    model(x).backward()
    optimizer.step()

    for p in model.parameters():
        assert p.requires_grad
        assert p.is_leaf


# Full pipeline.

def test_full_pipeline():
    """Forward, CE loss, backward, gradient shape verification."""
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )

    batch_size = 4
    x = orthon.full([batch_size, 784], 0.1)

    # Forward
    logits = model(x)
    assert logits.shape == (batch_size, 10)
    assert logits.requires_grad

    # Cross-entropy loss returns a Tensor with grad_fn.
    targets = [i % 10 for i in range(batch_size)]
    ce_loss = orthon.cross_entropy(logits, targets)
    assert isinstance(ce_loss, orthon.Tensor)
    assert ce_loss.numel == 1
    assert ce_loss.item() > 0.0

    # Backward through the full graph via the loss tensor
    ce_loss.backward()

    # Verify gradient existence and shapes for all four parameters
    params = list(model.parameters())
    expected_shapes = [(784, 128), (128,), (128, 10), (10,)]
    assert len(params) == len(expected_shapes)

    for i, (p, exp) in enumerate(zip(params, expected_shapes)):
        assert p.grad is not None, f"param {i} has no gradient"
        assert p.grad.shape == exp, (
            f"param {i}: grad shape {p.grad.shape} != expected {exp}"
        )
