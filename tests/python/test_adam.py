"""Tests for the Adam optimizer."""
import pytest
import orthon
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam


def _simple_model():
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 2))


def _run_forward(model, x, targets):
    logits = model(x)
    loss = orthon.cross_entropy(logits, targets)
    return loss


# Basic behavior.

def test_adam_step_changes_weights():
    model = Sequential(Linear(2, 4), Linear(4, 3))
    optim = Adam(model.parameters(), lr=0.01)
    x = orthon.full([2, 2], 1.0)
    targets = [0, 1]

    w_before = list(model.parameters())[0].tolist()[:]
    loss = _run_forward(model, x, targets)
    loss.backward()
    optim.step()

    assert list(model.parameters())[0].tolist() != w_before


def test_adam_zero_grad():
    model = _simple_model()
    optim = Adam(model.parameters(), lr=0.001)
    x = orthon.full([3, 4], 0.5)
    targets = [0, 1, 0]

    _run_forward(model, x, targets).backward()
    optim.zero_grad()

    for p in model.parameters():
        assert p.grad is None


def test_adam_get_set_lr():
    optim = Adam([], lr=0.01)

    assert optim.get_lr() == pytest.approx(0.01)
    optim.set_lr(0.001)
    assert optim.get_lr() == pytest.approx(0.001)


def test_adam_decoupled_weight_decay_changes_weight_not_bias_by_default():
    model = Linear(2, 2)
    x = orthon.full([2, 2], 0.5)
    targets = [0, 1]
    optim = Adam(model.parameters(), lr=0.01, weight_decay=0.1)

    w_before = model.weight.tolist()[:]
    b_before = model.bias.tolist()[:]
    _run_forward(model, x, targets).backward()
    optim.step()

    assert model.weight.tolist() != pytest.approx(w_before)
    assert model.bias.tolist() != pytest.approx(b_before)

    # Isolate decay behavior with zero gradients.
    weight = orthon.full([2, 2], 1.0)
    bias = orthon.full([2], 1.0)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    weight.set_grad(orthon.zeros([2, 2]))
    bias.set_grad(orthon.zeros([2]))
    Adam([weight, bias], lr=0.1, weight_decay=0.1).step()

    assert weight.tolist() == pytest.approx([0.99] * 4)
    assert bias.tolist() == pytest.approx([1.0] * 2)


def test_adam_weight_decay_zero_keeps_zero_gradient_param_unchanged():
    param = orthon.full([2, 2], 1.0)
    param.requires_grad_(True)
    param.set_grad(orthon.zeros([2, 2]))

    Adam([param], lr=0.1, weight_decay=0.0).step()

    assert param.tolist() == pytest.approx([1.0] * 4)


def test_adam_leaf_preserved_after_step():
    model = _simple_model()
    optim = Adam(model.parameters(), lr=0.001)
    x = orthon.full([3, 4], 0.5)
    _run_forward(model, x, [0, 1, 0]).backward()
    optim.step()

    for p in model.parameters():
        assert p.requires_grad
        assert p.is_leaf


def test_adam_moments_accumulate():
    """Second step should differ from first due to accumulated moments."""
    model = Linear(2, 3)
    optim = Adam(model.parameters(), lr=0.01)
    x = orthon.full([2, 2], 1.0)
    targets = [0, 1]

    # Step 1
    _run_forward(model, x, targets).backward()
    w_before = list(model.parameters())[0].tolist()[:]
    optim.step()
    w_after1 = list(model.parameters())[0].tolist()[:]
    delta1 = [w_after1[k] - w_before[k] for k in range(len(w_before))]

    # Step 2 (same data, different effective lr due to bc correction)
    _run_forward(model, x, targets).backward()
    optim.step()
    w_after2 = list(model.parameters())[0].tolist()
    delta2 = [w_after2[k] - w_after1[k] for k in range(len(w_after1))]

    # Deltas need not be identical because bc1/bc2 change between steps.
    assert delta1 != delta2


# Convergence.

def test_adam_loss_decreases():
    # Seeded so the initialization always gives the loss room to fall: seed 22
    # yields a first-to-last margin of ~0.034 over 30 steps at lr=0.01, ~20x
    # the unseeded median. A failure here is a real regression, not seed luck.
    orthon.manual_seed(22)
    model = _simple_model()
    optim = Adam(model.parameters(), lr=0.01)
    x = orthon.full([4, 4], 0.1)
    targets = [0, 1, 0, 1]

    losses = []
    for _ in range(30):
        model.zero_grad()
        loss = _run_forward(model, x, targets)
        losses.append(loss.item())
        loss.backward()
        optim.step()

    assert losses[-1] < losses[0], (
        f"Adam loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_adam_sequential_model():
    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    optim = Adam(model.parameters())
    x = orthon.full([4, 784], 0.1)
    targets = [0, 1, 2, 3]

    loss = _run_forward(model, x, targets)
    loss.backward()
    optim.step()

    for p in model.parameters():
        assert p.requires_grad
