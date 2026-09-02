"""End-to-end training tests.

Verifies that cross_entropy returns a Tensor with grad_fn so that
loss.backward() correctly drives SGD to reduce the loss over time.
"""
import pytest
import orthon
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import SGD


def test_cross_entropy_returns_tensor():
    logits = orthon.full([4, 10], 1.0)
    loss = orthon.cross_entropy(logits, [0, 1, 2, 3])
    assert isinstance(loss, orthon.Tensor)
    assert loss.numel == 1
    assert loss.item() > 0.0


def test_cross_entropy_requires_grad_when_logits_do():
    model = Linear(4, 3)
    x = orthon.full([2, 4], 0.5)
    logits = model(x)
    loss = orthon.cross_entropy(logits, [0, 1])
    assert loss.requires_grad
    assert loss._has_grad_fn


def test_cross_entropy_no_grad_when_logits_plain():
    logits = orthon.full([2, 3], 1.0)
    loss = orthon.cross_entropy(logits, [0, 1])
    assert not loss.requires_grad


def test_add_inplace_basic():
    a = orthon.full([3], 1.0)
    b = orthon.full([3], 2.0)
    a.add_(b, alpha=0.5)
    assert a.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_add_inplace_negative_alpha():
    a = orthon.full([4], 5.0)
    b = orthon.full([4], 1.0)
    a.add_(b, alpha=-1.0)
    assert a.tolist() == pytest.approx([4.0, 4.0, 4.0, 4.0])


def test_add_inplace_preserves_requires_grad():
    a = orthon.zeros([3])
    a.requires_grad_(True)
    b = orthon.full([3], 1.0)
    a.add_(b)
    assert a.requires_grad
    assert a.is_leaf


def test_sgd_uses_add_inplace():
    """SGD step must update weights without breaking leaf status."""
    model = Sequential(Linear(2, 3), Linear(3, 4))
    optimizer = SGD(model.parameters(), lr=0.1)
    x = orthon.full([2, 2], 1.0)
    targets = [0, 1]

    w_before = list(model.parameters())[0].tolist()[:]

    loss = orthon.cross_entropy(model(x), targets)
    loss.backward()
    optimizer.step()

    w_after = list(model.parameters())[0].tolist()
    assert w_after != w_before, "SGD did not update weights"

    for p in model.parameters():
        assert p.requires_grad
        assert p.is_leaf


def test_loss_decreases_over_training():
    """Full training loop: loss must strictly decrease over 30 steps."""
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )
    optimizer = SGD(model.parameters(), lr=0.01)

    batch_size = 16
    x = orthon.full([batch_size, 784], 0.1)
    targets = [i % 10 for i in range(batch_size)]

    losses = []
    for _ in range(30):
        model.zero_grad()
        logits = model(x)
        loss = orthon.cross_entropy(logits, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}\n"
        f"First 5: {[f'{v:.4f}' for v in losses[:5]]}"
    )


def test_gradient_shapes_via_ce_backward():
    """Gradients from cross_entropy backward have correct shapes."""
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )
    x = orthon.full([4, 784], 0.1)
    targets = [0, 1, 2, 3]

    loss = orthon.cross_entropy(model(x), targets)
    loss.backward()

    expected_shapes = [(784, 128), (128,), (128, 10), (10,)]
    for i, (p, exp) in enumerate(zip(model.parameters(), expected_shapes)):
        assert p.grad is not None, f"param {i} has no gradient"
        assert p.grad.shape == exp, f"param {i}: {p.grad.shape} != {exp}"


def test_zero_grad_after_step():
    model = Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=0.01)
    x = orthon.full([3, 4], 1.0)

    loss = orthon.cross_entropy(model(x), [0, 1, 0])
    loss.backward()
    optimizer.zero_grad()

    for p in model.parameters():
        assert p.grad is None
