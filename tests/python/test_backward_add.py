import orthon as ortho
import pytest


def test_backward_add_left_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    c = a + b
    c.backward()

    assert a.grad.tolist() == pytest.approx([1.0] * 6)
    assert b.grad is None


def test_backward_add_right_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    b.requires_grad_(True)

    c = a + b
    c.backward()

    assert a.grad is None
    assert b.grad.tolist() == pytest.approx([1.0] * 6)


def test_backward_add_both_require_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    b.requires_grad_(True)

    c = a + b
    c.backward()

    assert a.grad.tolist() == pytest.approx([1.0] * 6)
    assert b.grad.tolist() == pytest.approx([1.0] * 6)


def test_backward_accumulates():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    c = a + b
    c.backward(retain_graph=True)
    c.backward()

    assert a.grad.tolist() == pytest.approx([2.0] * 6)


def test_backward_no_requires_grad_raises():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    c = a + b

    with pytest.raises(RuntimeError):
        c.backward()


# ---------------------------------------------------------------------------
# Leaf gradient ownership
#
# AddNode hands the SAME gradient object to both of its inputs, because both
# partial derivatives are the identity. The arithmetic is right, but .grad is
# public mutable state - DDP rescales it after all-reduce, AMP unscales it, and
# user code may too - so two distinct leaves must own two distinct buffers or
# `a.grad *= k` silently edits b.grad.
# ---------------------------------------------------------------------------

_CUDA = ortho.cuda_enabled()
_DEVICES = ["cpu"] + (["cuda"] if _CUDA else [])
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


def _leaves(device):
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    if device != "cpu":
        a, b = a.to(device), b.to(device)
    a.requires_grad_(True)
    b.requires_grad_(True)
    return a, b


@pytest.mark.parametrize("device", _DEVICES)
def test_add_backward_gives_each_leaf_its_own_grad(device):
    a, b = _leaves(device)
    (a + b).backward()

    assert a.grad.tolist() == pytest.approx([1.0] * 6)
    assert b.grad.tolist() == pytest.approx([1.0] * 6)
    assert a.grad._impl_id != b.grad._impl_id


@pytest.mark.parametrize("device", _DEVICES)
def test_mutating_one_leaf_grad_does_not_touch_the_other(device):
    # Both directions: a one-directional check still passes if the private copy
    # was handed to the wrong leaf.
    a, b = _leaves(device)
    (a + b).backward()

    a.grad.mul_(2.0)
    assert b.grad.tolist() == pytest.approx([1.0] * 6)

    b.grad.mul_(5.0)
    assert a.grad.tolist() == pytest.approx([2.0] * 6)


@pytest.mark.parametrize("device", _DEVICES)
def test_second_backward_accumulates_and_leaves_stay_independent(device):
    a, b = _leaves(device)
    (a + b).backward()
    (a + b).backward()

    assert a.grad.tolist() == pytest.approx([2.0] * 6)
    assert b.grad.tolist() == pytest.approx([2.0] * 6)
    assert a.grad._impl_id != b.grad._impl_id

    a.grad.mul_(3.0)
    assert b.grad.tolist() == pytest.approx([2.0] * 6)


@pytest.mark.parametrize("device", _DEVICES)
def test_broadcast_add_backward_gradients_unchanged(device):
    # The broadcast path reduces one side, so unbroadcast already allocated and
    # these leaves were never aliased. Pinned so the ownership fix cannot be
    # "achieved" by changing what a broadcast add computes.
    a = ortho.ones([2, 3])
    b = ortho.ones([3])
    if device != "cpu":
        a, b = a.to(device), b.to(device)
    a.requires_grad_(True)
    b.requires_grad_(True)

    (a + b).backward()

    assert a.grad.tolist() == pytest.approx([1.0] * 6)
    assert b.grad.tolist() == pytest.approx([2.0] * 3)
    assert a.grad._impl_id != b.grad._impl_id


@pytest.mark.parametrize("device", _DEVICES)
def test_same_leaf_on_both_sides_still_accumulates(device):
    # One leaf used twice is a different situation: there is only ONE leaf, so
    # its gradient must accumulate to 2 rather than be split into two owners.
    a = ortho.ones([2, 3])
    if device != "cpu":
        a = a.to(device)
    a.requires_grad_(True)

    (a + a).backward()

    assert a.grad.tolist() == pytest.approx([2.0] * 6)


def test_two_module_parameters_added_directly_train_independently():
    # The shipped failure mode: a module whose two same-shaped parameters are
    # added directly. With aliased gradients an SGD step moved both parameters
    # by the same in-place-edited buffer.
    from orthon.nn import Module, Parameter
    from orthon.optim import SGD

    class TwoParams(Module):
        def __init__(self):
            super().__init__()
            self.p = Parameter(ortho.ones([2, 2]))
            self.q = Parameter(ortho.full([2, 2], 3.0))

        def forward(self, x):
            return (self.p + self.q) * x

    model = TwoParams()
    optimizer = SGD(model.parameters(), lr=0.1)

    out = model(ortho.full([2, 2], 2.0))
    out.backward()

    assert model.p.grad._impl_id != model.q.grad._impl_id
    assert model.p.grad.tolist() == pytest.approx([2.0] * 4)
    assert model.q.grad.tolist() == pytest.approx([2.0] * 4)

    # Scaling one parameter's gradient must not move the other's step.
    model.p.grad.mul_(0.5)
    optimizer.step()

    assert model.p.tolist() == pytest.approx([1.0 - 0.1 * 1.0] * 4)
    assert model.q.tolist() == pytest.approx([3.0 - 0.1 * 2.0] * 4)
