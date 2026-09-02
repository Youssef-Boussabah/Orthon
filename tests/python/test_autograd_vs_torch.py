import pytest

torch = pytest.importorskip("torch")
import orthon as ortho


def assert_close(actual, expected):
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_add_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = a + b
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((2, 3), 3.0, requires_grad=True)
    tout = ta + tb
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())


def test_mul_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = a * b
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((2, 3), 3.0, requires_grad=True)
    tout = ta * tb
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())


def test_matmul_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3, 4], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = a @ b
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((3, 4), 3.0, requires_grad=True)
    tout = ta @ tb
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())


def test_relu_grad_matches_torch():
    x = ortho.full([2, 3], -1.0)
    x[0, 0] = 2.0
    x[1, 1] = 3.0
    x.requires_grad_(True)

    out = ortho.relu(x)
    out.backward()

    tx = torch.full((2, 3), -1.0, requires_grad=True)
    with torch.no_grad():
        tx[0, 0] = 2.0
        tx[1, 1] = 3.0

    tout = torch.relu(tx)
    tout.backward(torch.ones_like(tout))

    assert_close(x.grad.tolist(), tx.grad.flatten().tolist())


def test_chain_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    c = ortho.full([2, 3], 4.0)
    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)

    out = ortho.relu((a * b) + c)
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((2, 3), 3.0, requires_grad=True)
    tc = torch.full((2, 3), 4.0, requires_grad=True)
    tout = torch.relu((ta * tb) + tc)
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())
    assert_close(c.grad.tolist(), tc.grad.flatten().tolist())


def test_branching_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = (a * b) + (a + b)
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((2, 3), 3.0, requires_grad=True)
    tout = (ta * tb) + (ta + tb)
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())


def test_broadcast_add_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = a + b
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((3,), 3.0, requires_grad=True)
    tout = ta + tb
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())


def test_broadcast_mul_grad_matches_torch():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3], 3.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    out = a * b
    out.backward()

    ta = torch.full((2, 3), 2.0, requires_grad=True)
    tb = torch.full((3,), 3.0, requires_grad=True)
    tout = ta * tb
    tout.backward(torch.ones_like(tout))

    assert_close(a.grad.tolist(), ta.grad.flatten().tolist())
    assert_close(b.grad.tolist(), tb.grad.flatten().tolist())
