import pytest
import orthon as ortho


def test_reshape_preserves_requires_grad():
    x = ortho.full([2, 3], 1.0)
    x.requires_grad_(True)
    y = ortho.reshape(x, [6])
    assert y.requires_grad


def test_reshape_no_grad_when_input_no_grad():
    x = ortho.full([2, 3], 1.0)
    y = ortho.reshape(x, [6])
    assert not y.requires_grad
    assert not y._has_grad_fn


def test_reshape_backward_ones_grad():
    x = ortho.full([2, 3], 1.0)
    x.requires_grad_(True)
    y = ortho.reshape(x, [6])
    y.backward()
    assert list(x.grad.shape) == [2, 3]
    assert x.grad.tolist() == pytest.approx([1.0] * 6)


def test_reshape_backward_grad_shape():
    """Gradient has the same shape as the original input."""
    x = ortho.full([3, 4], 1.0)
    x.requires_grad_(True)
    ortho.reshape(x, [12]).backward()
    assert list(x.grad.shape) == [3, 4]


def test_reshape_4d_to_2d_backward():
    """NCHW → (N, C*H*W) flatten: gradient flows back to 4-D shape."""
    N, C, H, W = 2, 3, 4, 4
    x = ortho.full([N, C, H, W], 1.0)
    x.requires_grad_(True)
    ortho.reshape(x, [N, C * H * W]).backward()
    assert list(x.grad.shape) == [N, C, H, W]
    assert x.grad.tolist() == pytest.approx([1.0] * (N * C * H * W))


def test_flatten_then_add_autograd():
    """Flatten followed by element-wise add: gradient reaches the input."""
    x = ortho.full([2, 3], 2.0)
    x.requires_grad_(True)
    flat = ortho.reshape(x, [6])
    out = flat + ortho.full([6], 1.0)
    out.backward()
    assert x.grad.tolist() == pytest.approx([1.0] * 6)


def test_reshape_chain_two_reshapes():
    """Two consecutive reshapes: gradient flows through both."""
    x = ortho.full([2, 6], 1.0)
    x.requires_grad_(True)
    y = ortho.reshape(x, [4, 3])
    z = ortho.reshape(y, [12])
    z.backward()
    assert list(x.grad.shape) == [2, 6]
    assert x.grad.tolist() == pytest.approx([1.0] * 12)


def test_reshape_output_shape_unchanged():
    """reshape does not mutate the input's shape."""
    x = ortho.full([2, 3], 1.0)
    x.requires_grad_(True)
    y = ortho.reshape(x, [6])
    assert list(x.shape) == [2, 3]
    assert list(y.shape) == [6]


# CUDA.

def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


cuda_available = _cuda_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


@requires_cuda
def test_cuda_reshape_preserves_requires_grad():
    x = ortho.full([6], 1.0).to("cuda")
    x.requires_grad_(True)
    y = ortho.reshape(x, [2, 3])
    assert y.requires_grad


@requires_cuda
def test_cuda_reshape_backward():
    x = ortho.full([2, 3], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.reshape(x, [6]).backward()
    grad = x.grad.to("cpu")
    assert list(grad.shape) == [2, 3]
    assert grad.tolist() == pytest.approx([1.0] * 6)


@requires_cuda
def test_cuda_reshape_grad_lives_on_cuda():
    x = ortho.full([6], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.reshape(x, [2, 3]).backward()
    assert x.grad.device.is_cuda()


@requires_cuda
def test_cuda_reshape_backward_matches_cpu():
    x_cpu = ortho.full([2, 4], 1.0)
    x_cpu.requires_grad_(True)
    ortho.reshape(x_cpu, [8]).backward()

    x_cuda = ortho.full([2, 4], 1.0).to("cuda")
    x_cuda.requires_grad_(True)
    ortho.reshape(x_cuda, [8]).backward()

    assert x_cpu.grad.tolist() == pytest.approx(x_cuda.grad.to("cpu").tolist())
