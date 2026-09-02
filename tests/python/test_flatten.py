import pytest

import orthon as ortho
import orthon.functional as F
from orthon import nn


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def test_flatten_top_level_shape_default():
    x = ortho.ones([2, 3, 4])
    y = ortho.flatten(x)
    assert list(y.shape) == [24]


def test_flatten_start_end_dims():
    x = ortho.ones([2, 3, 4, 5])
    y = ortho.flatten(x, start_dim=1, end_dim=2)
    assert list(y.shape) == [2, 12, 5]


def test_flatten_module_defaults_keep_batch_dim():
    x = ortho.ones([2, 3, 4])
    y = nn.Flatten()(x)
    assert list(y.shape) == [2, 12]


def test_flatten_functional_matches_tensor_method():
    x = ortho.ones([2, 3, 4])
    assert F.flatten(x, 1).tolist() == pytest.approx(x.flatten(1).tolist())


def test_flatten_backward_uses_reshape_autograd():
    x = ortho.ones([2, 3, 4])
    x.requires_grad_(True)
    y = nn.Flatten()(x)
    y.backward()
    assert list(x.grad.shape) == [2, 3, 4]
    assert x.grad.tolist() == pytest.approx([1.0] * 24)


@requires_cuda
def test_flatten_cuda_backward_and_device():
    x = ortho.ones([2, 3, 4]).to("cuda")
    x.requires_grad_(True)
    y = nn.Flatten()(x)
    assert y.device.is_cuda()
    assert list(y.shape) == [2, 12]
    y.backward()
    assert x.grad.device.is_cuda()
    assert x.grad.to("cpu").tolist() == pytest.approx([1.0] * 24)


def test_flatten_invalid_dims_raise():
    x = ortho.ones([2, 3])
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        x.flatten(2)
