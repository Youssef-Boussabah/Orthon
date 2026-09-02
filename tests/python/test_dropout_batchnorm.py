import numpy as np
import pytest

import orthon
from orthon import nn


def _tensor(values, shape, requires_grad=False):
    t = orthon.from_numpy(np.array(values, dtype=np.float32).reshape(shape))
    if requires_grad:
        t.requires_grad_(True)
    return t


def test_dropout_eval_is_passthrough():
    x = _tensor([1, 2, 3, 4], (2, 2), requires_grad=True)
    drop = nn.Dropout(0.75, seed=0).eval()
    y = drop(x)
    assert y is x


def test_dropout_training_masks_and_backpropagates():
    x = orthon.ones([1000])
    x.requires_grad_(True)
    drop = nn.Dropout(0.25, seed=123)
    y = drop(x)
    values = y.tolist()
    dropped = sum(1 for v in values if v == 0.0)
    assert 180 <= dropped <= 320
    assert max(values) == pytest.approx(1.0 / 0.75)
    y.backward()
    assert x.grad.tolist() == pytest.approx(values)


def test_batchnorm1d_forward_and_running_stats():
    x = _tensor([[1, 2, 3], [4, 5, 6]], (2, 3), requires_grad=True)
    bn = nn.BatchNorm1d(3)
    y = bn(x)
    expected = ((np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32) - [2.5, 3.5, 4.5])
                / np.sqrt(np.array([2.25, 2.25, 2.25], dtype=np.float32) + np.float32(1e-5)))
    assert y.tolist() == pytest.approx(expected.flatten().tolist(), abs=1e-5)
    assert bn.running_mean.tolist() == pytest.approx([0.25, 0.35, 0.45], abs=1e-6)

    y.backward()
    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None


def test_batchnorm2d_eval_uses_running_stats():
    x = _tensor([1, 2, 3, 4], (1, 2, 1, 2))
    bn = nn.BatchNorm2d(2)
    # Running statistics are registered Tensor buffers, so they are assigned as
    # Tensors; assigning a raw ndarray is rejected by the buffer contract.
    bn.running_mean = orthon.from_numpy(np.array([1.0, 3.0], dtype=np.float32))
    bn.running_var = orthon.from_numpy(np.array([4.0, 9.0], dtype=np.float32))
    bn.eval()
    y = bn(x)
    expected = (np.array([[[[1, 2]], [[3, 4]]]], dtype=np.float32)
                - np.array([1.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1))
    expected = expected / np.sqrt(np.array([4.0, 9.0], dtype=np.float32).reshape(1, 2, 1, 1) + 1e-5)
    assert y.tolist() == pytest.approx(expected.flatten().tolist(), abs=1e-5)
