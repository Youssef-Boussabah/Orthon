import numpy as np
import pytest

import orthon as ortho

from .graph_helpers import gradient_check


PROGRAMS = [
    pytest.param(lambda x, y: ((x * y + x) / (y + 2)).mean(),
                 [[.5, 1., 2.], [1., 2., 3.]], id="multiply-divide"),
    pytest.param(lambda x: (ortho.sigmoid(x * x + 1) * x).sum(),
                 [[-.5, .25, .75]], id="sigmoid-product"),
    pytest.param(lambda x: (ortho.tanh(x) * ortho.exp(x / 2)).mean(),
                 [[-.5, .25, .75]], id="tanh-exp"),
    pytest.param(lambda x: (ortho.gelu(x) * ortho.gelu(x) + x).mean(),
                 [[-.5, .25, .75]], id="exact-gelu"),
    pytest.param(lambda x: (ortho.sqrt(x + 2) * ortho.reciprocal(x + 3)).sum(),
                 [[.5, 1., 2.]], id="sqrt-reciprocal"),
    pytest.param(lambda x: ((ortho.relu(x) + ortho.leaky_relu(x, .125)) * x).mean(),
                 [[-.5, .25, .75]], id="piecewise-away-from-kinks"),
    pytest.param(lambda a, b: ((a @ b) * (a @ b)).mean(),
                 [[[.5, 1.], [-.5, .25]], [[1., .5], [.25, -.5]]], id="two-matmul-consumers"),
    pytest.param(lambda a, b: ((a + b) * (a + b)).mean(),
                 [[[.5], [1.]], [[.25, .5, .75]]], id="multi-axis-broadcast"),
    pytest.param(lambda x: ((x.transpose(0, 1).slice(0, 1, 3) * x.transpose(0, 1).slice(0, 1, 3)) + 1).mean(),
                 [[[.5, 1., 1.5], [.25, .75, 1.25]]], id="overlapping-view-paths"),
    pytest.param(lambda x: (x.mean(dim=0) * x.mean(dim=0)).sum(),
                 [[[.5, 1.], [.25, .75]]], id="shared-axis-reductions"),
]


@pytest.mark.parametrize("function,arrays", PROGRAMS)
def test_smooth_composed_graph_matches_central_finite_differences(function, arrays, tensor_case):
    device, dtype = tensor_case
    leaves, numerical = gradient_check(function, arrays, dtype=dtype, device=device)
    relative, absolute = (2e-7, 2e-8) if dtype == "float64" else (4e-3, 4e-4)
    for leaf, expected in zip(leaves, numerical):
        assert leaf.grad.shape == expected.shape
        assert leaf.grad.dtype == dtype and leaf.grad.device == leaf.device
        assert leaf.grad.requires_grad is False and leaf.grad.is_leaf is True
        assert np.asarray(leaf.grad.tolist()).reshape(expected.shape) == pytest.approx(
            expected, rel=relative, abs=absolute)


def test_softmax_loss_graph_matches_finite_differences_in_its_supported_profile(operator_device):
    weights = ortho.from_numpy(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)).to(operator_device)
    function = lambda x: (ortho.softmax(x * 2) * weights).mean()
    leaves, numerical = gradient_check(function, [[[.25, -.5, .75], [.5, -.25, 1.]]],
                                       dtype="float32", device=operator_device)
    assert leaves[0].grad.tolist() == pytest.approx(numerical[0].reshape(-1).tolist(), rel=4e-3, abs=4e-4)
