from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest
import orthon as ortho

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _tensor(shape, values):
    t = ortho.zeros(shape)
    if len(shape) == 1:
        for i, value in enumerate(values):
            t[i] = float(value)
    elif len(shape) == 2:
        cols = shape[1]
        for i, value in enumerate(values):
            t[[i // cols, i % cols]] = float(value)
    else:
        raise ValueError("test helper only supports 1D/2D")
    return t


def _fixture():
    x = _tensor([3, 4], [
        -1.0, 0.5, 2.0, 1.0,
        0.25, -0.5, 1.5, 2.0,
        3.0, -1.0, 0.0, 0.75,
    ])
    weight = _tensor([4, 5], [
        0.5, -1.0, 0.25, 1.0, 0.75,
        -0.25, 0.5, 1.0, -0.5, 0.25,
        1.0, 0.25, -0.75, 0.5, -1.0,
        -0.5, 1.0, 0.5, 0.25, 0.5,
    ])
    bias = _tensor([5], [0.1, -0.2, 0.3, -0.4, 0.5])
    return x, weight, bias


def _separate(x, weight, bias):
    return ortho.relu(ortho.matmul(x, weight) + bias)


def test_fused_linear_relu_cpu_matches_separate_ops():
    x, weight, bias = _fixture()

    fused = ortho.fused_linear_relu(x, weight, bias)
    separate = _separate(x, weight, bias)

    assert fused.shape == separate.shape
    assert fused.tolist() == pytest.approx(separate.tolist(), rel=1e-6, abs=1e-6)


def test_fused_linear_relu_cpu_matches_torch_reference():
    x, weight, bias = _fixture()

    fused = ortho.fused_linear_relu(x, weight, bias)
    tx = torch.tensor(x.tolist(), dtype=torch.float32).reshape(3, 4)
    tw = torch.tensor(weight.tolist(), dtype=torch.float32).reshape(4, 5)
    tb = torch.tensor(bias.tolist(), dtype=torch.float32)
    expected = torch.relu(tx @ tw + tb).flatten().tolist()

    assert fused.tolist() == pytest.approx(expected, rel=1e-6, abs=1e-6)


@requires_cuda
def test_fused_linear_relu_cuda_matches_separate_ops():
    x, weight, bias = _fixture()
    x = x.to("cuda")
    weight = weight.to("cuda")
    bias = bias.to("cuda")

    fused = ortho.fused_linear_relu(x, weight, bias)
    separate = _separate(x, weight, bias)

    assert fused.device.is_cuda()
    assert fused.tolist() == pytest.approx(separate.tolist(), rel=1e-5, abs=1e-5)


@requires_cuda
def test_fused_linear_relu_cuda_matches_torch_reference():
    x, weight, bias = _fixture()

    fused = ortho.fused_linear_relu(x.to("cuda"), weight.to("cuda"), bias.to("cuda"))
    tx = torch.tensor(x.tolist(), dtype=torch.float32).reshape(3, 4)
    tw = torch.tensor(weight.tolist(), dtype=torch.float32).reshape(4, 5)
    tb = torch.tensor(bias.tolist(), dtype=torch.float32)
    expected = torch.relu(tx @ tw + tb).flatten().tolist()

    assert fused.tolist() == pytest.approx(expected, rel=1e-5, abs=1e-5)
