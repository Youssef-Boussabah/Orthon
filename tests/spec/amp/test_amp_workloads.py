"""FP32 AMP bypass across composed workloads, owners and checkpoint resume."""
import numpy as np
import pytest

import orthon as ortho
from orthon import nn
from orthon.optim import Adam, SGD
from orthon.amp import MixedPrecisionOptimizer
from ..nn.workload import TokenModel

CASES = [
    (SGD, 'mlp', [.04880497604608536, .04864173382520676, .048479288816452026]),
    (SGD, 'cnn', [2.104104518890381, 2.070575714111328, 2.0377871990203857]),
    (SGD, 'transformer', [3.8105976581573486, 3.055609941482544, 2.4907608032226562]),
    (Adam, 'mlp', [.04880497604608536, .047718483954668045, .04665311798453331]),
    (Adam, 'cnn', [2.104104518890381, 2.0800671577453613, 2.0562260150909424]),
    (Adam, 'transformer', [3.8105976581573486, 3.580655813217163, 3.360640525817871]),
]


@pytest.mark.parametrize('kind,model_kind,expected', CASES, ids=[f'{k.__name__}-{n}' for k, n, _ in CASES])
def test_fp32_amp_composed_workloads_and_checkpoint_continuation(
    typed_tensor, operator_device, tmp_path, kind, model_kind, expected,
):
    def compare_state(actual, expected):
        if isinstance(actual, ortho.Tensor):
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype and actual.device == expected.device
            if operator_device == 'cpu':
                assert actual.tolist() == expected.tolist()
            else:
                assert actual.tolist() == pytest.approx(expected.tolist(), rel=2e-5, abs=2e-6)
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                compare_state(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                compare_state(a, b)
        else:
            assert actual == expected

    def create():
        ortho.manual_seed(103)
        if model_kind == 'mlp':
            model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
            x = typed_tensor(np.arange(6).reshape(2, 3) / 10, device=operator_device)
        elif model_kind == 'cnn':
            # BatchNorm cancels a preceding convolution bias mathematically;
            # CUDA reduction rounding of that zero gradient is amplified by
            # Adam. Use the supported biasless layer for resume comparisons.
            model = nn.Sequential(nn.Conv2d(1, 2, 2, bias=False), nn.BatchNorm2d(2), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 2))
            x = typed_tensor(np.arange(72).reshape(2, 1, 6, 6) / 50, device=operator_device)
        else:
            model = TokenModel()
            x = typed_tensor([[1, 2, 3], [2, 3, 4]], dtype='int64', device=operator_device)
        model.to(operator_device)
        return model, MixedPrecisionOptimizer(model, kind, lr=.001), x

    def advance(model, opt, x):
        opt.zero_grad()
        output = model(x)
        loss = (output*output).mean()
        loss.backward()
        gradients = [p.grad.tolist() for p in model.parameters()]
        assert all(p.grad.dtype == p.dtype and p.grad.device == p.device for p in model.parameters())
        assert all(p is m for p, m in zip(model.parameters(), opt.state_dict()["masters"]))
        before = [p.tolist() for p in model.parameters()]
        opt.step()
        assert any(p.tolist() != values for p, values in zip(model.parameters(), before))
        assert [p.grad.tolist() for p in model.parameters()] == gradients
        return loss.item()

    source, opt, x = create()
    owners = list(source.parameters())
    losses = [advance(source, opt, x)]
    checkpoint = tmp_path / 'workload.ckpt'
    ortho.save_checkpoint(checkpoint, model=source, optimizer=opt)
    # Supported stable-owner movement with initialized history.
    source.to(operator_device)
    assert all(p is q for p, q in zip(source.parameters(), owners))
    losses.extend(advance(source, opt, x) for _ in range(2))
    assert losses == pytest.approx(expected, rel=2e-5, abs=2e-6)
    restored, resumed, y = create()
    held = list(restored.parameters())
    ortho.load_checkpoint(checkpoint, model=restored, optimizer=resumed, map_location=ortho.Device(operator_device))
    assert all(p is q for p, q in zip(restored.parameters(), held))
    continued = [advance(restored, resumed, y) for _ in range(2)]
    if operator_device == 'cpu':
        assert continued == losses[1:]
    else:
        assert continued == pytest.approx(losses[1:], rel=2e-5, abs=2e-6)
    compare_state(restored.state_dict(), source.state_dict())
    compare_state(resumed.state_dict(), opt.state_dict())
