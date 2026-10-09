"""Small deterministic workloads covering deterministic model compositions."""
import numpy as np
import pytest

import orthon as ortho
from orthon import nn
from orthon.optim import SGD
from .workload import TokenModel




@pytest.mark.parametrize('kind,expected', [
    ('mlp', [.04880497604608536, .04864173382520676, .048479288816452026]),
    ('cnn', [2.104104518890381, 2.070575714111328, 2.0377871990203857]),
    ('transformer', [3.8105976581573486, 3.055609941482544, 2.4907608032226562]),
])
def test_small_workloads_preserve_pre_migration_losses_state_and_training(typed_tensor, operator_device, kind, expected):
    ortho.manual_seed(103)
    if kind == 'mlp':
        model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
        x = typed_tensor(np.arange(6).reshape(2, 3) / 10, device=operator_device)
    elif kind == 'cnn':
        model = nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 2))
        x = typed_tensor(np.arange(72).reshape(2, 1, 6, 6) / 50, device=operator_device)
    else:
        model = TokenModel()
        x = typed_tensor([[1, 2, 3], [2, 3, 4]], dtype='int64', device=operator_device)
    owners = list(model.parameters())
    names = list(model.state_dict())
    model.to(operator_device)
    assert list(model.parameters()) == owners
    optimizer = SGD(model.parameters(), lr=.001)
    losses = []
    for _ in range(3):
        assert model.train() is model
        assert model.zero_grad() is None
        output = model(x)
        assert output.shape == (2, 2)
        loss = (output * output).mean()
        loss.backward()
        assert all(p.grad is not None for p in owners)
        losses.append(loss.item())
        optimizer.step()
    assert losses == pytest.approx(expected, rel=2e-5, abs=2e-6)
    assert list(model.state_dict()) == names
    assert list(model.parameters()) == owners
    assert model.eval() is model
    before = model(x).tolist()
    model.load_state_dict(model.state_dict())
    assert list(model.parameters()) == owners
    assert model(x).tolist() == pytest.approx(before, rel=2e-6, abs=2e-6)
    model.zero_grad()
    assert all(p.grad is None for p in owners)
