"""Literal public AMP workloads and independent state snapshots."""

import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter


class AffineModel(Module):
    def __init__(self, create):
        super().__init__()
        self.weight = Parameter(create([[1., 2.], [3., 4.]]))
        self.bias = Parameter(create([.5, -.5]))
        self.unused = Parameter(create([6., 8.]))

    def forward(self, inputs):
        return inputs @ self.weight + self.bias


def snapshot(value):
    if isinstance(value, ortho.Tensor):
        with ortho.no_grad():
            return value.to(value.device)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return value


def assert_state(actual, expected):
    if isinstance(expected, ortho.Tensor):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype and actual.device == expected.device
        tolerance = {"float32": 2e-6, "float16": 5e-3, "bfloat16": 4e-2}[actual.dtype]
        assert actual.tolist() == pytest.approx(expected.tolist(), rel=tolerance, abs=tolerance)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state(left, right)
    else:
        assert actual == expected


def train(model, optimizer, scaler, inputs):
    optimizer.zero_grad()
    output = model(inputs)
    loss = (output * output).mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def numbers(value):
    return np.asarray(value.tolist()).reshape(value.shape)


def pair(create, dtype="float16", device="cpu"):
    model = Module()
    model.a = Parameter(create([1., 2.], dtype=dtype, device=device))
    model.b = Parameter(create([3., 4.], dtype=dtype, device=device))
    return model

def write_master(master, value):
    # CUDA fill_ remains deliberately unsupported; use existing FP32 arithmetic.
    with ortho.no_grad():
        master.mul_(0.)
        master.add_(ortho.full(list(master.shape), value).to(master.device))

def gradient(p, value=1.):
    p.set_grad(ortho.full(list(p.shape), value, dtype=p.dtype).to(p.device))

def observation(model, opt):
    state = opt.state_dict()
    owners = list(model.parameters())
    masters = state["masters"]
    return (snapshot(model.state_dict()), snapshot(state), owners, masters,
            [None if p.grad is None else (p.grad._impl_id, snapshot(p.grad)) for p in owners + masters],
            [p._storage_version for p in owners + masters],
            [v for name in ("m", "v") for v in state["inner"].get(name, [])])

def unchanged(model, opt, before):
    values, state, owners, masters, grads, versions, moments = before
    assert_state(model.state_dict(), values)
    assert_state(opt.state_dict(), state)
    assert all(a is b for a, b in zip(model.parameters(), owners))
    assert all(a is b for a, b in zip(opt.state_dict()["masters"], masters))
    for owner, grad in zip(owners + masters, grads):
        if grad is None:
            assert owner.grad is None
        else:
            assert owner.grad._impl_id == grad[0]
            assert owner.grad.tolist() == grad[1].tolist()
    assert [p._storage_version for p in owners + masters] == versions
    current = opt.state_dict()["inner"]
    assert all(a is b for a, b in zip(
        [v for name in ("m", "v") for v in current.get(name, [])], moments))
