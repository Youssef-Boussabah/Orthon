"""Deterministic regression workload shared with fresh-process resume checks."""

import numpy as np

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def tensor(values, dtype="float32", device="cpu"):
    value = ortho.from_numpy(np.array(values, dtype=dtype, order="C"))
    return value if device == "cpu" else value.to(device)


class RegressionModel(Module):
    def __init__(self, dtype="float32", device="cpu"):
        super().__init__()
        self.weight = Parameter(tensor([[0.5], [-0.25]], dtype, device))
        self.spare = Parameter(tensor([5.], dtype, device))
        self.head = Module()
        self.head.bias = Parameter(tensor([0.125], dtype, device))
        self.register_buffer("offset", tensor([0.25], dtype, device))
        self.register_buffer("cache", tensor([42.], dtype, device), persistent=False)
        self.ordinary = tensor([99.], dtype, device)

    def forward(self, x):
        return x @ self.weight + self.head.bias + self.offset


def make_optimizer(model, name, *, lr=0.125):
    options = {"weight_decay": 0.125, "decay_bias": False}
    if name == "Adam":
        options.update(betas=(0.5, 0.75), eps=0.25)
    return {"SGD": SGD, "Adam": Adam}[name](model.parameters(), lr=lr, **options)


def advance(model, optimizer, steps):
    device = "cpu" if model.weight.device.is_cpu() else "cuda:0"
    x = tensor([[1., 0.5], [-0.5, 1.]], model.weight.dtype, device)
    target = tensor([[1.], [-0.25]], model.weight.dtype, device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        residual = model(x) - target
        loss = (residual * residual).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def value_tree(value):
    if isinstance(value, ortho.Tensor):
        return {"values": value.tolist(), "shape": list(value.shape),
                "dtype": value.dtype, "device": str(value.device)}
    if isinstance(value, dict):
        return {key: value_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [value_tree(item) for item in value]
    return value


def training_state(model, optimizer):
    return {"model": value_tree(model.state_dict()),
            "optimizer": value_tree(optimizer.state_dict())}


class StateSource:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state
