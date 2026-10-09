"""Public-operation workloads shared by the RNG specifications."""

import numpy as np

import orthon as ortho
from orthon import nn


def values(tensor):
    return tensor.to("cpu").tolist() if tensor.device.is_cuda() else tensor.tolist()


def parameters(model):
    return {name: values(value) for name, value in model.named_parameters()}


def attention_inputs(device="cpu"):
    arrays = [np.zeros((1, 8, 8), dtype=np.float32),
              np.zeros((1, 8, 8), dtype=np.float32), np.eye(8, dtype=np.float32)[None]]
    return [ortho.from_numpy(array).to(device) for array in arrays]


def mixed_trace(seed, device="cpu"):
    ortho.manual_seed(seed)
    first = nn.Linear(8, 8).to(device)
    drop = nn.Dropout(.25)
    embedding = nn.Embedding(9, 8, padding_idx=2, device=device)
    x = ortho.ones([8, 8]).to(device)
    q, k, v = attention_inputs(device)
    return {
        "linear": parameters(first), "embedding": parameters(embedding),
        "module_calls": [values(drop(x)) for _ in range(3)],
        "attention_calls": [values(ortho.scaled_dot_product_attention(
            q, k, v, dropout_p=.25)) for _ in range(3)],
        "next_linear": parameters(nn.Linear(8, 8)),
    }


class StochasticModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.drop = nn.Dropout(.5)
        self.to(device)

    def forward(self, x):
        return self.drop(self.projection(x))


def train(model, optimizer, steps):
    x = ortho.from_numpy(np.array([[1, .5, -.5, 2], [-1, 2, .25, .5]],
                                 dtype=np.float32)).to(model.projection.weight.device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        residual = model(x) - .25
        loss = (residual * residual).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses
