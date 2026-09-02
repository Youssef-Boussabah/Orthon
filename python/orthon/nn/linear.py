import math

import orthon as ortho
from .._random import uniform as _uniform
from .module import Module, Parameter


def _init_weight(in_f, out_f):
    """Kaiming uniform init: U(-sqrt(6/fan_in), sqrt(6/fan_in))."""
    bound = math.sqrt(6.0 / in_f)
    t = ortho.empty([in_f, out_f])
    for i in range(in_f):
        for j in range(out_f):
            t.set_item([i, j], _uniform(-bound, bound))
    return t


class Linear(Module):
    """Fully-connected layer: y = x @ weight + bias.

    weight: (in_features, out_features)
    bias:   (out_features,)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(_init_weight(in_features, out_features))
        self.bias = Parameter(ortho.zeros([out_features]))

    def forward(self, x):
        return x @ self.weight + self.bias


class FusedLinearReLU(Module):
    """Fully-connected layer fused with ReLU: y = relu(x @ weight + bias)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(_init_weight(in_features, out_features))
        self.bias = Parameter(ortho.zeros([out_features]))

    def forward(self, x):
        return ortho.fused_linear_relu(x, self.weight, self.bias)
