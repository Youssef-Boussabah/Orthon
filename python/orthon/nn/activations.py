from .module import Module


class ReLU(Module):
    """Element-wise ReLU activation (wraps the existing C++ relu op)."""

    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    """Element-wise sigmoid activation."""

    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    """Element-wise hyperbolic tangent activation."""

    def forward(self, x):
        return x.tanh()


class LeakyReLU(Module):
    """Element-wise LeakyReLU activation."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        return x.leaky_relu(self.negative_slope)


class GELU(Module):
    """Element-wise exact GELU activation."""

    def forward(self, x):
        return x.gelu()
