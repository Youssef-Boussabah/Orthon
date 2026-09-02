from .module import Module


class Flatten(Module):
    """Flatten a contiguous range of dimensions."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)
