from .module import Module


class Sequential(Module):
    """Applies a sequence of modules in the order they are passed."""

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._register_module(str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x
