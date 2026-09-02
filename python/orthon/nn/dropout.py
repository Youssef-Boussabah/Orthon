import orthon as ortho
from .._random import randbits as _randbits
from .module import Module


class Dropout(Module):
    """Inverted dropout with native CPU/CUDA mask generation."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout p must be in [0, 1)")
        self.p = float(p)
        # An explicit seed stays authoritative and ignores the framework
        # stream; the default path draws from the manual_seed()-controlled
        # stream, so a model that contains Dropout is reproducible.
        self.seed = int(seed) if seed is not None else _randbits(64)
        self._counter = 0

    def _next_seed(self):
        # Stateless kernels use this per-call seed plus element index. Fixed
        # module seeds are deterministic; repeated calls still advance masks.
        seed = (self.seed + self._counter * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        self._counter += 1
        return seed

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        return ortho.dropout(x, self.p, self._next_seed())
