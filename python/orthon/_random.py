"""Framework-local RNG, independent of Python's global random state.

manual_seed resets one process-local stream. Layers consume consecutive
values without reseeding; distributed callers choose any rank-specific seed."""
import random as _stdlib_random

# The single framework generator. Unseeded here on purpose (see the module
# docstring): a constant seed at import would make every process start from
# identical weights.
_rng = _stdlib_random.Random()


def manual_seed(seed):
    """Reset the Orthon RNG stream.

    Affects Orthon module initialization and default stochastic behavior
    only. It does not touch Python's global ``random``, NumPy, PyTorch, or any
    CUDA device state, and it does not modify tensors that already exist.
    """
    _rng.seed(int(seed))


def uniform(a, b):
    """Draw from U(a, b) on the framework stream."""
    return _rng.uniform(a, b)


def randbits(k):
    """Draw a k-bit unsigned integer from the framework stream."""
    return _rng.getrandbits(k)
