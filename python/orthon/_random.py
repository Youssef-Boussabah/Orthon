"""Framework-local random number generator.

Orthon owns exactly one generator instance, private to this module. Nothing
outside gets a reference to it, and it is deliberately NOT Python's module-level
``random`` state.

That separation is the point. With a shared generator, unrelated user code --
``random.shuffle(dataset)``, a third-party library drawing a few values at
import time -- would silently shift model initialization, so a "reproducible"
run would stop reproducing for reasons nowhere near the model. Conversely,
seeding Orthon would yank the user's own sampling out from under them.

The generator is left in its default process-seeded state at import, so
constructing a layer without seeding behaves exactly as before. Determinism is
opt-in through ``orthon.manual_seed()``, which resets this one stream; the
modules that draw from it never reseed, so consecutive layers keep consuming
consecutive values and stay statistically distinct.

Scope is deliberately process-local Python state: no thread-local generators,
no per-CUDA-device streams, no automatic rank offsets. A distributed user who
wants rank-specific initialization calls ``manual_seed(base + rank)``.
"""
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
