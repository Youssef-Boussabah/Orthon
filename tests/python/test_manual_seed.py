"""Deterministic framework RNG streams independent of user random state.

Layers consume consecutive values without reseeding; determinism is opt-in."""
from tests.support.capabilities import orthon_cuda_available
import math
import os
import random

import numpy as np
import pytest

import orthon as ortho
from orthon.nn import (Conv2d, Dropout, Embedding, FusedLinearReLU, Linear)

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


def _v(t):
    return np.array(t.tolist(), dtype=np.float64)


def _weights(m):
    return [_v(p) for p in m.parameters()]


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# Public API
def test_manual_seed_is_public_and_returns_none():
    assert callable(ortho.manual_seed)
    assert ortho.manual_seed(0) is None


def test_manual_seed_accepts_int_convertible_input():
    ortho.manual_seed(5)
    a = _v(Linear(4, 3).weight)
    ortho.manual_seed(5.0)              # int-convertible
    b = _v(Linear(4, 3).weight)
    assert np.array_equal(a, b)


def test_manual_seed_does_not_touch_existing_tensors():
    ortho.manual_seed(3)
    layer = Linear(4, 3)
    before = _v(layer.weight).copy()
    ortho.manual_seed(99)
    assert _v(layer.weight) == pytest.approx(before, abs=0)


# Reproducible initialization, per module type
BUILDERS = {
    "Linear": lambda: Linear(12, 7),
    "FusedLinearReLU": lambda: FusedLinearReLU(12, 7),
    "Conv2d": lambda: Conv2d(3, 5, 3),
    "Embedding": lambda: Embedding(9, 6),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_same_seed_reproduces_initialization_exactly(name):
    ortho.manual_seed(1234)
    first = _weights(BUILDERS[name]())
    ortho.manual_seed(1234)
    second = _weights(BUILDERS[name]())
    assert _same(first, second)
    # ...and the values are actually random, not a constant fill.
    assert len(np.unique(first[0])) > 1


def test_linear_stays_within_kaiming_bounds():
    ortho.manual_seed(2)
    layer = Linear(16, 8)
    bound = math.sqrt(6.0 / 16)
    w = _v(layer.weight)
    assert np.abs(w).max() <= bound
    assert np.abs(w).max() > 0.5 * bound      # spread, not clustered at zero
    assert _v(layer.bias) == pytest.approx(np.zeros(8), abs=0)


def test_conv2d_stays_within_kaiming_bounds():
    ortho.manual_seed(2)
    layer = Conv2d(3, 4, 3)
    bound = math.sqrt(6.0 / (3 * 3 * 3))
    w = _v(layer.weight)
    assert np.abs(w).max() <= bound
    assert np.abs(w).max() > 0.5 * bound


def test_embedding_stays_within_bounds_and_keeps_padding_row_zero():
    ortho.manual_seed(2)
    layer = Embedding(9, 4, padding_idx=2)
    bound = 1.0 / math.sqrt(9)
    w = _v(layer.weight).reshape(9, 4)
    assert np.abs(w).max() <= bound
    assert np.all(w[2] == 0.0)
    assert np.abs(np.delete(w, 2, axis=0)).max() > 0.0


def test_embedding_padding_row_stays_zero_across_reseeds():
    for seed in (1, 2, 3):
        ortho.manual_seed(seed)
        w = _v(Embedding(6, 3, padding_idx=0).weight).reshape(6, 3)
        assert np.all(w[0] == 0.0)


# Stream progression -- the heart of the contract
def test_consecutive_layers_differ():
    """A per-layer constant seed would make these identical."""
    ortho.manual_seed(123)
    a1, a2 = Linear(8, 8), Linear(8, 8)
    assert not np.array_equal(_v(a1.weight), _v(a2.weight))


def test_reseeding_reproduces_the_whole_layer_sequence():
    ortho.manual_seed(123)
    a1, a2 = Linear(8, 8), Linear(8, 8)
    ortho.manual_seed(123)
    b1, b2 = Linear(8, 8), Linear(8, 8)
    assert np.array_equal(_v(a1.weight), _v(b1.weight))
    assert np.array_equal(_v(a2.weight), _v(b2.weight))
    assert not np.array_equal(_v(b1.weight), _v(b2.weight))


def test_mixed_module_sequence_reproduces():
    def build():
        ortho.manual_seed(7)
        return (_v(Linear(4, 3).weight), _v(Conv2d(2, 3, 3).weight),
                _v(Embedding(5, 4).weight), _v(FusedLinearReLU(3, 3).weight))

    first, second = build(), build()
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


def test_a_layer_does_not_reseed_the_stream():
    """Constructing a module mid-sequence must consume from the stream, not
    restart it: the value drawn after two layers differs from after one."""
    ortho.manual_seed(50)
    Linear(4, 4)
    one = _v(Linear(4, 4).weight)
    ortho.manual_seed(50)
    Linear(4, 4)
    Linear(4, 4)
    two = _v(Linear(4, 4).weight)
    assert not np.array_equal(one, two)


# Isolation from Python's global random
def test_python_random_activity_cannot_perturb_orthon():
    ortho.manual_seed(123)
    expected = _v(Linear(6, 5).weight)

    ortho.manual_seed(123)
    random.seed(999)
    for _ in range(1000):
        random.random()
    actual = _v(Linear(6, 5).weight)

    assert np.array_equal(actual, expected)


def test_manual_seed_does_not_perturb_python_random():
    random.seed(456)
    expected = random.random()

    random.seed(456)
    ortho.manual_seed(999)
    actual = random.random()

    assert actual == expected


def test_orthon_construction_does_not_consume_python_random():
    random.seed(11)
    expected = [random.random() for _ in range(3)]

    random.seed(11)
    Linear(8, 8)
    Conv2d(2, 2, 3)
    Dropout(0.5)
    actual = [random.random() for _ in range(3)]

    assert actual == expected


def test_manual_seed_does_not_seed_numpy():
    np.random.seed(3)
    expected = np.random.rand(4).tolist()
    np.random.seed(3)
    ortho.manual_seed(12345)
    assert np.random.rand(4).tolist() == expected


# Dropout
def test_default_dropout_seed_is_reproducible():
    ortho.manual_seed(5)
    d1 = Dropout(0.5)
    ortho.manual_seed(5)
    d2 = Dropout(0.5)
    assert d1.seed == d2.seed


def test_consecutive_default_dropouts_get_different_seeds():
    ortho.manual_seed(5)
    d1, d2 = Dropout(0.5), Dropout(0.5)
    assert d1.seed != d2.seed
    ortho.manual_seed(5)
    e1, e2 = Dropout(0.5), Dropout(0.5)
    assert (e1.seed, e2.seed) == (d1.seed, d2.seed)


def test_explicit_dropout_seed_ignores_the_framework_stream():
    ortho.manual_seed(1)
    a = Dropout(0.5, seed=7)
    ortho.manual_seed(999)
    b = Dropout(0.5, seed=7)
    assert a.seed == b.seed == 7


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if CUDA else []))
def test_default_dropout_outputs_reproduce(dev):
    x = ortho.from_numpy(np.ones((4, 8), dtype=np.float32))
    if dev != "cpu":
        x = x.to(dev)

    ortho.manual_seed(11)
    d1 = Dropout(0.5)
    d1.train()
    ortho.manual_seed(11)
    d2 = Dropout(0.5)
    d2.train()

    assert np.array_equal(_v(d1(x)), _v(d2(x)))


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if CUDA else []))
def test_explicit_dropout_outputs_reproduce_regardless_of_framework_seed(dev):
    x = ortho.from_numpy(np.ones((4, 8), dtype=np.float32))
    if dev != "cpu":
        x = x.to(dev)

    ortho.manual_seed(1)
    a = Dropout(0.5, seed=7)
    a.train()
    ortho.manual_seed(999)
    b = Dropout(0.5, seed=7)
    b.train()

    assert np.array_equal(_v(a(x)), _v(b(x)))


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if CUDA else []))
def test_repeated_dropout_calls_still_advance_the_mask(dev):
    """The per-module dropout counter is independent of manual_seed."""
    x = ortho.from_numpy(np.ones((4, 8), dtype=np.float32))
    if dev != "cpu":
        x = x.to(dev)
    d = Dropout(0.5, seed=7)
    d.train()
    first, second = _v(d(x)), _v(d(x))
    assert not np.array_equal(first, second)
    assert d._counter == 2


def test_dropout_in_eval_mode_is_identity_and_draws_nothing():
    x = ortho.from_numpy(np.ones((4, 8), dtype=np.float32))
    d = Dropout(0.5, seed=7)
    d.eval()
    assert np.array_equal(_v(d(x)), _v(x))
    assert d._counter == 0


# Functional attention dropout
def _qkv(dev="cpu"):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(3):
        t = ortho.from_numpy(rng.normal(size=(2, 4, 6)).astype(np.float32))
        out.append(t.to(dev) if dev != "cpu" else t)
    return out


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if CUDA else []))
def test_attention_dropout_reproduces_under_the_same_seed(dev):
    q, k, v = _qkv(dev)
    ortho.manual_seed(21)
    y1 = _v(ortho.scaled_dot_product_attention(q, k, v, dropout_p=0.25, training=True))
    ortho.manual_seed(21)
    y2 = _v(ortho.scaled_dot_product_attention(q, k, v, dropout_p=0.25, training=True))
    assert np.array_equal(y1, y2)


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if CUDA else []))
def test_attention_dropout_advances_without_reseeding(dev):
    q, k, v = _qkv(dev)
    ortho.manual_seed(21)
    y1 = _v(ortho.scaled_dot_product_attention(q, k, v, dropout_p=0.25, training=True))
    y2 = _v(ortho.scaled_dot_product_attention(q, k, v, dropout_p=0.25, training=True))
    assert not np.array_equal(y1, y2)


@pytest.mark.parametrize("p,training", [(0.0, True), (0.25, False), (0.0, False)])
def test_disabled_attention_dropout_does_not_advance_the_stream(p, training):
    """Checked through a surrounding seeded draw rather than by inspecting the
    generator: if the disabled call consumed a value, the Linear built after it
    would differ from one built with no call at all."""
    q, k, v = _qkv()

    ortho.manual_seed(33)
    ortho.scaled_dot_product_attention(q, k, v, dropout_p=p, training=training)
    after_call = _v(Linear(4, 4).weight)

    ortho.manual_seed(33)
    no_call = _v(Linear(4, 4).weight)

    assert np.array_equal(after_call, no_call)


def test_enabled_attention_dropout_does_advance_the_stream():
    """Companion to the test above: proves that check is not vacuous."""
    q, k, v = _qkv()

    ortho.manual_seed(33)
    ortho.scaled_dot_product_attention(q, k, v, dropout_p=0.25, training=True)
    after_call = _v(Linear(4, 4).weight)

    ortho.manual_seed(33)
    no_call = _v(Linear(4, 4).weight)

    assert not np.array_equal(after_call, no_call)


# Full-model reproducibility
def test_a_model_containing_dropout_is_fully_reproducible():
    """Dropout's default seed comes from the framework generator, so a
    model containing one reproduces. A secrets.randbits default would put it
    beyond the reach of any seed."""
    from orthon.nn import ReLU, Sequential

    def build():
        ortho.manual_seed(404)
        m = Sequential(Linear(6, 5), ReLU(), Dropout(0.3), Linear(5, 2))
        return _weights(m), [mod.seed for mod in m._modules.values()
                             if isinstance(mod, Dropout)]

    (w1, s1), (w2, s2) = build(), build()
    assert _same(w1, w2)
    assert s1 == s2 and len(s1) == 1


def test_adam_convergence_model_is_deterministic():
    """Pins the exact fix applied to test_adam_loss_decreases."""
    from orthon.nn import ReLU, Sequential
    from orthon.optim import Adam

    def run():
        ortho.manual_seed(22)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        optim = Adam(model.parameters(), lr=0.01)
        x = ortho.full([4, 4], 0.1)
        losses = []
        for _ in range(30):
            model.zero_grad()
            loss = ortho.cross_entropy(model(x), [0, 1, 0, 1])
            losses.append(loss.item())
            loss.backward()
            optim.step()
        return losses[0], losses[-1]

    a, b = run(), run()
    assert a == b                       # bit-identical across runs
    assert b[1] < b[0]
    # A healthy margin, not a threshold squeak: the unseeded median was ~0.0017.
    assert b[0] - b[1] > 0.01


# Source-level contract
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNTIME = os.path.join(REPO, "python", "orthon")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _runtime_files():
    out = []
    for root, _dirs, files in os.walk(RUNTIME):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    return sorted(out)


def test_framework_rng_is_not_seeded_to_a_constant_at_import():
    """Determinism is opt-in. Asserted by inspection, not statistically: a test
    that two processes differ could fail by coincidence."""
    src = _read(os.path.join(RUNTIME, "_random.py"))
    assert "_stdlib_random.Random()" in src
    body = src[src.index("_rng = "):]
    module_level = body[:body.index("def manual_seed")]
    assert "seed(" not in module_level


def test_the_generator_object_is_not_exposed():
    assert not hasattr(ortho, "_rng")
    from orthon import _random
    assert "_rng" in dir(_random)          # private module attribute only
    assert set(n for n in dir(_random) if not n.startswith("_")) == {
        "manual_seed", "randbits", "uniform"}


def test_manual_seed_does_not_use_the_global_random_module_state():
    src = _read(os.path.join(RUNTIME, "_random.py"))
    # The stdlib module is aliased and only used to construct Random(); no bare
    # random.seed()/random.uniform() module-level calls.
    assert "_stdlib_random.seed(" not in src
    assert "_stdlib_random.uniform(" not in src


def test_only_the_rng_module_touches_stdlib_random_or_secrets():
    """Census: no framework stochastic source outside _random.py."""
    offenders = []
    for path in _runtime_files():
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        if rel.endswith("orthon/_random.py"):
            continue
        src = _read(path)
        # strip comments so explanatory prose does not trip the check
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        for token in ("import random", "import secrets", "secrets.randbits",
                      "random.uniform", "random.randrange", "random.getrandbits"):
            if token in code:
                offenders.append((rel, token))
    assert offenders == [], "uncontrolled framework randomness: %s" % offenders
