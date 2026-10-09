import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

import orthon as ortho
from orthon import nn

from .workload import mixed_trace, parameters


@pytest.mark.parametrize("seed", [0, 17, -17, 2**64 - 1, 2**64, 2**256 + 23])
def test_integer_seeding_replays_a_mixed_operation_sequence(seed, operator_device):
    assert ortho.manual_seed(seed) is None
    expected = mixed_trace(seed, operator_device)
    mixed_trace(951, operator_device)
    assert mixed_trace(seed, operator_device) == expected


def test_selected_different_seeds_change_the_observable_sequence():
    assert mixed_trace(17) != mixed_trace(18)


@pytest.mark.parametrize("seed", [5.0, "5", np.int64(5)])
def test_integer_convertible_seeds_replay_the_integer_seed(seed):
    assert mixed_trace(seed) == mixed_trace(5)


@pytest.mark.parametrize("seed,error", [(None, TypeError), (object(), TypeError),
                                        ("invalid", ValueError), (float("nan"), ValueError),
                                        (float("inf"), OverflowError)])
def test_rejected_seed_conversion_leaves_the_stream_usable(seed, error):
    ortho.manual_seed(29)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(29)
    with pytest.raises(error):
        ortho.manual_seed(seed)
    assert parameters(nn.Linear(8, 8)) == expected


def test_reseeding_preserves_existing_values_gradients_and_saved_backward():
    ortho.manual_seed(23)
    model = nn.Linear(4, 4)
    before = parameters(model)
    (model.weight * 2).backward()
    saved = model.weight * model.weight
    ortho.manual_seed(24)
    assert parameters(model) == before
    assert model.weight.grad.tolist() == [2.] * 16
    saved.backward()
    assert model.weight.grad.tolist() == pytest.approx(
        [2 + 2 * value for value in before["weight"]])


def test_orthon_and_external_python_numpy_streams_are_isolated_both_ways():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(71)
        np.random.seed(72)
        expected_external = (random.random(), np.random.random(4).tolist())
        random.seed(71)
        np.random.seed(72)
        expected_orthon = mixed_trace(31)
        assert (random.random(), np.random.random(4).tolist()) == expected_external
        ortho.manual_seed(31)
        random.seed(99)
        np.random.seed(98)
        random.random()
        np.random.random(100)
        # Build without another seed call, so outside activity cannot be hidden by a reset.
        assert parameters(nn.Linear(8, 8)) == expected_orthon["linear"]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_orthon_seeding_and_draws_leave_torch_cpu_rng_unchanged():
    torch = pytest.importorskip("torch")
    state = torch.get_rng_state()
    try:
        torch.manual_seed(51)
        expected = torch.rand(8)
        torch.manual_seed(51)
        mixed_trace(52)
        assert torch.equal(torch.rand(8), expected)
        ortho.manual_seed(53)
        expected_weights = parameters(nn.Linear(8, 8))
        ortho.manual_seed(53)
        torch.manual_seed(54)
        torch.rand(100)
        assert parameters(nn.Linear(8, 8)) == expected_weights
    finally:
        torch.set_rng_state(state)


@pytest.mark.parametrize("seed", [37, -37, 2**128 + 37])
def test_seeded_sequence_replays_in_fresh_cpu_processes(seed, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(__file__).parent), env.get("PYTHONPATH"))))
    script = "import json,sys; from workload import mixed_trace; print(json.dumps(mixed_trace(int(sys.argv[1]))))"
    expected = mixed_trace(seed)
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", script, str(seed)], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout) == expected
