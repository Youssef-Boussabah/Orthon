import pytest

import orthon as ortho
from orthon import nn

from .workload import parameters, values


@pytest.mark.parametrize("build", [
    pytest.param(lambda device: nn.Linear(8, 8).to(device), id="linear-move"),
    pytest.param(lambda device: nn.Embedding(9, 8, device=device), id="embedding-device"),
])
def test_initialization_and_later_draws_replay_on_the_selected_device(build, operator_device):
    ortho.manual_seed(107)
    model = build(operator_device)
    expected, following = parameters(model), parameters(nn.Linear(8, 8))
    ortho.manual_seed(107)
    replay = build(operator_device)
    assert parameters(replay) == expected
    assert parameters(nn.Linear(8, 8)) == following
    assert all(p.device == ortho.Device(operator_device) for p in replay.parameters())


def test_explicit_cuda_dropout_sequence_is_unaffected_by_cpu_rng_activity(cuda_device):
    x = ortho.ones([256]).to(cuda_device)
    reference, subject = nn.Dropout(.5, seed=109), nn.Dropout(.5, seed=109)
    expected = [values(reference(x)) for _ in range(2)]
    assert values(subject(x)) == expected[0]
    ortho.manual_seed(111)
    nn.Linear(8, 8)
    nn.Dropout(.5)(ortho.ones([256]))
    assert values(subject(x)) == expected[1]


def test_default_cuda_dropout_construction_shares_the_host_seed_stream(cuda_device):
    x = ortho.ones([256]).to(cuda_device)
    ortho.manual_seed(113)
    expected = values(nn.Dropout(.5)(x))
    ortho.manual_seed(113)
    nn.Linear(8, 8)
    shifted = values(nn.Dropout(.5)(x))
    assert shifted != expected
    ortho.manual_seed(113)
    nn.Linear(8, 8)
    assert values(nn.Dropout(.5)(x)) == shifted
