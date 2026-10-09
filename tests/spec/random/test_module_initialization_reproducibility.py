import pytest

import orthon as ortho
from orthon import nn

from .workload import parameters, values


BUILDERS = [pytest.param(lambda: nn.Linear(8, 8), id="linear"),
            pytest.param(lambda: nn.FusedLinearReLU(8, 8), id="fused-linear"),
            pytest.param(lambda: nn.Conv2d(2, 4, 3), id="convolution"),
            pytest.param(lambda: nn.Embedding(9, 8, padding_idx=2), id="embedding"),
            pytest.param(lambda: nn.SelfAttention(8, dropout=.25), id="self-attention"),
            pytest.param(lambda: nn.MultiHeadSelfAttention(8, 2, dropout=.25), id="multihead"),
            pytest.param(lambda: nn.TransformerEncoderBlock(8, 2, dropout=.25), id="transformer")]


@pytest.mark.parametrize("build", BUILDERS)
def test_construction_advances_but_reseeding_replays_parameter_sequence(build):
    ortho.manual_seed(41)
    first, second = build(), build()
    a, b = parameters(first), parameters(second)
    assert a != b
    ortho.manual_seed(41)
    replay_first, replay_second = build(), build()
    assert parameters(replay_first) == a
    assert parameters(replay_second) == b
    ortho.manual_seed(42)
    assert parameters(build()) != a
    # Equal initial values belong to independent trainable owners.
    with ortho.no_grad():
        ortho.fill_(next(iter(replay_first.parameters())), 7.)
    assert parameters(first) == a
    assert parameters(second) == b


@pytest.mark.parametrize("build", BUILDERS)
def test_construction_order_changes_the_subsequent_layer_stream(build):
    ortho.manual_seed(43)
    build()
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(43)
    assert parameters(nn.Linear(8, 8)) != expected
    ortho.manual_seed(43)
    build()
    assert parameters(nn.Linear(8, 8)) == expected


def test_attention_projections_receive_independent_samples():
    ortho.manual_seed(47)
    model = nn.MultiHeadSelfAttention(8, 2)
    weights = [values(layer.weight) for layer in
               (model.q_proj, model.k_proj, model.v_proj, model.out_proj)]
    assert len({tuple(weight) for weight in weights}) == 4
    held = weights[1:]
    with ortho.no_grad():
        ortho.fill_(model.q_proj.weight, 0.)
    assert [values(layer.weight) for layer in
            (model.k_proj, model.v_proj, model.out_proj)] == held


def test_deterministic_parameter_and_normalization_construction_do_not_draw():
    ortho.manual_seed(49)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(49)
    nn.Parameter(ortho.ones([8, 8]))
    nn.LayerNorm(8)
    nn.BatchNorm1d(8)
    nn.BatchNorm2d(8)
    nn.ReLU()
    assert parameters(nn.Linear(8, 8)) == expected


@pytest.mark.parametrize("build,shape", [
    (lambda: nn.Linear(8, 5), (8, 5)),
    (lambda: nn.FusedLinearReLU(8, 5), (8, 5)),
    (lambda: nn.Conv2d(2, 5, (2, 3)), (5, 2, 2, 3)),
    (lambda: nn.Embedding(9, 5, padding_idx=-1), (9, 5)),
])
def test_random_initial_parameters_have_public_metadata_and_fixed_zero_entries(build, shape):
    ortho.manual_seed(50)
    layer = build()
    assert layer.weight.shape == shape
    for parameter in layer.parameters():
        assert parameter.dtype == "float32"
        assert parameter.device == ortho.Device("cpu")
        assert parameter.requires_grad and parameter.is_leaf
        assert parameter.grad is None
    if isinstance(layer, nn.Embedding):
        assert values(layer.weight[-1]) == [0.] * 5
    else:
        assert values(layer.bias) == [0.] * 5
