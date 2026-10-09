import numpy as np
import pytest

import orthon as ortho
from orthon import nn

from .workload import attention_inputs, parameters, values


@pytest.mark.parametrize("seed", [None, 0, -7, 2**64 - 1, 2**128 + 7])
def test_module_call_sequence_replays_from_equivalent_construction(seed, operator_device):
    x = ortho.ones([256]).to(operator_device)
    ortho.manual_seed(61)
    first = nn.Dropout(.5, seed=seed)
    expected = [values(first(x)) for _ in range(3)]
    assert expected[0] != expected[1]
    ortho.manual_seed(61)
    second = nn.Dropout(.5, seed=seed)
    assert [values(second(x)) for _ in range(3)] == expected


@pytest.mark.parametrize("seed", [None, 7])
def test_manual_reseed_does_not_rewind_an_existing_dropout(seed, operator_device):
    x = ortho.ones([256]).to(operator_device)
    ortho.manual_seed(63)
    reference = nn.Dropout(.5, seed=seed)
    expected = [values(reference(x)) for _ in range(2)]
    ortho.manual_seed(63)
    subject = nn.Dropout(.5, seed=seed)
    assert values(subject(x)) == expected[0]
    ortho.manual_seed(999)
    assert values(subject(x)) == expected[1]


@pytest.mark.parametrize("disabled", ["eval", "zero-probability"])
def test_disabled_calls_preserve_the_modules_next_active_result(disabled, operator_device):
    x = ortho.ones([256]).to(operator_device)
    subject, reference = nn.Dropout(.5, seed=67), nn.Dropout(.5, seed=67)
    assert values(subject(x)) == values(reference(x))
    if disabled == "eval":
        subject.eval()
    else:
        subject.p = 0.
    for _ in range(3):
        assert subject(x) is x
    subject.train()
    subject.p = .5
    assert values(subject(x)) == values(reference(x))


@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("p", [0., .5])
def test_only_default_module_construction_consumes_the_global_stream(seed, p, operator_device):
    x = ortho.ones([256]).to(operator_device)
    ortho.manual_seed(69)
    baseline = parameters(nn.Linear(8, 8))
    ortho.manual_seed(69)
    drop = nn.Dropout(p, seed=seed)
    after_construction = parameters(nn.Linear(8, 8))
    assert (after_construction == baseline) is (seed is not None)
    ortho.manual_seed(69)
    for _ in range(3):
        drop(x)
    assert parameters(nn.Linear(8, 8)) == baseline


@pytest.mark.parametrize("p", [.25, .5, .75])
def test_mask_scale_is_used_in_weighted_backward_without_new_rng_draws(
    p, operator_device, typed_tensor, assert_output,
):
    data = np.arange(1, 129, dtype=np.float32).reshape(8, 16) / 8
    coefficient = np.arange(128, 0, -1, dtype=np.float32).reshape(8, 16) / 16
    x = typed_tensor(data, device=operator_device).requires_grad_(True)
    c = typed_tensor(coefficient, device=operator_device)
    drop, reference = nn.Dropout(p, seed=71), nn.Dropout(p, seed=71)
    y = drop(x)
    observed = np.asarray(values(y)).reshape(data.shape)
    kept = observed != 0.
    scale = 1 / (1 - p)
    assert np.allclose(observed, np.where(kept, data * scale, 0.), rtol=2e-6)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    replay_input = typed_tensor(data, device=operator_device).requires_grad_(True)
    replay_output = reference(replay_input)
    assert values(replay_output) == values(y)
    ortho.manual_seed(73)
    expected_next_layer = parameters(nn.Linear(8, 8))
    ortho.manual_seed(73)
    (y * c).backward()
    assert_output(x.grad, np.where(kept, coefficient * scale, 0.),
                  dtype="float32", device=operator_device)
    (replay_output * c).backward()
    assert values(replay_input.grad) == values(x.grad)
    assert y.grad is None
    assert parameters(nn.Linear(8, 8)) == expected_next_layer
    assert values(drop(x)) == values(reference(x))


@pytest.mark.parametrize("shape", [(), (256,), (4, 8, 8), (0,), (2, 0, 3)])
def test_native_dropout_replays_and_preserves_scalar_empty_and_ranked_shapes(shape, operator_device):
    x = ortho.full(list(shape), 3.).to(operator_device)
    first = ortho.dropout(x, .5, 79)
    assert first.shape == shape and first.dtype == x.dtype and first.device == x.device
    assert values(first) == values(ortho.dropout(x, .5, 79))
    assert set(values(first)) <= {0., 6.}


@pytest.mark.parametrize("layout", ["transpose", "offset-slice"])
def test_explicit_seed_uses_logical_view_order(layout, operator_device, typed_tensor):
    base = typed_tensor(np.arange(1, 129).reshape(8, 16), device=operator_device)
    view = base.transpose(0, 1) if layout == "transpose" else base.slice(0, 2, 6)
    before = values(base)
    actual = ortho.dropout(view, .5, 83)
    assert actual.shape == view.shape
    assert values(actual) == values(ortho.dropout(view.contiguous(), .5, 83))
    assert values(base) == before


def test_native_explicit_seed_calls_are_stateless_and_ignore_manual_seed(operator_device):
    x = ortho.ones([256]).to(operator_device)
    ortho.manual_seed(85)
    expected_layer = parameters(nn.Linear(8, 8))
    ortho.manual_seed(85)
    expected = values(ortho.dropout(x, .5, 2**64 - 1))
    ortho.dropout(x, .5, 0)
    assert parameters(nn.Linear(8, 8)) == expected_layer
    ortho.manual_seed(86)
    assert values(ortho.dropout(x, .5, 2**64 - 1)) == expected


@pytest.mark.parametrize("seed", [-1, 2**64, "7", None])
def test_native_seed_requires_an_unsigned_64_bit_integer(seed):
    x = ortho.ones([4])
    with pytest.raises(TypeError):
        ortho.dropout(x, .5, seed)
    assert values(x) == [1.] * 4


@pytest.mark.parametrize("p", [-.25, 1., 1.25, float("inf"), float("-inf")])
def test_invalid_probability_rejection_preserves_input_and_constructor_stream(p):
    ortho.manual_seed(89)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(89)
    with pytest.raises(ValueError):
        nn.Dropout(p)
    assert parameters(nn.Linear(8, 8)) == expected
    x = ortho.ones([4]).requires_grad_(True)
    with pytest.raises(RuntimeError):
        ortho.dropout(x, p, 7)
    assert values(x) == [1.] * 4
    assert x.grad is None


def test_active_dropout_rejects_other_storage_dtypes(storage_dtype, operator_device):
    if storage_dtype == "float32":
        x = ortho.ones([4]).to(operator_device)
        assert ortho.dropout(x, .5, 7).dtype == storage_dtype
    else:
        x = ortho.ones([4], storage_dtype).to(operator_device)
        with pytest.raises(RuntimeError):
            ortho.dropout(x, .5, 7)
        assert values(x) == [1.] * 4


@pytest.mark.parametrize("mode", ["eval", "module-zero", "native-zero"])
def test_dropout_passthrough_accepts_all_storage_dtypes(mode, storage_dtype, operator_device):
    x = ortho.ones([4], storage_dtype).to(operator_device)
    if mode == "native-zero":
        y = ortho.dropout(x, 0., 7)
    else:
        drop = nn.Dropout(0. if mode == "module-zero" else .5, seed=7)
        if mode == "eval":
            drop.eval()
        y = drop(x)
    assert values(y) == values(x)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device


@pytest.mark.parametrize("entry", [ortho.scaled_dot_product_attention,
                                   ortho.functional.scaled_dot_product_attention])
def test_attention_draws_per_active_call_and_reseeding_replays_sequence(entry, operator_device):
    q, k, v = attention_inputs(operator_device)
    ortho.manual_seed(97)
    expected = [values(entry(q, k, v, dropout_p=.5)) for _ in range(3)]
    next_layer = parameters(nn.Linear(8, 8))
    assert expected[0] != expected[1]
    # With zero scores and identity V, the result exposes weights numerically.
    assert all(abs(value) < 1e-6 or abs(value - .25) < 1e-6
               for output in expected for value in output)
    ortho.manual_seed(97)
    assert [values(entry(q, k, v, dropout_p=.5)) for _ in range(3)] == expected
    assert parameters(nn.Linear(8, 8)) == next_layer
    ortho.manual_seed(97)
    assert parameters(nn.Linear(8, 8)) != next_layer


@pytest.mark.parametrize("p,training", [(0., True), (.5, False)])
def test_disabled_attention_is_deterministic_and_does_not_consume_a_draw(
    p, training, operator_device,
):
    q, k, v = attention_inputs(operator_device)
    ortho.manual_seed(101)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(101)
    for _ in range(3):
        y = ortho.scaled_dot_product_attention(q, k, v, dropout_p=p, training=training)
        assert values(y) == pytest.approx([.125] * 64, abs=1e-6)
    assert parameters(nn.Linear(8, 8)) == expected


@pytest.mark.parametrize("p", [1., 1.25, float("inf")])
def test_active_attention_rejects_out_of_range_positive_probability(p, operator_device):
    q, k, v = attention_inputs(operator_device)
    with pytest.raises(RuntimeError):
        ortho.scaled_dot_product_attention(q, k, v, dropout_p=p)


@pytest.mark.parametrize("invalid", ["rank", "matching-shape"])
def test_attention_input_validation_occurs_without_consuming_the_stream(invalid):
    q, k, v = attention_inputs()
    q = q.reshape([8, 8]) if invalid == "rank" else q.slice(1, 0, 4)
    ortho.manual_seed(102)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(102)
    with pytest.raises(ValueError):
        ortho.scaled_dot_product_attention(q, k, v, dropout_p=.5)
    assert parameters(nn.Linear(8, 8)) == expected


@pytest.mark.parametrize("build", [
    pytest.param(lambda: nn.SelfAttention(8, dropout=.5), id="self-attention"),
    pytest.param(lambda: nn.MultiHeadSelfAttention(8, 2, dropout=.5), id="multihead"),
    pytest.param(lambda: nn.TransformerEncoderBlock(8, 2, dropout=.5), id="transformer"),
])
def test_composed_stochastic_layers_replay_train_sequence_and_have_stable_eval(
    build, operator_device, typed_tensor,
):
    x = typed_tensor(np.arange(64).reshape(1, 8, 8) / 64, device=operator_device)
    ortho.manual_seed(103)
    model = build().to(operator_device)
    expected = [values(model(x)) for _ in range(2)]
    assert expected[0] != expected[1]
    ortho.manual_seed(103)
    replay = build().to(operator_device)
    assert [values(replay(x)) for _ in range(2)] == expected
    replay.eval()
    evaluation = values(replay(x))
    assert values(replay(x)) == evaluation
