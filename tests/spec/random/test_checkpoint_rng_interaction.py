import pytest

import orthon as ortho
from orthon import nn, optim

from .workload import StochasticModel, attention_inputs, parameters, train, values


@pytest.mark.parametrize("api", ["checkpoint", "state-dict"])
def test_checkpoint_io_is_observational_for_the_current_global_stream(api, tmp_path):
    ortho.manual_seed(127)
    source, destination = nn.Linear(8, 8), nn.Linear(8, 8)
    expected = parameters(nn.Linear(8, 8))
    ortho.manual_seed(127)
    source, destination = nn.Linear(8, 8), nn.Linear(8, 8)
    path = tmp_path / "rng.ckpt"
    if api == "checkpoint":
        ortho.save_checkpoint(path, model=source, metadata={"seed": 127})
        assert ortho.load_checkpoint(path, model=destination) == {"seed": 127}
    else:
        ortho.save_state_dict(source, path)
        ortho.load_state_dict(destination, path)
    assert parameters(destination) == parameters(source)
    assert parameters(nn.Linear(8, 8)) == expected


def test_saving_does_not_interrupt_an_existing_stochastic_training_sequence(operator_device, tmp_path):
    ortho.manual_seed(131)
    reference = StochasticModel(operator_device)
    optimizer = optim.SGD(reference.parameters(), lr=.01)
    reference.eval()
    train(reference, optimizer, 2)
    reference.train()
    expected_losses = train(reference, optimizer, 3)
    expected_state = parameters(reference)
    ortho.manual_seed(131)
    subject = StochasticModel(operator_device)
    subject_optimizer = optim.SGD(subject.parameters(), lr=.01)
    subject.eval()
    train(subject, subject_optimizer, 2)
    subject.train()
    losses = train(subject, subject_optimizer, 1)
    ortho.save_checkpoint(tmp_path / "stochastic.ckpt", model=subject, optimizer=subject_optimizer)
    losses += train(subject, subject_optimizer, 2)
    assert losses == expected_losses
    assert parameters(subject) == expected_state


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam"])
def test_caller_replay_can_reconstruct_stochastic_position_before_file_restore(
    optimizer_name, operator_device, tmp_path,
):
    ortho.manual_seed(137)
    model = StochasticModel(operator_device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=.01)
    model.eval()
    train(model, optimizer, 2)
    model.train()
    train(model, optimizer, 2)
    path = tmp_path / "replay.ckpt"
    ortho.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"seed": 137})
    expected = train(model, optimizer, 2)
    expected_state = parameters(model)
    # This replay, rather than loading alone, reconstructs the stochastic position.
    ortho.manual_seed(137)
    resumed = StochasticModel(operator_device)
    restored = getattr(optim, optimizer_name)(resumed.parameters(), lr=.01)
    resumed.eval()
    train(resumed, restored, 2)
    resumed.train()
    train(resumed, restored, 2)
    assert ortho.load_checkpoint(path, model=resumed, optimizer=restored) == {"seed": 137}
    assert train(resumed, restored, 2) == expected
    assert parameters(resumed) == expected_state


def test_loading_into_an_existing_dropout_preserves_its_active_call_position(operator_device, tmp_path):
    x = ortho.ones([256]).to(operator_device)
    model, reference = nn.Dropout(.5, seed=139), nn.Dropout(.5, seed=139)
    assert values(model(x)) == values(reference(x))
    path = tmp_path / "dropout.ckpt"
    ortho.save_checkpoint(path, model=model)
    assert values(model(x)) == values(reference(x))
    ortho.load_checkpoint(path, model=model)
    assert values(model(x)) == values(reference(x))


def test_caller_replay_restores_functional_attention_draw_position(operator_device, tmp_path):
    q, k, v = attention_inputs(operator_device)
    ortho.manual_seed(149)
    source = nn.Linear(8, 8).to(operator_device)
    prefix = values(ortho.scaled_dot_product_attention(q, k, v, dropout_p=.5))
    path = tmp_path / "attention.ckpt"
    ortho.save_checkpoint(path, model=source)
    expected = values(ortho.scaled_dot_product_attention(q, k, v, dropout_p=.5))
    ortho.manual_seed(149)
    resumed = nn.Linear(8, 8).to(operator_device)
    assert values(ortho.scaled_dot_product_attention(q, k, v, dropout_p=.5)) == prefix
    ortho.load_checkpoint(path, model=resumed)
    assert values(ortho.scaled_dot_product_attention(q, k, v, dropout_p=.5)) == expected
