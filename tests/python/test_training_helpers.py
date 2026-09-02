import pytest

import orthon
from orthon.nn import Linear
from orthon.optim import SGD
from orthon.training import eval_step, train_step, weighted_metric_average


class SpyOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def test_train_step_changes_model_parameters():
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    x = orthon.full([2, 2], 1.0)
    before = model.weight.tolist()[:]

    train_step(model, x, [0, 1], orthon.cross_entropy, optimizer)

    assert model.weight.tolist() != before


def test_train_step_returns_metrics():
    model = Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    x = orthon.full([3, 2], 0.5)

    result = train_step(model, x, [0, 1, 0], orthon.cross_entropy, optimizer)

    assert set(result.keys()) == {"loss", "accuracy", "num_examples"}
    assert isinstance(result["loss"], float)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["num_examples"] == 3


def test_eval_step_does_not_call_optimizer_step():
    model = Linear(2, 2)
    optimizer = SpyOptimizer()
    x = orthon.full([2, 2], 1.0)
    before = model.weight.tolist()[:]

    result = eval_step(model, x, [0, 1], orthon.cross_entropy)

    assert optimizer.step_calls == 0
    assert optimizer.zero_grad_calls == 0
    assert model.weight.tolist() == pytest.approx(before)
    assert result["num_examples"] == 2


def test_weighted_metric_average_handles_uneven_batch_sizes():
    result = weighted_metric_average([
        {"loss": 2.0, "accuracy": 0.5, "num_examples": 2},
        {"loss": 1.0, "accuracy": 1.0, "num_examples": 6},
    ])

    assert result["loss"] == pytest.approx(1.25)
    assert result["accuracy"] == pytest.approx(0.875)
    assert result["num_examples"] == 8
