"""Current CIFAR models, learning-rate schedules and training/evaluation flow."""
import math

import orthon
import pytest
from orthon.optim import SGD

from tests.support.capabilities import orthon_cuda_available
from tests.support.examples import load_example

try:
    import torch
except ImportError:  # pragma: no cover - optional reference framework
    torch = None


def test_minivgg_forward_shape_cpu():
    cifar = load_example("cifar10_cnn.py")
    model = cifar.CIFAR10MiniVGG(hidden=32)
    x = orthon.full([2, 3, 32, 32], 0.1)

    logits = model(x)

    assert list(logits.shape) == [2, 10]


def test_make_model_constructs_minivgg():
    cifar = load_example("cifar10_cnn.py")

    class Args:
        model = "minivgg"
        hidden = 32
        channels = 8
        vgg_base_channels = 32
        init_scale = 0.5

    model = cifar.make_model(Args())

    assert isinstance(model, cifar.CIFAR10MiniVGG)


def test_parse_lr_step_epochs():
    cifar = load_example("cifar10_cnn.py")

    assert cifar.parse_lr_step_epochs("") == set()
    assert cifar.parse_lr_step_epochs("15,25") == {15, 25}
    assert cifar.parse_lr_step_epochs(" 5, 10, ") == {5, 10}


def test_apply_lr_step_schedule():
    cifar = load_example("cifar10_cnn.py")

    class Optimizer:
        def __init__(self):
            self.lr = 0.1

        def get_lr(self):
            return self.lr

        def set_lr(self, lr):
            self.lr = lr

    class Args:
        lr_schedule = "step"
        lr_gamma = 0.1
        epochs = 3
        lr = 0.1
        min_lr = 0.0

    opt = Optimizer()
    assert cifar.apply_lr_schedule(opt, Args(), 1, {2}) == pytest.approx(0.1)
    assert cifar.apply_lr_schedule(opt, Args(), 2, {2}) == pytest.approx(0.01)


def test_apply_lr_cosine_schedule():
    cifar = load_example("cifar10_cnn.py")

    class Optimizer:
        def __init__(self):
            self.lr = 0.1

        def get_lr(self):
            return self.lr

        def set_lr(self, lr):
            self.lr = lr

    class Args:
        lr_schedule = "cosine"
        lr_gamma = 0.1
        epochs = 3
        lr = 0.1
        min_lr = 0.01

    opt = Optimizer()
    assert cifar.apply_lr_schedule(opt, Args(), 1, set()) == pytest.approx(0.1)
    assert cifar.apply_lr_schedule(opt, Args(), 3, set()) == pytest.approx(0.01)


@pytest.mark.skipif(not orthon_cuda_available(), reason="CUDA build required")
def test_minivgg_forward_shape_cuda():
    cifar = load_example("cifar10_cnn.py")
    model = cifar.CIFAR10MiniVGG(hidden=32)
    model.to("cuda")
    x = orthon.full([2, 3, 32, 32], 0.1).to("cuda")

    logits = model(x)

    assert list(logits.shape) == [2, 10]
    assert logits.device.is_cuda()


def _run_epoch_and_evaluate(device):
    cifar = load_example("cifar10_cnn.py")
    orthon.manual_seed(7)
    model = cifar.CIFAR10CNN(hidden=4, channels=2)
    model.to(device)
    images = torch.linspace(-0.25, 0.25, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    batches = [(images, torch.tensor([0, 1]))] * 2
    optimizer = SGD(model.parameters(), lr=0.01)
    bias = list(model.parameters())[-1]
    before = bias.to("cpu").tolist()

    initial = cifar.evaluate(model, batches, max_batches=1, device=device)
    train = cifar.run_epoch(model, batches, optimizer, max_batches=1, device=device)
    final = cifar.evaluate(model, batches, max_batches=1, device=device)

    for result in (initial, train, final):
        assert result["samples"] == 2
        assert math.isfinite(result["loss"])
        assert 0 <= result["accuracy"] <= 1
    assert initial["batches"] == final["batches"] == 1
    assert bias.to("cpu").tolist() != before


@pytest.mark.skipif(torch is None, reason="torch is required for raw loader batches")
def test_run_epoch_and_evaluate_cpu():
    _run_epoch_and_evaluate("cpu")


@pytest.mark.skipif(
    torch is None or not orthon_cuda_available(),
    reason="CUDA build and torch are required",
)
def test_run_epoch_and_evaluate_cuda():
    _run_epoch_and_evaluate("cuda")
