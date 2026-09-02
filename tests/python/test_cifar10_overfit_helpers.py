import importlib.util
import math
import sys
from pathlib import Path

import pytest

import orthon
from orthon.nn import Linear
from orthon.optim import SGD


try:
    import torch
except ImportError:  # pragma: no cover - depends on local test environment
    torch = None


def _load_cifar_example():
    root = Path(__file__).resolve().parents[2]
    examples_dir = root / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "cifar10_cnn_example",
            examples_dir / "cifar10_cnn.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


@pytest.mark.skipif(torch is None, reason="torch is required for raw loader batches")
def test_collect_fixed_batches_keeps_exact_batch_order():
    cifar = _load_cifar_example()
    batches = [
        (torch.full((2, 4), float(idx)), torch.tensor([idx, idx + 1]))
        for idx in range(4)
    ]

    fixed = cifar.collect_fixed_batches(batches, 2)

    assert fixed == batches[:2]
    assert cifar.batch_fingerprints(fixed) == cifar.batch_fingerprints(batches[:2])


def test_flatten_nested_and_parameter_update_stats(capsys):
    cifar = _load_cifar_example()
    assert cifar.flatten_nested([[1.0, -2.0], (3.0, [4.0])]) == [1.0, -2.0, 3.0, 4.0]

    model = Linear(2, 1)
    before = cifar.snapshot_parameters(model)
    params = list(model.parameters())
    params[0].set_grad(orthon.full(list(params[0].shape), 0.25))
    params[0].add_(params[0].grad, alpha=-0.1)

    rows = cifar.debug_parameter_updates(model, before)
    output = capsys.readouterr().out

    assert "debug all parameter updates:" in output
    assert rows[0]["shape"] == list(params[0].shape)
    assert rows[0]["grad_max"] == pytest.approx(0.25)
    assert rows[0]["update_max"] == pytest.approx(0.025, abs=1e-6)


def test_minivgg_forward_shape_cpu():
    cifar = _load_cifar_example()
    model = cifar.CIFAR10MiniVGG(hidden=32)
    x = orthon.full([2, 3, 32, 32], 0.1)

    logits = model(x)

    assert list(logits.shape) == [2, 10]


def test_make_model_constructs_minivgg():
    cifar = _load_cifar_example()

    class Args:
        model = "minivgg"
        hidden = 32
        channels = 8
        vgg_base_channels = 32
        init_scale = 0.5

    model = cifar.make_model(Args())

    assert isinstance(model, cifar.CIFAR10MiniVGG)


def test_parse_lr_step_epochs():
    cifar = _load_cifar_example()

    assert cifar.parse_lr_step_epochs("") == set()
    assert cifar.parse_lr_step_epochs("15,25") == {15, 25}
    assert cifar.parse_lr_step_epochs(" 5, 10, ") == {5, 10}


def test_apply_lr_step_schedule():
    cifar = _load_cifar_example()

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
    cifar = _load_cifar_example()

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


def test_profile_percent_sums_close_to_100():
    cifar = _load_cifar_example()
    profile = {key: 0.0 for key in cifar.PROFILE_KEYS}
    profile["data_loading"] = 1.0
    profile["forward"] = 2.0
    profile["backward"] = 7.0

    percentages = cifar.profile_percent(profile)

    assert sum(percentages.values()) == pytest.approx(100.0)
    assert percentages["backward"] == pytest.approx(70.0)


def test_maybe_synchronize_noops_on_cpu():
    cifar = _load_cifar_example()

    assert cifar.maybe_synchronize("cpu") is None


@pytest.mark.skipif(not orthon.cuda_enabled(), reason="CUDA build required")
def test_minivgg_forward_shape_cuda():
    cifar = _load_cifar_example()
    model = cifar.CIFAR10MiniVGG(hidden=32)
    model.to("cuda")
    x = orthon.full([2, 3, 32, 32], 0.1).to("cuda")

    logits = model(x)

    assert list(logits.shape) == [2, 10]
    assert logits.device.is_cuda()


@pytest.mark.skipif(torch is None, reason="torch is required for raw loader batches")
def test_fixed_batch_overfit_loop_runs_on_cpu_without_crashing():
    cifar = _load_cifar_example()
    batches = [
        (torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]), torch.tensor([0, 1])),
    ]
    model = Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=0.01)

    initial = cifar.evaluate_fixed_batches(model, batches, "cpu")
    for _ in range(2):
        train = cifar.run_epoch_fixed_batches(model, batches, optimizer, "cpu")
    final = cifar.evaluate_fixed_batches(model, batches, "cpu")

    assert initial["samples"] == 2
    assert train["samples"] == 2
    assert final["samples"] == 2
    assert math.isfinite(initial["loss"])
    assert math.isfinite(train["loss"])
    assert math.isfinite(final["loss"])


@pytest.mark.skipif(
    torch is None or not orthon.cuda_enabled(),
    reason="CUDA build and torch are required",
)
def test_fixed_batch_overfit_loop_runs_on_cuda_without_crashing():
    cifar = _load_cifar_example()
    batches = [
        (torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]), torch.tensor([0, 1])),
    ]
    model = Linear(4, 2)
    model.to("cuda")
    optimizer = SGD(model.parameters(), lr=0.01)

    result = cifar.run_epoch_fixed_batches(model, batches, optimizer, "cuda")

    assert result["samples"] == 2
    assert math.isfinite(result["loss"])
