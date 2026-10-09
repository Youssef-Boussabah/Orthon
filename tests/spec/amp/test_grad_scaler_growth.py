import numpy as np

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam


def test_scale_adapts_in_step_and_backoff_restarts_consecutive_growth(
    amp_model, model_gradient,
):
    optimizer = MixedPrecisionOptimizer(amp_model, Adam, lr=.125)
    scaler = GradScaler(init_scale=8, growth_factor=4, backoff_factor=.25, growth_interval=2)
    sequence = [(8., 8., 1, False), (8., 32., 2, False),
                (8., 32., 3, False), (float("inf"), 8., 3, True),
                (8., 8., 4, False), (8., 32., 5, False)]
    for gradient, expected_scale, expected_steps, nonfinite in sequence:
        optimizer.zero_grad()
        model_gradient(amp_model.weight, np.full((2, 2), gradient))
        assert scaler.step(optimizer) is None
        assert scaler.scale_factor == expected_scale
        assert scaler.found_inf is nonfinite
        assert optimizer.state_dict()["inner"]["t"] == expected_steps
        for _ in range(2):
            assert scaler.update() is None
            assert scaler.scale_factor == expected_scale and scaler.found_inf is False


def test_unscale_detection_alone_does_not_backoff_until_step(amp_model, model_gradient):
    optimizer = MixedPrecisionOptimizer(amp_model, Adam, lr=.125)
    scaler = GradScaler(init_scale=8, backoff_factor=.25)
    model_gradient(amp_model.bias, [1., float("nan")])
    assert scaler.unscale_(optimizer) is None
    assert scaler.found_inf is True and scaler.scale_factor == 8
    scaler.step(optimizer)
    assert scaler.scale_factor == 2 and scaler.found_inf is True
    assert optimizer.state_dict()["inner"]["t"] == 0
