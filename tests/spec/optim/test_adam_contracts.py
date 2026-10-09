import math

import numpy as np
import pytest

from orthon.optim import Adam


@pytest.mark.parametrize("values,gradient", [
    (3., -2.),
    ([3., -2., 1.], [2., -4., 0.]),
    ([[3., -2.], [1., 4.]], [[2., -4.], [0., 1.]]),
])
@pytest.mark.parametrize("betas", [(0., 0.), (0.5, 0.75)])
@pytest.mark.parametrize("eps", [0.25, 1.])
def test_first_update_uses_bias_correction_and_epsilon_outside_sqrt(
    adam_parameter, assign_gradient, assert_tensor, values, gradient, betas, eps,
):
    p = adam_parameter(values)
    assign_gradient(p, gradient)
    opt = Adam([p], lr=0.25, betas=betas, eps=eps)
    assert opt.step() is None
    g = np.asarray(gradient)
    assert_tensor(p, np.asarray(values) - 0.25 * g / (np.abs(g) + eps))
    assert p.dtype == "float32"
    assert p.is_leaf and p.requires_grad
    state = opt.state_dict()
    assert_tensor(state["m"][0], (1 - betas[0]) * g)
    assert_tensor(state["v"][0], (1 - betas[1]) * g * g)


def test_second_update_and_exported_moments_have_analytic_values(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3., -2.])
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    assign_gradient(p, [2., -4.])
    opt.step()
    assert_tensor(p, [3. - 1 / 6, -1.8])
    assign_gradient(p, [4., 2.])
    opt.step()
    state = opt.state_dict()
    assert state["t"] == 2
    assert_tensor(state["m"][0], [2.5, 0.])
    assert_tensor(state["v"][0], [4.75, 4.])
    expected = 3. - 1 / 6 - 0.25 * (2.5 / 0.75) / (math.sqrt(4.75 / 0.4375) + 1.)
    assert_tensor(p, [expected, -1.8])


def test_zero_betas_remove_history_but_learning_rate_still_scales_update(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3., -2.])
    opt = Adam([p], lr=0.25, betas=(0., 0.), eps=1.)
    assign_gradient(p, [2., -4.])
    opt.step()
    opt.set_lr(0.5)
    assign_gradient(p, [-4., 2.])
    opt.step()
    assert_tensor(p, [3. - 1 / 6 + 0.4, -1.8 - 1 / 3])
    assert_tensor(opt.state_dict()["m"][0], [-4., 2.])
    assert_tensor(opt.state_dict()["v"][0], [16., 4.])


@pytest.mark.parametrize("values", [2., [2., -4.], [[2., -4.]], [[[2., -4.]]]])
@pytest.mark.parametrize("decay_bias", [False, True])
def test_decay_does_not_enter_moment_estimation(
    adam_parameter, assign_gradient, assert_tensor, values, decay_bias,
):
    initial = np.asarray(values)
    p = adam_parameter(values)
    assign_gradient(p, np.full(initial.shape, 2.))
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.,
               weight_decay=0.5, decay_bias=decay_bias)
    opt.step()
    scale = 0.875 if decay_bias or initial.ndim > 1 else 1.
    assert_tensor(p, initial * scale - 1 / 6)
    assert_tensor(opt.state_dict()["m"][0], np.ones(initial.shape))
    assert_tensor(opt.state_dict()["v"][0], np.ones(initial.shape))


def test_repeated_zero_gradient_decay_and_missing_gradient_state(
    adam_parameter, assign_gradient, assert_tensor,
):
    zero, missing = adam_parameter([[4.]]), adam_parameter([[4.]])
    assign_gradient(zero, [[0.]])
    opt = Adam([zero, missing], lr=0.25, weight_decay=0.5)
    for expected in (3.5, 3.0625):
        opt.step()
        assert_tensor(zero, [[expected]])
        assert_tensor(missing, [[4.]])
        state = opt.state_dict()
        for name in ("m", "v"):
            assert_tensor(state[name][0], [[0.]])
            assert state[name][1] is None


def test_zero_gradient_after_nonzero_gradient_keeps_momentum_update(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter(3.)
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    assign_gradient(p, 2.)
    opt.step()
    assign_gradient(p, 0.)
    opt.step()
    assert_tensor(opt.state_dict()["m"][0], 0.5)
    assert_tensor(opt.state_dict()["v"][0], 0.75)
    assert_tensor(p, 3. - 1 / 6 - 0.25 * (0.5 / 0.75) / (math.sqrt(0.75 / 0.4375) + 1.))


def test_delayed_first_gradient_uses_optimizer_global_step(
    adam_parameter, assign_gradient, assert_tensor,
):
    early, delayed = adam_parameter([3.]), adam_parameter([3.])
    opt = Adam([early, delayed], lr=0.25, betas=(0.5, 0.75), eps=1.)
    assign_gradient(early, [2.])
    opt.step()
    early.zero_grad()
    assign_gradient(delayed, [2.])
    opt.step()
    state = opt.state_dict()
    assert state["t"] == 2
    assert_tensor(state["m"][0], [1.])
    assert_tensor(state["v"][0], [1.])
    assert_tensor(early, [3. - 1 / 6])
    assert_tensor(delayed, [3. - 0.25 * (1 / 0.75) / (math.sqrt(1 / 0.4375) + 1.)])


def test_no_gradient_steps_advance_counter_without_allocating_moments(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    opt.step()
    opt.step()
    state = opt.state_dict()
    assert state["t"] == 2
    assert state["m"] == state["v"] == [None]
    assign_gradient(p, [2.])
    opt.step()
    assert_tensor(p, [3. - 0.25 * (1 / 0.875) / (math.sqrt(1 / 0.578125) + 1.)])


def test_multiple_parameter_shapes_keep_distinct_numerical_histories(
    adam_parameter, assign_gradient, assert_tensor,
):
    vector, matrix = adam_parameter([3., -2.]), adam_parameter([[4.]])
    opt = Adam([vector, matrix], lr=0.25, betas=(0.5, 0.75), eps=1.)
    assign_gradient(vector, [2., -4.])
    assign_gradient(matrix, [[-2.]])
    opt.step()
    assign_gradient(vector, [4., 2.])
    assign_gradient(matrix, [[-2.]])
    opt.step()
    state = opt.state_dict()
    assert_tensor(state["m"][0], [2.5, 0.])
    assert_tensor(state["v"][0], [4.75, 4.])
    assert_tensor(state["m"][1], [[-1.5]])
    assert_tensor(state["v"][1], [[1.75]])
    assert_tensor(matrix, [[4. + 1 / 3]])


def test_zero_learning_rate_still_updates_adam_history(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    opt = Adam([p], lr=0., betas=(0.5, 0.75), eps=1., weight_decay=0.5, decay_bias=True)
    assign_gradient(p, [2.])
    opt.step()
    assign_gradient(p, [4.])
    opt.step()
    assert_tensor(p, [3.])
    assert opt.state_dict()["t"] == 2
    assert_tensor(opt.state_dict()["m"][0], [2.5])
    assert_tensor(opt.state_dict()["v"][0], [4.75])
