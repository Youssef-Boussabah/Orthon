"""Optimizer tests: CPU correctness plus CUDA SGD parameter updates."""

from tests.support.capabilities import orthon_cuda_available as _cuda_available
import math

import pytest
import orthon as ortho




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def assert_close(actual, expected, rel=1e-5, abs_tol=1e-5):
    assert actual == pytest.approx(expected, rel=rel, abs=abs_tol)


def _fill_sequential(tensor, start=1.0):
    for i in range(int(tensor.numel)):
        if len(tensor.shape) == 1:
            tensor[i] = start + i
        else:
            cols = tensor.shape[-1]
            tensor[[i // cols, i % cols]] = start + i
    return tensor


def test_cpu_sgd_updates_parameters():
    from orthon.nn import Linear
    from orthon.optim import SGD

    layer = Linear(3, 2)
    w_before = layer.weight.tolist()
    b_before = layer.bias.tolist()

    x = _fill_sequential(ortho.zeros([4, 3]))
    targets = [0, 1, 0, 1]

    ortho.cross_entropy(layer(x), targets).backward()
    w_grad = layer.weight.grad.tolist()
    b_grad = layer.bias.grad.tolist()

    SGD(layer.parameters(), lr=0.1).step()

    assert_close(
        layer.weight.tolist(),
        [w - 0.1 * g for w, g in zip(w_before, w_grad)],
    )
    assert_close(
        layer.bias.tolist(),
        [b - 0.1 * g for b, g in zip(b_before, b_grad)],
    )


def test_sgd_get_set_lr():
    from orthon.optim import SGD

    opt = SGD([], lr=0.1)

    assert opt.get_lr() == pytest.approx(0.1)
    opt.set_lr(0.01)
    assert opt.get_lr() == pytest.approx(0.01)


def test_sgd_decoupled_weight_decay_smoke():
    from orthon.optim import SGD

    weight = ortho.full([2, 2], 1.0)
    bias = ortho.full([2], 1.0)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    weight.set_grad(ortho.zeros([2, 2]))
    bias.set_grad(ortho.zeros([2]))

    SGD([weight, bias], lr=0.1, weight_decay=0.1).step()

    assert weight.tolist() == pytest.approx([0.99] * 4)
    assert bias.tolist() == pytest.approx([1.0] * 2)


@requires_cuda
def test_cuda_add_inplace_updates_values():
    if not hasattr(ortho.Tensor, "add_"):
        pytest.skip("Tensor.add_ is not available")

    a = _fill_sequential(ortho.zeros([4])).to("cuda")
    b = _fill_sequential(ortho.zeros([4])).to("cuda")

    a.add_(b, alpha=2.0)

    assert_close(a.tolist(), [3.0, 6.0, 9.0, 12.0])
    assert a.device.is_cuda()


@requires_cuda
def test_cuda_mul_inplace_updates_values():
    if not hasattr(ortho.Tensor, "mul_"):
        pytest.skip("Tensor.mul_ is not available")

    a = _fill_sequential(ortho.zeros([4])).to("cuda")

    a.mul_(3.0)

    assert_close(a.tolist(), [3.0, 6.0, 9.0, 12.0])
    assert a.device.is_cuda()


@requires_cuda
def test_cuda_add_inplace_negative_alpha_matches_sgd_pattern():
    a = ortho.full([3], 1.0).to("cuda")
    b = ortho.full([3], 0.5).to("cuda")

    a.add_(b, alpha=-0.1)

    assert_close(a.tolist(), [0.95, 0.95, 0.95])


@requires_cuda
def test_cuda_sgd_changes_linear_parameters_after_backward():
    from orthon.nn import Linear
    from orthon.optim import SGD

    layer = Linear(4, 3).to("cuda")
    w_before = layer.weight.tolist()
    b_before = layer.bias.tolist()

    x = _fill_sequential(ortho.zeros([2, 4])).to("cuda")
    x.requires_grad_(True)
    ortho.cross_entropy(layer(x), [0, 2]).backward()

    SGD(layer.parameters(), lr=0.05).step()

    assert layer.weight.tolist() != pytest.approx(w_before)
    assert layer.bias.tolist() != pytest.approx(b_before)
    assert layer.weight.device.is_cuda()
    assert layer.bias.device.is_cuda()


@requires_cuda
def test_cuda_linear_cross_entropy_sgd_step_changes_loss_and_parameters():
    from orthon.nn import Linear
    from orthon.optim import SGD

    layer = Linear(3, 2).to("cuda")
    x = _fill_sequential(ortho.zeros([4, 3]), start=0.25).to("cuda")
    targets = [0, 1, 0, 1]

    loss_before = ortho.cross_entropy(layer(x), targets).item()
    w_before = layer.weight.tolist()

    ortho.cross_entropy(layer(x), targets).backward()
    SGD(layer.parameters(), lr=0.05).step()

    loss_after = ortho.cross_entropy(layer(x), targets).item()

    assert math.isfinite(loss_before)
    assert math.isfinite(loss_after)
    assert loss_after != pytest.approx(loss_before)
    assert layer.weight.tolist() != pytest.approx(w_before)


@requires_cuda
def test_cuda_sgd_matches_cpu_sgd_for_linear():
    from orthon.nn import Linear
    from orthon.optim import SGD

    n, in_f, out_f = 3, 4, 3
    lr = 0.1
    targets = [0, 1, 2]

    cpu_layer = Linear(in_f, out_f)
    w_vals = cpu_layer.weight.tolist()
    b_vals = cpu_layer.bias.tolist()

    x_cpu = _fill_sequential(ortho.zeros([n, in_f]))
    ortho.cross_entropy(cpu_layer(x_cpu), targets).backward()
    SGD(cpu_layer.parameters(), lr=lr).step()

    cuda_layer = Linear(in_f, out_f)
    for i in range(in_f):
        for j in range(out_f):
            cuda_layer.weight[[i, j]] = w_vals[i * out_f + j]
    for j in range(out_f):
        cuda_layer.bias[j] = b_vals[j]
    cuda_layer.to("cuda")

    x_cuda = x_cpu.to("cuda")
    ortho.cross_entropy(cuda_layer(x_cuda), targets).backward()
    SGD(cuda_layer.parameters(), lr=lr).step()

    assert_close(cuda_layer.weight.tolist(), cpu_layer.weight.tolist(), rel=1e-4, abs_tol=1e-4)
    assert_close(cuda_layer.bias.tolist(), cpu_layer.bias.tolist(), rel=1e-4, abs_tol=1e-4)


@requires_cuda
def test_cuda_conv_relu_pool_flatten_linear_cross_entropy_sgd_step_runs():
    from orthon.nn import Conv2d, Linear, MaxPool2d, ReLU
    from orthon.optim import SGD

    n, c_in, height, width = 2, 1, 8, 8
    c_out, kernel, pool_kernel = 4, 3, 2
    classes = 3
    flat_features = c_out * (height // pool_kernel) * (width // pool_kernel)

    conv = Conv2d(c_in, c_out, kernel, padding=1).to("cuda")
    relu = ReLU()
    pool = MaxPool2d(pool_kernel, stride=pool_kernel)
    fc = Linear(flat_features, classes).to("cuda")
    opt = SGD(list(conv.parameters()) + list(fc.parameters()), lr=0.01)

    x = ortho.full([n, c_in, height, width], 0.5).to("cuda")
    x.requires_grad_(True)

    out = conv(x)
    out = relu(out)
    out = pool(out)
    out = ortho.reshape(out, [n, flat_features])
    out = fc(out)
    loss = ortho.cross_entropy(out, [0, 1])
    loss.backward()
    opt.step()

    for param in list(conv.parameters()) + list(fc.parameters()):
        assert param.device.is_cuda()
