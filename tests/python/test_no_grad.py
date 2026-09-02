import pytest

import orthon


def _cuda_available():
    if not orthon.cuda_enabled():
        return False
    try:
        orthon.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def test_grad_enabled_default_and_set_restore():
    previous = orthon.is_grad_enabled()
    try:
        orthon.set_grad_enabled(True)
        assert orthon.is_grad_enabled() is True
        orthon.set_grad_enabled(False)
        assert orthon.is_grad_enabled() is False
    finally:
        orthon.set_grad_enabled(previous)


def test_no_grad_restores_previous_state():
    orthon.set_grad_enabled(True)
    with orthon.no_grad():
        assert orthon.is_grad_enabled() is False
    assert orthon.is_grad_enabled() is True


def test_nested_no_grad_restores_outer_state():
    orthon.set_grad_enabled(True)
    with orthon.no_grad():
        assert orthon.is_grad_enabled() is False
        with orthon.no_grad():
            assert orthon.is_grad_enabled() is False
        assert orthon.is_grad_enabled() is False
    assert orthon.is_grad_enabled() is True


def test_no_grad_disables_basic_graph_creation():
    a = orthon.full([2, 3], 2.0)
    b = orthon.full([3], 1.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    with orthon.no_grad():
        add_out = a + b
        mul_out = a * a
        matmul_out = orthon.matmul(a, orthon.ones([3, 4]))
        relu_out = orthon.relu(a)
        reshaped = orthon.reshape(a, [6])

    for out in [add_out, mul_out, matmul_out, relu_out, reshaped]:
        assert out.requires_grad is False
        assert out._has_grad_fn is False


def test_no_grad_linear_and_cross_entropy_outputs_are_plain_tensors():
    from orthon.nn import Linear

    layer = Linear(3, 2)
    x = orthon.full([4, 3], 1.0)
    x.requires_grad_(True)

    with orthon.no_grad():
        logits = layer(x)
        loss = orthon.cross_entropy(logits, [0, 1, 0, 1])

    assert layer.weight.requires_grad is True
    assert layer.bias.requires_grad is True
    assert logits.requires_grad is False
    assert logits._has_grad_fn is False
    assert loss.requires_grad is False
    assert loss._has_grad_fn is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


def test_eval_forward_with_no_grad_does_not_build_graph():
    from orthon.nn import Linear, ReLU, Sequential

    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    model.eval()
    x = orthon.full([2, 3], 1.0)

    with orthon.no_grad():
        out = model(x)

    assert out.requires_grad is False
    assert out._has_grad_fn is False
    for param in model.parameters():
        assert param.requires_grad is True
        assert param.grad is None


def test_grad_mode_reenabled_after_no_grad_builds_graph_again():
    a = orthon.full([2], 1.0)
    b = orthon.full([2], 2.0)
    a.requires_grad_(True)
    b.requires_grad_(True)

    with orthon.no_grad():
        out = a + b
    assert out.requires_grad is False

    out = a + b
    assert out.requires_grad is True
    assert out._has_grad_fn is True


@requires_cuda
def test_cuda_no_grad_linear_cross_entropy_stays_cuda_and_plain():
    from orthon.nn import Linear

    layer = Linear(3, 2).to("cuda")
    x = orthon.full([4, 3], 1.0).to("cuda")
    x.requires_grad_(True)

    with orthon.no_grad():
        logits = layer(x)
        loss = orthon.cross_entropy(logits, [0, 1, 0, 1])

    assert logits.device.is_cuda()
    assert loss.device.is_cuda()
    assert logits.requires_grad is False
    assert logits._has_grad_fn is False
    assert loss.requires_grad is False
    assert loss._has_grad_fn is False
