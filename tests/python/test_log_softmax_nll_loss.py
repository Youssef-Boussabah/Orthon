import numpy as np
import pytest

import orthon as ortho
import orthon.functional as F
import orthon.nn as nn

torch = pytest.importorskip("torch")


def _cuda_available():
    if not torch.cuda.is_available():
        return False
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _tensor(values, device="cpu"):
    t = ortho.from_numpy(np.asarray(values, dtype=np.float32))
    if device == "cuda":
        t = t.to("cuda")
    return t


def _targets(values, device="cpu", dtype=np.int64):
    t = ortho.from_numpy(np.asarray(values, dtype=dtype))
    if device == "cuda":
        t = t.to("cuda")
    return t


def _assert_tensor_close(actual, expected, rel=1e-5, abs=1e-5):
    if actual.device.is_cuda():
        actual = actual.to("cpu")
    assert list(actual.shape) == list(expected.shape)
    assert actual.tolist() == pytest.approx(
        expected.detach().cpu().flatten().tolist(), rel=rel, abs=abs
    )


def test_log_softmax_cpu_forward_dim_last_matches_torch():
    values = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    y = ortho.log_softmax(_tensor(values), dim=-1)
    expected = torch.nn.functional.log_softmax(torch.tensor(values), dim=-1)
    _assert_tensor_close(y, expected)


def test_log_softmax_functional_and_tensor_method():
    x = _tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    expected = torch.nn.functional.log_softmax(
        torch.tensor(x.tolist(), dtype=torch.float32).reshape(2, 3), dim=1
    )
    _assert_tensor_close(F.log_softmax(x, dim=1), expected)
    _assert_tensor_close(x.log_softmax(dim=1), expected)


def test_log_softmax_cpu_forward_dim0_matches_torch():
    values = np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32)
    y = ortho.log_softmax(_tensor(values), dim=0)
    expected = torch.nn.functional.log_softmax(torch.tensor(values), dim=0)
    _assert_tensor_close(y, expected)


def test_log_softmax_cpu_backward_matches_torch():
    values = np.random.default_rng(2).normal(size=(4, 5)).astype(np.float32)
    x = _tensor(values)
    x.requires_grad_(True)
    ortho.log_softmax(x, dim=-1).sum().backward()

    tx = torch.tensor(values, requires_grad=True)
    torch.nn.functional.log_softmax(tx, dim=-1).sum().backward()

    _assert_tensor_close(x.grad, tx.grad)


def test_nll_loss_cpu_forward_mean_sum_and_none_match_torch():
    values = np.random.default_rng(3).normal(size=(4, 5)).astype(np.float32)
    log_probs = ortho.log_softmax(_tensor(values), dim=-1)
    targets = [0, 3, 1, 4]

    tlog_probs = torch.nn.functional.log_softmax(torch.tensor(values), dim=-1)
    tt = torch.tensor(targets, dtype=torch.long)

    for reduction in ("mean", "sum", "none"):
        y = ortho.nll_loss(log_probs, targets, reduction=reduction)
        expected = torch.nn.functional.nll_loss(tlog_probs, tt, reduction=reduction)
        if reduction == "none":
            _assert_tensor_close(y, expected)
        else:
            assert y.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-5)


def test_nll_loss_accepts_numpy_int32_targets():
    log_probs = F.log_softmax(_tensor([[1.0, 2.0], [3.0, 1.0]]), dim=-1)
    targets = np.asarray([1, 0], dtype=np.int32)
    y = F.nll_loss(log_probs, targets, reduction="sum")
    expected = torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(torch.tensor([[1.0, 2.0], [3.0, 1.0]]), dim=-1),
        torch.tensor([1, 0], dtype=torch.long),
        reduction="sum",
    )
    assert y.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-5)


def test_nll_loss_cpu_backward_matches_torch():
    values = np.random.default_rng(4).normal(size=(4, 5)).astype(np.float32)
    log_probs = _tensor(values)
    log_probs.requires_grad_(True)
    ortho.nll_loss(log_probs, [0, 3, 3, 1], reduction="mean").backward()

    tlog_probs = torch.tensor(values, requires_grad=True)
    torch.nn.functional.nll_loss(
        tlog_probs, torch.tensor([0, 3, 3, 1], dtype=torch.long), reduction="mean"
    ).backward()

    _assert_tensor_close(log_probs.grad, tlog_probs.grad)


def test_nll_loss_cpu_backward_repeated_classes_sum_matches_torch():
    values = np.random.default_rng(5).normal(size=(5, 4)).astype(np.float32)
    log_probs = _tensor(values)
    log_probs.requires_grad_(True)
    ortho.nll_loss(log_probs, [2, 2, 2, 1, 1], reduction="sum").backward()

    tlog_probs = torch.tensor(values, requires_grad=True)
    torch.nn.functional.nll_loss(
        tlog_probs, torch.tensor([2, 2, 2, 1, 1], dtype=torch.long), reduction="sum"
    ).backward()

    _assert_tensor_close(log_probs.grad, tlog_probs.grad)


def test_nll_loss_invalid_cpu_target_raises():
    log_probs = F.log_softmax(_tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]), dim=-1)
    with pytest.raises(RuntimeError):
        ortho.nll_loss(log_probs, [0, 3])
    with pytest.raises(RuntimeError):
        ortho.nll_loss(log_probs, [0, -1])


def test_embedding_linear_log_softmax_nll_loss_cpu_backward():
    embed = nn.Embedding(8, 4)
    linear = nn.Linear(4, 3)
    ids = _targets([[1, 2, 3], [2, 3, 4]])
    logits = linear(embed(ids).mean(dim=1))
    log_probs = ortho.log_softmax(logits, dim=-1)
    loss = ortho.nll_loss(log_probs, [0, 2])
    loss.backward()
    assert embed.weight.grad is not None
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None


@requires_cuda
def test_log_softmax_cuda_forward_backward_matches_torch():
    values = np.random.default_rng(6).normal(size=(4, 5)).astype(np.float32)
    x = _tensor(values, device="cuda")
    x.requires_grad_(True)
    y = ortho.log_softmax(x, dim=-1)
    y.sum().backward()

    tx = torch.tensor(values, device="cuda", requires_grad=True)
    ty = torch.nn.functional.log_softmax(tx, dim=-1)
    ty.sum().backward()

    assert y.device.is_cuda()
    assert x.grad.device.is_cuda()
    _assert_tensor_close(y, ty)
    _assert_tensor_close(x.grad, tx.grad)


@requires_cuda
def test_nll_loss_cuda_forward_backward_matches_torch():
    values = np.random.default_rng(7).normal(size=(4, 5)).astype(np.float32)
    log_probs = _tensor(values, device="cuda")
    log_probs.requires_grad_(True)
    targets = _targets([0, 3, 3, 1], device="cuda")
    loss = ortho.nll_loss(log_probs, targets, reduction="mean")
    loss.backward()

    tlog_probs = torch.tensor(values, device="cuda", requires_grad=True)
    tloss = torch.nn.functional.nll_loss(
        tlog_probs,
        torch.tensor([0, 3, 3, 1], device="cuda", dtype=torch.long),
        reduction="mean",
    )
    tloss.backward()

    assert loss.device.is_cuda()
    assert log_probs.grad.device.is_cuda()
    assert loss.to("cpu").item() == pytest.approx(tloss.item(), rel=1e-5, abs=1e-5)
    _assert_tensor_close(log_probs.grad, tlog_probs.grad)


@requires_cuda
def test_embedding_linear_log_softmax_nll_loss_cuda_backward():
    embed = nn.Embedding(8, 4).to("cuda")
    linear = nn.Linear(4, 3).to("cuda")
    ids = _targets([[1, 2, 3], [2, 3, 4]], device="cuda")
    logits = linear(embed(ids).mean(dim=1))
    log_probs = ortho.log_softmax(logits, dim=-1)
    loss = ortho.nll_loss(log_probs, _targets([0, 2], device="cuda"))
    loss.backward()

    assert loss.device.is_cuda()
    assert log_probs.device.is_cuda()
    assert embed.weight.grad.device.is_cuda()
    assert linear.weight.grad.device.is_cuda()
