from tests.support.capabilities import orthon_cuda_available
from tests.support.capabilities import orthon_cuda_available as _cuda_available
import numpy as np
import pytest

import orthon as ortho
import orthon.functional as F
import orthon.nn as nn

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_available()), reason="CUDA not available")


def _tensor(values, device="cpu"):
    t = ortho.from_numpy(np.asarray(values, dtype=np.float32))
    if device == "cuda":
        t = t.to("cuda")
    return t


def _targets(values, device="cpu"):
    t = ortho.from_numpy(np.asarray(values, dtype=np.int64))
    if device == "cuda":
        t = t.to("cuda")
    return t


def _assert_close(actual, expected, rel=1e-5, abs=1e-5):
    if actual.device.is_cuda():
        actual = actual.to("cpu")
    assert list(actual.shape) == list(expected.shape)
    assert actual.tolist() == pytest.approx(
        expected.detach().cpu().flatten().tolist(), rel=rel, abs=abs
    )


def _torch_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores + mask
    return torch.softmax(scores, dim=-1) @ v


def test_scaled_dot_product_attention_cpu_forward_matches_torch():
    rng = np.random.default_rng(0)
    qv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    kv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    vv = rng.normal(size=(2, 3, 4)).astype(np.float32)

    y = F.scaled_dot_product_attention(_tensor(qv), _tensor(kv), _tensor(vv))
    expected = _torch_attention(torch.tensor(qv), torch.tensor(kv), torch.tensor(vv))
    _assert_close(y, expected)


def test_scaled_dot_product_attention_cpu_backward_matches_torch():
    rng = np.random.default_rng(1)
    qv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    kv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    vv = rng.normal(size=(2, 3, 4)).astype(np.float32)

    q = _tensor(qv)
    k = _tensor(kv)
    v = _tensor(vv)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    F.scaled_dot_product_attention(q, k, v).sum().backward()

    tq = torch.tensor(qv, requires_grad=True)
    tk = torch.tensor(kv, requires_grad=True)
    tv = torch.tensor(vv, requires_grad=True)
    _torch_attention(tq, tk, tv).sum().backward()

    _assert_close(q.grad, tq.grad, rel=2e-5, abs=2e-5)
    _assert_close(k.grad, tk.grad, rel=2e-5, abs=2e-5)
    _assert_close(v.grad, tv.grad, rel=2e-5, abs=2e-5)


def test_scaled_dot_product_attention_additive_mask_matches_torch():
    rng = np.random.default_rng(2)
    qv = rng.normal(size=(1, 3, 4)).astype(np.float32)
    kv = rng.normal(size=(1, 3, 4)).astype(np.float32)
    vv = rng.normal(size=(1, 3, 4)).astype(np.float32)
    maskv = np.asarray([[[0.0, -1.0e9, 0.0]]], dtype=np.float32)

    y = ortho.scaled_dot_product_attention(
        _tensor(qv), _tensor(kv), _tensor(vv), mask=_tensor(maskv)
    )
    expected = _torch_attention(
        torch.tensor(qv), torch.tensor(kv), torch.tensor(vv), torch.tensor(maskv)
    )
    _assert_close(y, expected)


def test_self_attention_output_shape():
    layer = nn.SelfAttention(6)
    x = _tensor(np.random.default_rng(3).normal(size=(2, 4, 6)).astype(np.float32))
    y = layer(x)
    assert list(y.shape) == [2, 4, 6]


def test_tiny_attention_model_cpu_forward_backward_smoke():
    embed = nn.Embedding(16, 6)
    attn = nn.SelfAttention(6)
    norm = nn.LayerNorm(6)
    hidden = nn.Linear(6, 8)
    out = nn.Linear(8, 2)
    x = _targets([[1, 7, 2, 3], [4, 5, 6, 8]])
    y = _targets([1, 0])

    h = attn(embed(x))
    h = norm(h.mean(dim=1))
    logits = out(ortho.gelu(hidden(h)))
    loss = ortho.nll_loss(ortho.log_softmax(logits, dim=-1), y)
    loss.backward()

    assert embed.weight.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert hidden.weight.grad is not None
    assert out.weight.grad is not None


@requires_cuda
def test_scaled_dot_product_attention_cuda_forward_matches_torch():
    rng = np.random.default_rng(4)
    qv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    kv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    vv = rng.normal(size=(2, 3, 4)).astype(np.float32)

    y = F.scaled_dot_product_attention(
        _tensor(qv, "cuda"), _tensor(kv, "cuda"), _tensor(vv, "cuda")
    )
    expected = _torch_attention(
        torch.tensor(qv, device="cuda"),
        torch.tensor(kv, device="cuda"),
        torch.tensor(vv, device="cuda"),
    )
    assert y.device.is_cuda()
    _assert_close(y, expected, rel=2e-5, abs=2e-5)


@requires_cuda
def test_scaled_dot_product_attention_cuda_backward_matches_torch():
    rng = np.random.default_rng(5)
    qv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    kv = rng.normal(size=(2, 3, 4)).astype(np.float32)
    vv = rng.normal(size=(2, 3, 4)).astype(np.float32)

    q = _tensor(qv, "cuda")
    k = _tensor(kv, "cuda")
    v = _tensor(vv, "cuda")
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    ortho.scaled_dot_product_attention(q, k, v).sum().backward()

    tq = torch.tensor(qv, device="cuda", requires_grad=True)
    tk = torch.tensor(kv, device="cuda", requires_grad=True)
    tv = torch.tensor(vv, device="cuda", requires_grad=True)
    _torch_attention(tq, tk, tv).sum().backward()

    assert q.grad.device.is_cuda()
    assert k.grad.device.is_cuda()
    assert v.grad.device.is_cuda()
    _assert_close(q.grad, tq.grad, rel=3e-5, abs=3e-5)
    _assert_close(k.grad, tk.grad, rel=3e-5, abs=3e-5)
    _assert_close(v.grad, tv.grad, rel=3e-5, abs=3e-5)


@requires_cuda
def test_self_attention_module_to_cuda_moves_parameters():
    layer = nn.SelfAttention(6).to("cuda")
    assert layer.q_proj.weight.device.is_cuda()
    x = _tensor(np.random.default_rng(6).normal(size=(2, 4, 6)).astype(np.float32), "cuda")
    y = layer(x)
    assert y.device.is_cuda()
    assert list(y.shape) == [2, 4, 6]


@requires_cuda
def test_tiny_attention_model_cuda_forward_backward_smoke():
    embed = nn.Embedding(16, 6).to("cuda")
    attn = nn.SelfAttention(6).to("cuda")
    norm = nn.LayerNorm(6).to("cuda")
    hidden = nn.Linear(6, 8).to("cuda")
    out = nn.Linear(8, 2).to("cuda")
    x = _targets([[1, 7, 2, 3], [4, 5, 6, 8]], "cuda")
    y = _targets([1, 0], "cuda")

    h = attn(embed(x))
    h = norm(h.mean(dim=1))
    logits = out(ortho.gelu(hidden(h)))
    loss = ortho.nll_loss(ortho.log_softmax(logits, dim=-1), y)
    loss.backward()

    assert loss.device.is_cuda()
    assert embed.weight.grad.device.is_cuda()
    assert attn.q_proj.weight.grad.device.is_cuda()
    assert hidden.weight.grad.device.is_cuda()
    assert out.weight.grad.device.is_cuda()

def test_multihead_self_attention_forward_shape_cpu():
    x = ortho.from_numpy(np.random.randn(2,4,8).astype(np.float32))
    layer = nn.MultiHeadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0)
    y = layer(x)
    assert list(y.shape) == [2,4,8]

def test_multihead_self_attention_forward_shape_cuda():
    if not orthon_cuda_available():
        pytest.skip("CUDA not available")

    x = ortho.from_numpy(np.random.randn(2, 4, 8).astype(np.float32)).to("cuda")
    layer = nn.MultiHeadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0).to("cuda")

    y = layer(x)

    assert list(y.shape) == [2, 4, 8]
    assert "cuda" in str(y.device)

def test_multihead_self_attention_backward_cpu():
    x = ortho.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    x.requires_grad_(True)

    layer = nn.MultiHeadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for p in layer.parameters():
        assert p.grad is not None

def test_multihead_self_attention_backward_cuda():
    if not orthon_cuda_available():
        pytest.skip("CUDA not available")

    x = ortho.from_numpy(np.random.randn(2, 4, 8).astype(np.float32)).to("cuda")
    x.requires_grad_(True)

    layer = nn.MultiHeadSelfAttention(embed_dim=8, num_heads=2, dropout=0.0).to("cuda")
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for p in layer.parameters():
        assert p.grad is not None

def test_multihead_self_attention_invalid_heads():
    with pytest.raises(ValueError):
        nn.MultiHeadSelfAttention(embed_dim=10, num_heads=3)

    with pytest.raises(ValueError):
        nn.MultiHeadSelfAttention(embed_dim=8, num_heads=0)
