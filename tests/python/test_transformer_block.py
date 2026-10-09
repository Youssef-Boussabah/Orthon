from tests.support.capabilities import orthon_cuda_available as _cuda_available
import numpy as np
import pytest

import orthon as ortho
import orthon.nn as nn




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _randn(shape, device="cpu"):
    t = ortho.from_numpy(np.random.default_rng(42).normal(size=shape).astype(np.float32))
    return t.to("cuda") if device == "cuda" else t


# Construction validation.

def test_constructor_rejects_bad_embed_dim():
    with pytest.raises((ValueError, RuntimeError)):
        nn.TransformerEncoderBlock(embed_dim=0, num_heads=2)


def test_constructor_rejects_embed_not_divisible_by_heads():
    with pytest.raises((ValueError, RuntimeError)):
        nn.TransformerEncoderBlock(embed_dim=7, num_heads=3)


def test_constructor_rejects_bad_num_heads():
    with pytest.raises((ValueError, RuntimeError)):
        nn.TransformerEncoderBlock(embed_dim=8, num_heads=0)


# CPU forward.

def test_forward_shape_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2)
    x = _randn((2, 4, 8))
    y = block(x)
    assert list(y.shape) == [2, 4, 8]


def test_forward_shape_single_head_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=1)
    x = _randn((1, 6, 8))
    y = block(x)
    assert list(y.shape) == [1, 6, 8]


def test_forward_larger_embed_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=32, num_heads=4, mlp_ratio=2.0)
    x = _randn((4, 8, 32))
    y = block(x)
    assert list(y.shape) == [4, 8, 32]


def test_forward_with_dropout_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2, dropout=0.1)
    block.eval()
    x = _randn((2, 4, 8))
    y = block(x)
    assert list(y.shape) == [2, 4, 8]


# CPU backward.

def test_backward_x_grad_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2)
    x = _randn((2, 4, 8))
    x.requires_grad_(True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None


def test_backward_parameter_grads_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2)
    x = _randn((2, 4, 8))
    x.requires_grad_(True)
    block(x).sum().backward()
    params = list(block.parameters())
    assert len(params) > 0
    grads_computed = sum(1 for p in params if p.grad is not None)
    assert grads_computed > 0, "Expected at least some parameter gradients"


def test_backward_all_params_have_grad_cpu():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2)
    x = _randn((2, 4, 8))
    x.requires_grad_(True)
    block(x).sum().backward()
    for p in block.parameters():
        assert p.grad is not None, f"Parameter shape {list(p.shape)} has no grad"


# CUDA forward.

@requires_cuda
def test_forward_shape_cuda():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2).to("cuda")
    x = _randn((2, 4, 8), "cuda")
    y = block(x)
    assert list(y.shape) == [2, 4, 8]
    assert y.device.is_cuda()


@requires_cuda
def test_forward_larger_embed_cuda():
    block = nn.TransformerEncoderBlock(embed_dim=32, num_heads=4).to("cuda")
    x = _randn((4, 8, 32), "cuda")
    y = block(x)
    assert list(y.shape) == [4, 8, 32]
    assert y.device.is_cuda()


# CUDA backward.

@requires_cuda
def test_backward_x_grad_cuda():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2).to("cuda")
    x = _randn((2, 4, 8), "cuda")
    x.requires_grad_(True)
    block(x).sum().backward()
    assert x.grad is not None
    assert x.grad.device.is_cuda()


@requires_cuda
def test_backward_all_params_have_grad_cuda():
    block = nn.TransformerEncoderBlock(embed_dim=8, num_heads=2).to("cuda")
    x = _randn((2, 4, 8), "cuda")
    x.requires_grad_(True)
    block(x).sum().backward()
    for p in block.parameters():
        assert p.grad is not None
        assert p.grad.device.is_cuda()


# Tiny model smoke.

def _make_int64(arr, device="cpu"):
    t = ortho.from_numpy(np.asarray(arr, dtype=np.int64))
    return t.to("cuda") if device == "cuda" else t


def _tiny_model_forward(device="cpu"):
    B, T, E, V, C = 2, 4, 8, 16, 2
    token_embed = nn.Embedding(V, E)
    pos_embed = nn.Embedding(T, E)
    block = nn.TransformerEncoderBlock(embed_dim=E, num_heads=2)
    ln = nn.LayerNorm(E)
    classifier = nn.Linear(E, C)

    if device == "cuda":
        token_embed.to("cuda")
        pos_embed.to("cuda")
        block.to("cuda")
        ln.to("cuda")
        classifier.to("cuda")

    tokens = _make_int64([[1, 3, 2, 5], [7, 0, 4, 6]], device)
    pos_ids = _make_int64(np.tile(np.arange(T, dtype=np.int64), (B, 1)), device)
    targets = _make_int64([0, 1], device)

    x = ortho.embedding(tokens, token_embed.weight) + ortho.embedding(pos_ids, pos_embed.weight)
    x = block(x)
    x = ln(x)
    pooled = x.mean(dim=1)
    logits = classifier(pooled)
    log_probs = ortho.log_softmax(logits, dim=-1)
    loss = ortho.nll_loss(log_probs, targets, reduction="mean")
    return loss, token_embed, pos_embed, block, ln, classifier


def test_tiny_model_forward_backward_cpu():
    loss, te, pe, block, ln, clf = _tiny_model_forward("cpu")
    loss.backward()
    assert te.weight.grad is not None
    assert pe.weight.grad is not None
    for p in block.parameters():
        assert p.grad is not None
    for p in clf.parameters():
        assert p.grad is not None


@requires_cuda
def test_tiny_model_forward_backward_cuda():
    loss, te, pe, block, ln, clf = _tiny_model_forward("cuda")
    assert loss.device.is_cuda()
    loss.backward()
    assert te.weight.grad is not None
    assert te.weight.grad.device.is_cuda()
    for p in block.parameters():
        assert p.grad is not None
        assert p.grad.device.is_cuda()
