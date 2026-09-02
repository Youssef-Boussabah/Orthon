import os
import sys

import numpy as np
import pytest

# Import helpers directly from the example script.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")))
from tiny_causal_lm import make_causal_mask, TinyCausalLM, generate

import orthon as ortho


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


# make_causal_mask.

def test_causal_mask_shape():
    mask = make_causal_mask(5)
    assert list(mask.shape) == [1, 5, 5]


def test_causal_mask_lower_triangle_zero():
    T    = 4
    mask = make_causal_mask(T)
    flat = mask.tolist()                           # 1*T*T flat list
    m    = np.array(flat, dtype=np.float32).reshape(T, T)
    for i in range(T):
        for j in range(i + 1):                    # j <= i: allowed
            assert m[i, j] == pytest.approx(0.0)


def test_causal_mask_upper_triangle_large_negative():
    T    = 4
    mask = make_causal_mask(T)
    flat = mask.tolist()
    m    = np.array(flat, dtype=np.float32).reshape(T, T)
    for i in range(T):
        for j in range(i + 1, T):                 # j > i: masked
            assert m[i, j] == pytest.approx(-1e9)


def test_causal_mask_dtype_and_device_cpu():
    mask = make_causal_mask(3, device="cpu")
    assert not mask.device.is_cuda()


@requires_cuda
def test_causal_mask_on_cuda():
    mask = make_causal_mask(4, device="cuda")
    assert mask.device.is_cuda()
    assert list(mask.shape) == [1, 4, 4]


# TinyCausalLM construction.

def test_constructor_rejects_bad_vocab():
    with pytest.raises((ValueError, RuntimeError)):
        TinyCausalLM(vocab_size=0, block_size=8, embed_dim=8, num_heads=2)


def test_constructor_rejects_bad_block_size():
    with pytest.raises((ValueError, RuntimeError)):
        TinyCausalLM(vocab_size=16, block_size=0, embed_dim=8, num_heads=2)


def test_constructor_rejects_embed_not_divisible_by_heads():
    with pytest.raises((ValueError, RuntimeError)):
        TinyCausalLM(vocab_size=16, block_size=8, embed_dim=7, num_heads=3)


# TinyCausalLM CPU forward.

def _int_tensor(arr, device="cpu"):
    t = ortho.from_numpy(np.asarray(arr, dtype=np.int64))
    return t.to("cuda") if device == "cuda" else t


def test_forward_shape_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    idx   = _int_tensor([[1, 3, 2, 5, 0, 7, 4, 6],
                          [2, 0, 1, 4, 3, 6, 5, 7]])
    mask  = make_causal_mask(8)
    lp    = model(idx, mask=mask)
    assert list(lp.shape) == [2, 8, 16]


def test_forward_without_mask_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    idx   = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    lp    = model(idx, mask=None)
    assert list(lp.shape) == [1, 8, 16]


def test_forward_multi_layer_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                         num_heads=2, num_layers=2)
    idx  = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    mask = make_causal_mask(8)
    lp   = model(idx, mask=mask)
    assert list(lp.shape) == [1, 8, 16]


def test_log_probs_sum_to_one_cpu():
    """log_softmax output: exp(log_probs).sum(dim=-1) should be close to 1."""
    model = TinyCausalLM(vocab_size=16, block_size=4, embed_dim=8, num_heads=2)
    idx   = _int_tensor([[0, 1, 2, 3]])
    mask  = make_causal_mask(4)
    with ortho.no_grad():
        lp = model(idx, mask=mask)                        # [1, 4, 16]
    lp_cpu = lp.to("cpu") if lp.device.is_cuda() else lp
    lp_np  = np.array(lp_cpu.tolist(), dtype=np.float32).reshape(4, 16)
    row_sums = np.exp(lp_np).sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)


# TinyCausalLM CPU backward.

def test_backward_loss_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    idx   = _int_tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                          [2, 4, 6, 8, 10, 12, 14, 0]])
    tgt   = _int_tensor([[2, 3, 4, 5, 6, 7, 8, 9],
                          [4, 6, 8, 10, 12, 14, 0, 2]])
    mask  = make_causal_mask(8)

    lp   = model(idx, mask=mask)
    loss = ortho.nll_loss(lp.reshape([2 * 8, 16]), tgt.reshape([2 * 8]),
                          reduction="mean")
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_with_grad > 0


def test_backward_all_params_have_grad_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    idx  = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    tgt  = _int_tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    mask = make_causal_mask(8)

    lp   = model(idx, mask=mask)
    loss = ortho.nll_loss(lp.reshape([8, 16]), tgt.reshape([8]), reduction="mean")
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None, f"Parameter shape {list(p.shape)} missing grad"


def test_backward_multi_layer_all_params_cpu():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                         num_heads=2, num_layers=2)
    idx  = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    tgt  = _int_tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    mask = make_causal_mask(8)

    lp   = model(idx, mask=mask)
    loss = ortho.nll_loss(lp.reshape([8, 16]), tgt.reshape([8]), reduction="mean")
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None


# TinyCausalLM CUDA forward and backward.

@requires_cuda
def test_forward_shape_cuda():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                         num_heads=2).to("cuda")
    idx  = _int_tensor([[1, 3, 2, 5, 0, 7, 4, 6]], "cuda")
    mask = make_causal_mask(8, "cuda")
    lp   = model(idx, mask=mask)
    assert list(lp.shape) == [1, 8, 16]
    assert lp.device.is_cuda()


@requires_cuda
def test_backward_all_params_have_grad_cuda():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                         num_heads=2).to("cuda")
    idx  = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]], "cuda")
    tgt  = _int_tensor([[1, 2, 3, 4, 5, 6, 7, 0]], "cuda")
    mask = make_causal_mask(8, "cuda")

    lp   = model(idx, mask=mask)
    loss = ortho.nll_loss(lp.reshape([8, 16]), tgt.reshape([8]), reduction="mean")
    loss.backward()

    assert loss.device.is_cuda()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.device.is_cuda()


@requires_cuda
def test_forward_multi_layer_cuda():
    model = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                         num_heads=2, num_layers=2).to("cuda")
    idx  = _int_tensor([[0, 1, 2, 3, 4, 5, 6, 7]], "cuda")
    mask = make_causal_mask(8, "cuda")
    lp   = model(idx, mask=mask)
    assert list(lp.shape) == [1, 8, 16]
    assert lp.device.is_cuda()


# Autoregressive generation.

def test_generate_returns_correct_length_cpu():
    model  = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    prompt = [3, 5, 7]
    result = generate(model, prompt, max_new_tokens=5, device="cpu")
    assert len(result) == len(prompt) + 5


def test_generate_tokens_in_range_cpu():
    model  = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    result = generate(model, [0, 1, 2], max_new_tokens=8, device="cpu")
    assert all(0 <= tok < 16 for tok in result)


def test_generate_prompt_preserved_cpu():
    model  = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8, num_heads=2)
    prompt = [3, 5, 7, 9]
    result = generate(model, prompt, max_new_tokens=4, device="cpu")
    assert result[:len(prompt)] == prompt


@requires_cuda
def test_generate_cuda_smoke():
    model  = TinyCausalLM(vocab_size=16, block_size=8, embed_dim=8,
                          num_heads=2).to("cuda")
    result = generate(model, [0, 1, 2, 3], max_new_tokens=6, device="cuda")
    assert len(result) == 4 + 6
    assert all(0 <= tok < 16 for tok in result)
