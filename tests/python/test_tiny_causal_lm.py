from tests.support.capabilities import orthon_cuda_available as _cuda_available

import numpy as np
import pytest

from tests.support.examples import load_example

_example = load_example("tiny_causal_lm.py")
make_causal_mask, TinyCausalLM, generate = _example.make_causal_mask, _example.TinyCausalLM, _example.generate

import orthon as ortho




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


@pytest.mark.parametrize("vocab", [1, 2, 4, 8, 16, 32])
def test_r4_generate_wrapped_prompt_domain_and_length(vocab):
    model = TinyCausalLM(vocab, block_size=2, embed_dim=4, num_heads=2, mlp_ratio=1.)
    prompt = [(3 + 2 * k) % vocab for k in range(4)]
    original = prompt.copy()
    result = generate(model, prompt, 3, "cpu")
    assert prompt == original and result[:4] == original
    assert len(result) == 7 and all(0 <= token < vocab for token in result)


@pytest.mark.parametrize("prompt", [[], [0, 3]])
def test_r4_generate_zero_tokens_preserves_prompt_copy(prompt):
    model = TinyCausalLM(4, block_size=2, embed_dim=4, num_heads=2)
    result = generate(model, prompt, np.int64(0), "cpu")
    assert result == prompt and result is not prompt


def test_r4_generate_greedy_decoding_crops_context_and_masks():
    class GreedyModel:
        vocab_size, block_size = 4, 2

        def eval(self):
            self.evaluated = True

        def __call__(self, inp, mask):
            tokens = np.array(inp.tolist()).reshape(list(inp.shape))[0].tolist()
            seen.append(tokens)
            expected = np.where(np.triu(np.ones((len(tokens), len(tokens))), 1), -1e9, 0.)
            np.testing.assert_array_equal(np.array(mask.tolist()).reshape(expected.shape), expected)
            values = np.full((1, len(tokens), 4), -1., dtype=np.float32)
            values[:, :, 2] = 0.
            return ortho.from_numpy(values)

    seen = []
    model = GreedyModel()
    assert generate(model, [0, 3, 1], 3, "cpu") == [0, 3, 1, 2, 2, 2]
    assert seen == [[3, 1], [1, 2], [2, 2]] and model.evaluated


@pytest.mark.parametrize("prompt,count", [
    ([-1], 1), ([4], 1), ([1.25], 1), ([], 1), ([0], -1), ([0], .5),
], ids=["negative-token", "large-token", "noninteger-token", "empty-prompt",
        "negative-count", "noninteger-count"])
def test_r4_generate_rejects_invalid_domain_before_model(prompt, count):
    from types import SimpleNamespace
    model = SimpleNamespace(vocab_size=4, block_size=2,
                            eval=lambda: pytest.fail("invalid input reached model"))
    with pytest.raises(ValueError):
        generate(model, prompt, count, "cpu")


@pytest.mark.parametrize("arguments,domain", [
    *[(["--" + name, "0"], name) for name in [
        "vocab-size", "block-size", "embed-dim", "num-heads", "batch-size",
        "train-size", "test-size", "epochs"]],
    (["--num-layers", "-1"], "num-layers"),
    (["--generate-tokens", "-1"], "generate-tokens"),
    (["--embed-dim", "3", "--num-heads", "2"], "embed-dim"),
    *[(["--mlp-ratio", value], "mlp-ratio") for value in ["0", "nan", "1e308", "0.01"]],
    *[(["--dropout", value], "dropout") for value in ["-0.1", "1", "nan"]],
    *[(["--lr", value], "lr") for value in ["-1", "nan", "inf"]],
    (["--seed", "-1"], "seed"),
    (["--device", "invalid"], "device"),
    (["--vocab-size", "-1"], "vocab-size"),
])
def test_r4_cli_rejects_invalid_domain_before_training(monkeypatch, capsys, arguments, domain):
    import sys

    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI input reached allocation or data generation")

    monkeypatch.setattr(sys, "argv", ["tiny_causal_lm.py", *arguments])
    monkeypatch.setattr(_example.ortho, "cuda_enabled", forbidden)
    monkeypatch.setattr(_example.np.random, "default_rng", forbidden)
    monkeypatch.setattr(_example, "make_dataset", forbidden)
    monkeypatch.setattr(_example, "TinyCausalLM", forbidden)
    with pytest.raises(SystemExit) as error:
        _example.main()
    captured = capsys.readouterr()
    assert error.value.code == 2 and domain in captured.err
    assert not captured.out


def _r4_run_small_lm(tmp_path, vocab, device, *, zero_options=False):
    import ast
    import os
    import re
    import subprocess
    import sys

    script = _example.__file__
    code = ("import runpy, orthon; print('ORTHON_PATH', orthon.__file__); "
            "print('EXTENSION_PATH', orthon._C.__file__); "
            f"runpy.run_path({script!r}, run_name='__main__')")
    args = ["--device", device, "--epochs", "1", "--train-size", "4",
            "--test-size", "2", "--batch-size", "2", "--block-size", "2",
            "--embed-dim", "4", "--num-heads", "2", "--mlp-ratio", "1",
            "--generate-tokens", "0" if zero_options else "2"]
    if vocab != 32:
        args += ["--vocab-size", str(vocab)]
    if zero_options:
        args += ["--lr", "0", "--num-layers", "0"]
    result = subprocess.run([sys.executable, "-c", code, *args], cwd=tmp_path,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"device={device}" in result.stdout and "falling back" not in result.stdout
    rows = re.findall(r"^\s*1\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+s$",
                      result.stdout, re.MULTILINE)
    assert len(rows) == 1 and all(np.isfinite(float(value)) for value in rows[0])
    values = {}
    for label in ["Prompt", "Generated", "Expected", "Full seq"]:
        match = re.search(r"^" + label + r"\s*:\s*(\[.*\])$", result.stdout, re.MULTILINE)
        assert match, result.stdout
        values[label] = ast.literal_eval(match.group(1))
    count = 0 if zero_options else 2
    prompt = [(3 + 2 * k) % vocab for k in range(4)]
    assert values["Prompt"] == prompt and len(values["Generated"]) == count
    assert values["Expected"] == [(3 + 2 * k) % vocab for k in range(4, 4 + count)]
    assert values["Full seq"] == prompt + values["Generated"]
    assert all(0 <= token < vocab for token in values["Full seq"])


@pytest.mark.parametrize("vocab", [1, 2, 4, 8, 16, 32])
def test_r4_small_vocabulary_complete_cpu_cli(tmp_path, vocab):
    _r4_run_small_lm(tmp_path, vocab, "cpu")


@requires_cuda
@pytest.mark.parametrize("vocab", [2, 8])
def test_r4_small_vocabulary_complete_cuda_cli(tmp_path, vocab):
    _r4_run_small_lm(tmp_path, vocab, "cuda")


def test_r4_cli_legitimate_zero_options(tmp_path):
    _r4_run_small_lm(tmp_path, 1, "cpu", zero_options=True)
