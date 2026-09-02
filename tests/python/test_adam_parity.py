import pytest
import importlib.util
import sys
from pathlib import Path

import orthon
from orthon.optim import Adam

torch = pytest.importorskip("torch")


def flatten_nested(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(flatten_nested(item))
        return out
    return [float(x)]


def _stats(values):
    abs_values = [abs(float(v)) for v in values]
    return max(abs_values), sum(abs_values) / len(abs_values)


def _cifar_like_gradient_torch(shape, device="cpu"):
    total = shape[0] * shape[1]
    values = torch.arange(total, dtype=torch.float32, device=device).reshape(shape)
    grad = 0.00019 * torch.sin(values * 0.017) + 0.00008 * torch.cos(values * 0.071)
    grad[::17, ::5] *= 12.0
    grad[::31, ::7] *= -6.0
    return grad


def _orthon_from_torch(tensor, device="cpu"):
    d = orthon.from_numpy(tensor.detach().cpu().numpy())
    if device != "cpu":
        d = d.to(device)
    return d


def _adam_one_step_orthon(shape, grad_torch, device, lr=1e-5, eps=1e-6):
    p = orthon.full(list(shape), 0.5)
    if device != "cpu":
        p = p.to(device)
    p.requires_grad_(True)
    grad = _orthon_from_torch(grad_torch, device)
    before = flatten_nested(p.to("cpu").tolist() if device != "cpu" else p.tolist())

    out = p * grad
    out.backward()
    Adam([p], lr=lr, betas=(0.9, 0.999), eps=eps).step()

    after = flatten_nested(p.to("cpu").tolist() if device != "cpu" else p.tolist())
    updates = [abs(a - b) for a, b in zip(after, before)]
    return after, _stats(updates)


def _adam_one_step_torch(shape, grad, device, lr=1e-5, eps=1e-6):
    p = torch.full(shape, 0.5, dtype=torch.float32, device=device, requires_grad=True)
    before = p.detach().cpu().flatten().tolist()
    out = p * grad.to(device)
    out.backward(torch.ones_like(out))
    torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999), eps=eps).step()
    after = p.detach().cpu().flatten().tolist()
    updates = [abs(a - b) for a, b in zip(after, before)]
    return after, _stats(updates)


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


def _param_update_stats(param, before):
    after = flatten_nested(param.to("cpu").tolist())
    updates = [abs(a - b) for a, b in zip(after, before)]
    return _stats(updates)


def test_adam_one_step_matches_torch_cpu_simple():
    p_d = orthon.full([2, 3], 0.5)
    p_d.requires_grad_(True)

    g_d = orthon.full([2, 3], 0.01)

    # Orthon backward on tensor output seeds with ones,
    # so grad(p_d) = g_d.
    out_d = p_d * g_d
    out_d.backward()

    p_t = torch.full((2, 3), 0.5, dtype=torch.float32, requires_grad=True)
    g_t = torch.full((2, 3), 0.01, dtype=torch.float32)

    out_t = p_t * g_t
    out_t.backward(torch.ones_like(out_t))

    opt_d = Adam([p_d], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    opt_t = torch.optim.Adam([p_t], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    opt_d.step()
    opt_t.step()

    got = flatten_nested(p_d.tolist())
    expected = p_t.detach().flatten().tolist()

    assert got == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_adam_one_step_matches_torch_cuda_simple():
    if not orthon.cuda_enabled() or not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    p_d = orthon.full([2, 3], 0.5).to("cuda")
    p_d.requires_grad_(True)

    g_d = orthon.full([2, 3], 0.01).to("cuda")

    out_d = p_d * g_d
    out_d.backward()

    p_t = torch.full((2, 3), 0.5, dtype=torch.float32, device="cuda", requires_grad=True)
    g_t = torch.full((2, 3), 0.01, dtype=torch.float32, device="cuda")

    out_t = p_t * g_t
    out_t.backward(torch.ones_like(out_t))

    opt_d = Adam([p_d], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    opt_t = torch.optim.Adam([p_t], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    opt_d.step()
    opt_t.step()

    got = flatten_nested(p_d.to("cpu").tolist())
    expected = p_t.detach().cpu().flatten().tolist()

    assert got == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_adam_one_step_matches_torch_cpu_large_linear_weight_shape():
    shape = (512, 64)
    grad = _cifar_like_gradient_torch(shape)

    got, got_update_stats = _adam_one_step_orthon(shape, grad, "cpu")
    expected, expected_update_stats = _adam_one_step_torch(shape, grad, "cpu")

    assert got == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert got_update_stats == pytest.approx(expected_update_stats, rel=1e-4, abs=1e-7)


def test_adam_one_step_matches_torch_cuda_large_linear_weight_shape():
    if not orthon.cuda_enabled() or not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    shape = (512, 64)
    grad = _cifar_like_gradient_torch(shape)

    cpu_after, cpu_update_stats = _adam_one_step_orthon(shape, grad, "cpu")
    cuda_after, cuda_update_stats = _adam_one_step_orthon(shape, grad, "cuda")
    expected, expected_update_stats = _adam_one_step_torch(shape, grad, "cuda")

    assert cuda_after == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert cuda_after == pytest.approx(cpu_after, rel=1e-5, abs=1e-5)
    assert cuda_update_stats == pytest.approx(expected_update_stats, rel=1e-4, abs=1e-7)
    assert cuda_update_stats == pytest.approx(cpu_update_stats, rel=1e-4, abs=1e-7)


def test_adam_cifar_first_linear_weight_update_stats_match_cpu_cuda():
    if not orthon.cuda_enabled() or not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cifar = _load_cifar_example()
    try:
        train_loader, _ = cifar.build_cifar10_loaders(
            "data",
            batch_size=32,
            download=False,
            seed=1,
        )
        images, labels = next(iter(train_loader))
    except RuntimeError as exc:
        pytest.skip(f"CIFAR-10 data unavailable: {exc}")

    targets = cifar.labels_to_list(labels)
    results = {}
    for device in ["cpu", "cuda"]:
        # Both devices must start from IDENTICAL weights for this parity check
        # to mean anything. Orthon initialization does not read Python's
        # global random, so the framework RNG is what has to be reset.
        orthon.manual_seed(1)
        model = cifar.CIFAR10CNN(hidden=64, channels=8, init_scale=0.5)
        model.to(device)
        params = list(model.parameters())
        param4 = params[4]
        before = flatten_nested(param4.to("cpu").tolist())

        x = cifar.orthon_tensor_from_images(images, device)
        loss = orthon.cross_entropy(model(x), targets)
        loss.backward()
        Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-6).step()
        results[device] = _param_update_stats(param4, before)

    assert results["cuda"] == pytest.approx(results["cpu"], rel=1e-4, abs=1e-7)
