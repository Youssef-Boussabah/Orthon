"""Operator latency benchmarks for Orthon and PyTorch."""

import json
import statistics
import time
from pathlib import Path


RESULTS_PATH = Path("examples/results/op_benchmark.json")


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def summarize_ms(latencies):
    mean_ms = statistics.mean(latencies) * 1000.0
    return {
        "latency_ms": mean_ms,
        "p50_ms": percentile(latencies, 0.50) * 1000.0,
        "p95_ms": percentile(latencies, 0.95) * 1000.0,
        "throughput_calls_per_sec": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def sync_orthon(device):
    if device == "cuda":
        import orthon

        orthon.cuda_synchronize()


def sync_torch(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_calls(fn, sync, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    sync()

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        latencies.append(time.perf_counter() - start)
    return summarize_ms(latencies)


def orthon_full(shape, value, device):
    import orthon

    tensor = orthon.full(shape, value)
    return tensor.to("cuda") if device == "cuda" else tensor


def orthon_available(device):
    if device == "cpu":
        return True
    import orthon

    return bool(orthon.cuda_enabled())


def bench_orthon(device):
    import orthon

    if not orthon_available(device):
        return []

    results = []
    sync = lambda: sync_orthon(device)

    x = orthon_full([4, 3, 32, 32], 0.25, device)
    w = orthon_full([8, 3, 3, 3], 0.05, device)
    b = orthon_full([8], 0.0, device)
    results.append(record("conv2d_forward", "orthon", device, time_calls(
        lambda: orthon.conv2d(x, w, b, 1, 1, 1, 1), sync, iters=10)))

    def conv_backward():
        xi = orthon_full([2, 3, 16, 16], 0.25, device)
        wi = orthon_full([4, 3, 3, 3], 0.05, device)
        bi = orthon_full([4], 0.0, device)
        xi.requires_grad_(True)
        wi.requires_grad_(True)
        bi.requires_grad_(True)
        orthon.conv2d(xi, wi, bi, 1, 1, 1, 1).backward()

    results.append(record("conv2d_backward", "orthon", device, time_calls(
        conv_backward, sync, warmup=1, iters=5)))

    a = orthon_full([32, 128], 0.25, device)
    m = orthon_full([128, 64], 0.05, device)
    results.append(record("matmul_forward", "orthon", device, time_calls(
        lambda: orthon.matmul(a, m), sync, iters=20)))

    def matmul_backward():
        ai = orthon_full([32, 128], 0.25, device)
        mi = orthon_full([128, 64], 0.05, device)
        ai.requires_grad_(True)
        mi.requires_grad_(True)
        orthon.matmul(ai, mi).backward()

    results.append(record("matmul_backward", "orthon", device, time_calls(
        matmul_backward, sync, warmup=1, iters=10)))

    logits = orthon_full([32, 10], 0.1, device)
    targets = [i % 10 for i in range(32)]
    results.append(record("cross_entropy_forward", "orthon", device, time_calls(
        lambda: orthon.cross_entropy(logits, targets), sync, iters=20)))

    def ce_backward():
        li = orthon_full([32, 10], 0.1, device)
        li.requires_grad_(True)
        orthon.cross_entropy(li, targets).backward()

    results.append(record("cross_entropy_backward", "orthon", device, time_calls(
        ce_backward, sync, warmup=1, iters=10)))

    return results


def torch_available(torch, device):
    return device == "cpu" or bool(torch.cuda.is_available())


def bench_torch(device):
    import torch

    if not torch_available(torch, device):
        return []

    torch_device = torch.device(device)
    sync = lambda: sync_torch(torch, device)
    results = []

    x = torch.full((4, 3, 32, 32), 0.25, device=torch_device)
    w = torch.full((8, 3, 3, 3), 0.05, device=torch_device)
    b = torch.zeros(8, device=torch_device)
    results.append(record("conv2d_forward", "torch", device, time_calls(
        lambda: torch.nn.functional.conv2d(x, w, b, padding=1), sync, iters=20)))

    def conv_backward():
        xi = torch.full((2, 3, 16, 16), 0.25, device=torch_device, requires_grad=True)
        wi = torch.full((4, 3, 3, 3), 0.05, device=torch_device, requires_grad=True)
        bi = torch.zeros(4, device=torch_device, requires_grad=True)
        torch.nn.functional.conv2d(xi, wi, bi, padding=1).sum().backward()

    results.append(record("conv2d_backward", "torch", device, time_calls(
        conv_backward, sync, warmup=1, iters=10)))

    a = torch.full((32, 128), 0.25, device=torch_device)
    m = torch.full((128, 64), 0.05, device=torch_device)
    results.append(record("matmul_forward", "torch", device, time_calls(
        lambda: a @ m, sync, iters=30)))

    def matmul_backward():
        ai = torch.full((32, 128), 0.25, device=torch_device, requires_grad=True)
        mi = torch.full((128, 64), 0.05, device=torch_device, requires_grad=True)
        (ai @ mi).sum().backward()

    results.append(record("matmul_backward", "torch", device, time_calls(
        matmul_backward, sync, warmup=1, iters=20)))

    logits = torch.full((32, 10), 0.1, device=torch_device)
    targets = torch.tensor([i % 10 for i in range(32)], device=torch_device)
    results.append(record("cross_entropy_forward", "torch", device, time_calls(
        lambda: torch.nn.functional.cross_entropy(logits, targets), sync, iters=30)))

    def ce_backward():
        li = torch.full((32, 10), 0.1, device=torch_device, requires_grad=True)
        torch.nn.functional.cross_entropy(li, targets).backward()

    results.append(record("cross_entropy_backward", "torch", device, time_calls(
        ce_backward, sync, warmup=1, iters=20)))

    return results


def record(workload, framework, device, metrics):
    row = {
        "workload": workload,
        "framework": framework,
        "device": device,
    }
    row.update(metrics)
    print(
        f"{workload:<24} {framework:<8} {device:<4} "
        f"{row['latency_ms']:>9.4f} ms  {row['throughput_calls_per_sec']:>9.2f} calls/s"
    )
    return row


def main():
    results = {
        "schema": "orthon.op_benchmark.v1",
        "shapes": {
            "conv2d_forward": "input=[4,3,32,32], weight=[8,3,3,3], padding=1",
            "conv2d_backward": "input=[2,3,16,16], weight=[4,3,3,3], padding=1",
            "matmul": "A=[32,128], B=[128,64]",
            "cross_entropy": "logits=[32,10]",
        },
        "results": [],
    }

    for framework in ["orthon", "torch"]:
        for device in ["cpu", "cuda"]:
            if framework == "orthon":
                results["results"].extend(bench_orthon(device))
            else:
                results["results"].extend(bench_torch(device))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote operator benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
