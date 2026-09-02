"""Benchmark fused_linear_relu against separate ops and PyTorch."""

import argparse
import json
import statistics
import time
from pathlib import Path


def sync_orthon(device):
    if device == "cuda":
        import orthon

        orthon.cuda_synchronize()


def sync_torch(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def summarize(latencies):
    mean_ms = statistics.mean(latencies) * 1000.0
    return {
        "latency_ms": mean_ms,
        "calls_per_sec": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def time_calls(fn, sync, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        latencies.append(time.perf_counter() - start)
    return summarize(latencies)


def orthon_full(shape, value, device):
    import orthon

    t = orthon.full(shape, value)
    return t.to("cuda") if device == "cuda" else t


def bench_orthon(device, m, k, n, warmup, iters):
    import orthon

    if device == "cuda" and not orthon.cuda_enabled():
        return []

    x = orthon_full([m, k], 0.05, device)
    weight = orthon_full([k, n], 0.02, device)
    bias = orthon_full([n], 0.1, device)
    sync = lambda: sync_orthon(device)

    rows = []
    rows.append(record(
        "orthon_separate",
        device,
        time_calls(lambda: orthon.relu(orthon.matmul(x, weight) + bias),
                   sync, warmup, iters),
    ))
    rows.append(record(
        "orthon_fused",
        device,
        time_calls(lambda: orthon.fused_linear_relu(x, weight, bias),
                   sync, warmup, iters),
    ))
    return rows


def bench_torch(device, m, k, n, warmup, iters):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        return []

    torch_device = torch.device(device)
    x = torch.full((m, k), 0.05, device=torch_device)
    weight = torch.full((k, n), 0.02, device=torch_device)
    bias = torch.full((n,), 0.1, device=torch_device)
    sync = lambda: sync_torch(torch, device)

    return [record(
        "torch_equivalent",
        device,
        time_calls(lambda: torch.relu(x @ weight + bias), sync, warmup, iters),
    )]


def record(name, device, metrics):
    row = {
        "name": name,
        "device": device,
        **metrics,
    }
    print(
        f"{name:<18} {device:<4} "
        f"{row['latency_ms']:>9.4f} ms  {row['calls_per_sec']:>9.2f} calls/s"
    )
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark fused Linear+Bias+ReLU prototype.")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", default="examples/results/fusion_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for device in ["cpu", "cuda"]:
        rows.extend(bench_orthon(device, args.m, args.k, args.n, args.warmup, args.iters))
        rows.extend(bench_torch(device, args.m, args.k, args.n, args.warmup, args.iters))

    output = {
        "schema": "orthon.fusion_benchmark.v1",
        "shape": {"x": [args.m, args.k], "weight": [args.k, args.n], "bias": [args.n]},
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote fusion benchmark results to {path}")


if __name__ == "__main__":
    main()
