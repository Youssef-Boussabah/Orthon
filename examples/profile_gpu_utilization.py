"""Short CUDA training workload plus external profiling guidance."""

import argparse
import time


def cuda_device_summary():
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "name": "unknown (install torch for device metadata)",
            "allocated_mb": None,
            "reserved_mb": None,
        }

    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "unavailable",
            "allocated_mb": None,
            "reserved_mb": None,
        }

    device = torch.device("cuda")
    return {
        "available": True,
        "name": torch.cuda.get_device_name(device),
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
    }


def run_orthon_training(iters):
    import orthon
    from orthon.nn import Linear
    from orthon.optim import SGD

    if not orthon.cuda_enabled():
        print("Orthon CUDA is unavailable in this build.")
        return None

    model = Linear(32, 10).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.01)
    x = orthon.full([16, 32], 0.1).to("cuda")
    targets = [i % 10 for i in range(16)]

    orthon.cuda_synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        loss = orthon.cross_entropy(model(x), targets)
        loss.backward()
        optimizer.step()
    orthon.cuda_synchronize()
    return time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short CUDA training workload.")
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = cuda_device_summary()
    print(f"CUDA available: {summary['available']}")
    print(f"CUDA device: {summary['name']}")
    if summary["allocated_mb"] is not None:
        print(f"GPU memory allocated before workload: {summary['allocated_mb']:.2f} MiB")
        print(f"GPU memory reserved before workload:  {summary['reserved_mb']:.2f} MiB")

    elapsed = run_orthon_training(args.iters)
    if elapsed is not None:
        print(f"Orthon CUDA training iterations: {args.iters}")
        print(f"Runtime: {elapsed:.4f}s")

    summary_after = cuda_device_summary()
    if summary_after["allocated_mb"] is not None:
        print(f"GPU memory allocated after workload: {summary_after['allocated_mb']:.2f} MiB")
        print(f"GPU memory reserved after workload:  {summary_after['reserved_mb']:.2f} MiB")

    print()
    print("This script does not estimate utilization percentages.")
    print("Use external tools while it runs, for example:")
    print("  nvidia-smi dmon")
    print("  nsys profile -o profiles/orthon_cuda_training python examples/profile_gpu_utilization.py")
    print("  ncu --target-processes all python examples/profile_gpu_utilization.py")


if __name__ == "__main__":
    main()
