"""Measure repeated tensor allocation overhead with and without the memory pool."""

import argparse
import gc
import time

import orthon


def reset_pool(enabled):
    gc.collect()
    orthon._memory_pool_set_enabled(True)
    orthon._memory_pool_clear_cache()
    orthon._memory_pool_reset_stats()
    orthon._memory_pool_set_enabled(enabled)


def allocate_many(num_allocations, shape, device):
    start = time.perf_counter()
    for _ in range(num_allocations):
        t = orthon.zeros(shape)
        if device == "cuda":
            t = t.to("cuda")
        del t
    gc.collect()
    if device == "cuda":
        orthon.cuda_synchronize()
    return time.perf_counter() - start


def run_case(enabled, num_allocations, shape, device):
    reset_pool(enabled)
    elapsed = allocate_many(num_allocations, shape, device)
    stats = orthon._memory_pool_stats()
    return elapsed, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Orthon tensor allocation caching.")
    parser.add_argument("--num-allocations", type=int, default=20000)
    parser.add_argument("--num-elements", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not orthon.cuda_enabled():
        raise RuntimeError("CUDA is not available in this Orthon build.")

    shape = [args.num_elements]
    pooled_time, pooled_stats = run_case(True, args.num_allocations, shape, args.device)
    raw_time, raw_stats = run_case(False, args.num_allocations, shape, args.device)
    orthon._memory_pool_set_enabled(True)

    print(f"device: {args.device}")
    print(f"shape: {shape}")
    print(f"allocations: {args.num_allocations}")
    print()
    print(f"with pool:    {pooled_time:.4f}s")
    print(f"without pool: {raw_time:.4f}s")
    if pooled_time > 0:
        print(f"speedup:      {raw_time / pooled_time:.2f}x")
    print()
    print("pool stats:")
    print(f"  total allocations:  {pooled_stats['total_allocations']}")
    print(f"  raw allocations:    {pooled_stats['raw_allocations']}")
    print(f"  reused allocations: {pooled_stats['reused_allocations']}")
    print(f"  peak bytes:         {pooled_stats['peak_bytes']}")
    print(f"  cached bytes:       {pooled_stats['cached_bytes']}")
    print()
    print("raw stats:")
    print(f"  total allocations:  {raw_stats['total_allocations']}")
    print(f"  raw allocations:    {raw_stats['raw_allocations']}")
    print(f"  reused allocations: {raw_stats['reused_allocations']}")


if __name__ == "__main__":
    main()
