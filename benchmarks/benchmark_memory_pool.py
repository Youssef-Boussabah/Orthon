import argparse
import gc
import time

import orthon as ortho


UNLIMITED_CACHE = (1 << 64) - 1


def reset_pool(enabled=True):
    gc.collect()
    ortho._memory_pool_set_enabled(True)
    ortho._memory_pool_set_max_cached_bytes(UNLIMITED_CACHE)
    ortho._memory_pool_clear_cache()
    ortho._memory_pool_reset_scratch("cpu")
    if ortho.cuda_enabled():
        ortho._memory_pool_reset_scratch("cuda")
    ortho._memory_pool_reset_stats()
    ortho._memory_pool_set_enabled(enabled)


def time_allocations(enabled, iters, shape):
    reset_pool(enabled)
    start = time.perf_counter()
    for _ in range(iters):
        t = ortho.zeros(shape)
        del t
    gc.collect()
    elapsed = time.perf_counter() - start
    return elapsed, ortho._memory_pool_stats()


def time_cuda_reductions(enabled, iters, shape):
    reset_pool(enabled)
    x = ortho.ones(shape).to("cuda")
    ortho.cuda_synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        ortho.sum(x)
        ortho.mean(x)
        ortho.max(x)
    ortho.cuda_synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, ortho._memory_pool_stats(), ortho._memory_pool_scratch_stats("cuda")


def print_pool_result(label, elapsed, stats):
    print(f"{label:28s} {elapsed:8.4f}s")
    print(
        "  "
        f"total={stats['total_allocations']} "
        f"raw={stats['raw_allocations']} "
        f"reused={stats['reused_allocations']} "
        f"current={stats['current_bytes']} "
        f"cached={stats['cached_bytes']} "
        f"peak={stats['peak_bytes']}"
    )


def print_scratch_result(stats):
    print(
        "  "
        f"scratch_reserved={stats['reserved_bytes']} "
        f"scratch_used={stats['used_bytes']} "
        f"scratch_peak_used={stats['peak_used_bytes']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--alloc-size", type=int, default=1024 * 1024)
    parser.add_argument("--reduction-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    alloc_shape = [args.alloc_size]
    print(f"Repeated CPU zeros allocations, shape={alloc_shape}, iters={args.iters}")
    for enabled in (True, False):
        elapsed, stats = time_allocations(enabled, args.iters, alloc_shape)
        print_pool_result(f"cache {'enabled' if enabled else 'disabled'}", elapsed, stats)

    if not ortho.cuda_enabled():
        print("\nCUDA unavailable; skipped reduction scratch benchmark.")
        return

    reduction_shape = [args.reduction_size]
    print(f"\nCUDA full reductions, shape={reduction_shape}, iters={args.iters}")
    for enabled in (True, False):
        elapsed, stats, scratch = time_cuda_reductions(enabled, args.iters, reduction_shape)
        print_pool_result(f"cache {'enabled' if enabled else 'disabled'}", elapsed, stats)
        print_scratch_result(scratch)


if __name__ == "__main__":
    main()
