import argparse
import time

import numpy as np

import orthon as ortho


SHAPE = (1024, 1024)
WARMUP = 100
ITERS = 1000


def _tensor(dtype, fill=None):
    np_dtype = np.float32 if dtype == "float32" else np.float64
    if fill is None:
        arr = np.linspace(-1.0, 1.0, num=SHAPE[0] * SHAPE[1], dtype=np_dtype).reshape(SHAPE)
    else:
        arr = np.full(SHAPE, fill, dtype=np_dtype)
    return ortho.from_numpy(arr)


def _time(fn):
    for _ in range(WARMUP):
        fn()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    return time.perf_counter() - start


def _gbps(dtype, bytes_per_elem, touched_tensors, elapsed):
    total_bytes = SHAPE[0] * SHAPE[1] * bytes_per_elem * touched_tensors * ITERS
    return total_bytes / elapsed / 1e9


def run(dtype):
    bytes_per_elem = 4 if dtype == "float32" else 8
    a = _tensor(dtype)
    b = _tensor(dtype, fill=0.5)

    ops = [
        ("neg", lambda: ortho.neg(a), 2),
        ("relu", lambda: ortho.relu(a), 2),
        ("leaky_relu", lambda: ortho.leaky_relu(a, 0.1), 2),
        ("add", lambda: ortho.add(a, b), 3),
        ("mul", lambda: ortho.mul(a, b), 3),
    ]

    print(f"\n{dtype} {SHAPE[0]}x{SHAPE[1]} ({ITERS} iterations)")
    for name, fn, touched in ops:
        elapsed = _time(fn)
        print(f"{name:12s} {elapsed:8.3f}s  {_gbps(dtype, bytes_per_elem, touched, elapsed):8.2f} GB/s")

    inplace_mul = _tensor(dtype)
    elapsed = _time(lambda: inplace_mul.mul_(1.000001))
    print(f"{'mul_':12s} {elapsed:8.3f}s  {_gbps(dtype, bytes_per_elem, 2, elapsed):8.2f} GB/s")

    inplace_add = _tensor(dtype)
    elapsed = _time(lambda: inplace_add.add_(b, alpha=0.000001))
    print(f"{'add_':12s} {elapsed:8.3f}s  {_gbps(dtype, bytes_per_elem, 3, elapsed):8.2f} GB/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float64", "all"], default="all")
    args = parser.parse_args()

    dtypes = ["float32", "float64"] if args.dtype == "all" else [args.dtype]
    for dtype in dtypes:
        run(dtype)


if __name__ == "__main__":
    main()
