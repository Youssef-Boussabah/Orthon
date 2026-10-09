"""Canonical softmax_cross_entropy workloads; see benchmarks._support for methodology."""
from benchmarks._support.operators import run

if __name__ == "__main__":
    raise SystemExit(run("softmax_cross_entropy"))
