"""Completed-operation host latency, adaptive batching and descriptive statistics."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 25
    samples: int = 11
    target_ms: float = 40.0
    max_sample_ms: float = 500.0
    min_inner: int = 8
    max_inner: int = 50000
    fixed_inner: int | None = None

    def __post_init__(self):
        if (self.warmup < 0 or self.samples < 1 or self.target_ms <= 0 or
                self.max_sample_ms <= 0 or self.min_inner < 1 or
                self.max_inner < self.min_inner or
                self.fixed_inner is not None and self.fixed_inner < 1):
            raise ValueError("invalid timing configuration")


def quartiles(values):
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError("samples must be nonempty, finite and nonnegative")
    def pick(p):
        pos = p * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return pick(.25), pick(.5), pick(.75)


@dataclass(frozen=True)
class Statistics:
    samples: tuple
    inner: int

    @property
    def sample_count(self):
        return len(self.samples)

    def as_dict(self):
        q1, median, q3 = quartiles(self.samples)
        return dict(sample_count=self.sample_count, inner_iterations=self.inner,
                    median=median, q1=q1, q3=q3, iqr=q3-q1,
                    samples=list(self.samples), units="ms/call")


def batch_ms(fn, inner, sync):
    sync()  # outside the timed interval
    start = time.perf_counter()
    for _ in range(inner):
        fn()
    sync()  # completion belongs inside the interval; no per-call sync
    return (time.perf_counter() - start) * 1000


def estimate_ms(fn, sync, trials=5):
    fn()
    sync()
    return max(min(batch_ms(fn, 1, sync) for _ in range(trials)), 1e-4)


def choose_inner(est_a, est_b, cfg):
    if cfg.fixed_inner is not None:
        return cfg.fixed_inner
    if min(est_a, est_b) <= 0:
        raise ValueError("estimates must be positive")
    wanted = math.ceil(cfg.target_ms / min(est_a, est_b))
    cap = max(1, int(cfg.max_sample_ms / max(est_a, est_b)))
    # The duration cap wins over the preferred minimum for slow workloads.
    return max(1, min(max(cfg.min_inner, wanted), cap, cfg.max_inner))


def warmup(fn, sync, count):
    for _ in range(count):
        fn()
    sync()


def measure(fn, sync=lambda: None, config=TimingConfig()):
    warmup(fn, sync, config.warmup)
    inner = choose_inner(estimate_ms(fn, sync), estimate_ms(fn, sync), config)
    return Statistics(tuple(batch_ms(fn, inner, sync)/inner
                            for _ in range(config.samples)), inner)


def measure_pair(a, b, sync_a, sync_b, config=TimingConfig()):
    warmup(a, sync_a, config.warmup)
    warmup(b, sync_b, config.warmup)
    inner = choose_inner(estimate_ms(a, sync_a), estimate_ms(b, sync_b), config)
    values = [[], []]
    for sample in range(config.samples):
        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
            values[index].append(batch_ms((a,b)[index], inner, (sync_a,sync_b)[index])/inner)
    return Statistics(tuple(values[0]), inner), Statistics(tuple(values[1]), inner)


def comparison(a, b, correct):
    if not correct:
        return dict(speedup=None, verdict="invalid", stability="invalid", band=None)
    aq1, am, aq3 = quartiles(a.samples)
    bq1, bm, bq3 = quartiles(b.samples)
    if am == 0 or aq1 == 0 or bm == 0:
        return dict(speedup=None, verdict="unreliable", stability="unavailable", band=None)
    band = (bq1/aq3, bq3/aq1)
    ratio = bm/am
    verdict = ("unreliable" if band[0] <= 1 <= band[1] else
               "orthon" if ratio >= 1.1 else "pytorch" if ratio <= 1/1.1 else "tie")
    noisy = (aq3-aq1)/am > .15 or (bq3-bq1)/bm > .15
    return dict(speedup=ratio, verdict=verdict, band=band,
                stability="noisy" if noisy else "stable")


@contextmanager
def region(sync=lambda: None):
    """One continuous stateful region; never represented as independent runs."""
    sync()
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        sync()
        result['seconds'] = time.perf_counter() - start


def hot_loop(fn, sync, warmup_count, iterations):
    """Profiler driver: deterministic fixed calls, no benchmark wall-time claim."""
    warmup(fn, sync, warmup_count)
    for _ in range(iterations):
        fn()
    sync()
