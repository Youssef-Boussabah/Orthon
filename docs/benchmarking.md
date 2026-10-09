# Benchmarking

Run the repository-only `benchmarks` package against an intentionally installed
Orthon or explicit complete package stage. It is not shipped in the runtime wheel.
PyTorch is needed for reference comparisons, not for ordinary Orthon applications.
Performance observations are specific to hardware, workload and measurement
configuration; Orthon 1.0.0rc1 has no performance contract.

## Shared latency method

`benchmarks/_support/timing.py` owns TimingConfig and completed-operation host
latency. Defaults are 25 warmup calls, 11 samples, 40 ms target sample duration,
500 ms estimated maximum sample duration, preferred minimum 8 and maximum 50,000
inner calls. The duration cap wins over the preferred minimum for slow workloads.
Calibration estimates choose a shared count for a pair: target the faster route
while limiting the slower route. Actual duration may differ from the estimate.
`--inner` explicitly overrides adaptive batching.

Each timed batch synchronizes before the clock, invokes its route repeatedly and
synchronizes completion inside the interval. CUDA synchronization occurs at batch
boundaries, not once per inner call. Paired comparisons use identical inner counts
and alternate A/B then B/A sample order. The GPU prelude warms idle clocks outside
samples. This measures host-observed completed-operation latency including
launch/Python overhead, not event-only kernel duration.

Results retain samples, median, Q1, Q3 and IQR in ms/call. Interpolated quartiles
summarize observed dispersion, not a statistical confidence interval. Correctness
is checked before interpreting comparisons; failed correctness gives an invalid
verdict and no speedup. Dispersion bands can make a comparison unreliable; a
single ratio does not establish a performance claim. There is no timing pass/fail
threshold used to accept the project.

## Commands

From the repository root, with an installed package and any required reference
framework:

```bash
python -m benchmarks.compare_pytorch --help
python -m benchmarks.bench_matmul --help
python -m benchmarks.benchmark_conv2d --help
python -m benchmarks.profile_cuda_workload --help
python -m benchmarks.compare_pytorch --device cpu --quick --benchmark elementwise_add --json /tmp/comparison.json
python -m benchmarks.bench_matmul --device cuda:0 --dtype bfloat16 --json /tmp/matmul.json
python -m benchmarks.benchmark_conv2d --device cuda:0 --json /tmp/conv2d.json
python -m benchmarks.report_benchmarks /tmp/comparison.json --markdown /tmp/comparison.md
```

`--quick` uses bounded smoke settings (one warmup, three samples, one inner call)
and records `representative=false`. Those timings are not publication results.
Shared latency options include `--warmup`, `--samples`, `--target-ms`,
`--max-sample-ms`, `--inner` and `--json`. Check each module's help for workload
options; not every tool is a stateless latency comparison.

## Results and provenance

The comparison driver selects and renders rows.
`benchmarks/_support/comparison_cases.py` owns deterministic setup, conversions
and correctness checks.

`benchmarks/_support/results.py` is the shared strict UTF-8 JSON writer. Documents
use `schema="orthon-benchmark"`, `schema_version=1`, benchmark name, mode, config,
provenance and cases. `allow_nan=False` rejects nonstandard numeric JSON.
Case fields include shape, dtype, exact device, implementation/mode, correctness,
warmup, samples and summary statistics where applicable.

`benchmarks/_support/environment.py` records commit/dirty state, package/extension
paths, Python/platform/CPU/logical CPU count, CUDA/NCCL compile flags, GPU/driver,
toolkit, PyTorch/CUDA versions and precision flags. Best-effort unavailable fields
are null, not invented. Record command/options and artifact paths with results.
Keep generated JSON/Markdown/profiler reports outside tracked evidence files.

Reporting validates the result schema and correctness; aggregation is not another
timer. Legacy snapshots require explicit `--include-legacy` where supported and
retain their original label and cannot establish current performance.

`compare_pytorch --check-only` prints each numerical PASS/FAIL without warmup,
timing samples or JSON/Markdown output. It exits 0 only when every selected check
passes, and 1 if any check fails. An empty selection is a usage error; `--list`
still exits successfully. Checker exceptions remain errors.

## Stateful workloads and measurement limits

Training drivers in `benchmarks/` measure explicitly labeled continuous regions
of changing model/optimizer state. Repeated epochs are not independent samples
of an identically reset model. `bench_model_training`, `bench_tiny_mlp_training`,
`bench_amp_vs_fp32` and dataset `benchmark_training` retain those semantics.
Applications/dataset recipes are described in [training workloads](training_workloads.md).
Checkpoint timing measures serialization service latency with filesystem caching,
not storage-device bandwidth. Local DDP tools measure host/local communication;
their results provide no NCCL throughput evidence. Multirank CUDA/NCCL tools need
real dependency and hardware gates.

Private `orthon._diagnostics` provides forced matmul/Conv2d routes, planner guards
and memory-pool observations. A forced route bypasses ordinary selection and must
be labeled diagnostic. Scoped selector environment changes restore previous
values in `finally`; process-global selectors are not ordinary user API. Pool
on/off experiments must identify the setting and preserve workload equivalence.

## Profiling is separate

`benchmarks/profile_cuda_workload.py` runs fixed deterministic hot loops and
emits workload/provenance metadata. It does not report normal latency or a
comparison verdict. Nsight Systems owns timelines; Nsight Compute owns kernel
counters/durations. Instrumented wall time cannot substitute for unprofiled
benchmark samples. See [Nsight workflow](nsight_profiling.md),
[CUDA runtime](cuda_runtime.md) and [test architecture](testing.md).
