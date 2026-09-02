# Profiling Guide

Profiles are collected into this directory. All commands below are run from the repo root.

---

## Benchmark `--profile` mode

Every benchmark script accepts `--profile` and `--shape` flags that narrow the
run to a single large shape and increase the rep count, making it easier to
get a clean signal in a profiler without drowning in startup noise.

```bash
# matmul — single 1024×1024 run, 100 reps
python benchmarks/bench_matmul.py --profile --shape 1024

# softmax / cross_entropy — single [2048, 4096] run, 100 reps
python benchmarks/bench_softmax_cross_entropy.py --profile --shape 2048 4096

# elementwise - single 16 M element run, 100 reps
python benchmarks/bench_elementwise.py --profile --shape 16000000

# reductions - single 16 M element run, 100 reps
python benchmarks/bench_reductions.py --profile --shape 16000000

# eval forward with and without autograd graph construction
python examples/benchmark_no_grad.py --device cuda --iterations 100 --warmup 10
```

Pass `--profile` together with the profiler to focus collection on just the
hot loop and skip the small warm-up shapes.

### A100 BF16 matmul profiling

Use the clean-dispatch mode when comparing BF16 matmul dispatch choices without
extra guard-report or profiling overhead:

```bash
python benchmarks/bench_matmul.py --dtype bfloat16 --device cuda --clean-dispatch
```

Use the cuBLAS overhead profile when investigating the BF16 cuBLAS path around
the actual GEMM call:

```bash
python benchmarks/bench_matmul.py --dtype bfloat16 --device cuda --profile-cublas-overhead
```

Interpret the BF16 rows as follows:

- `custom-tiled-bfloat16` is the handwritten CUDA tiled kernel.
- `dispatch-bfloat16 (...)` is the normal matmul dispatch path. The label in
  parentheses shows whether dispatch selected a custom kernel or cuBLAS.
- `forced-cublas-bfloat16` bypasses the normal work-threshold decision and is
  useful for checking cuBLAS eligibility.
- `cublas-event-only-bfloat16` times CUDA events around `cublasGemmEx` only.
  It excludes Orthon allocation, dispatch, guard checks, and benchmark setup.

Guard reports explain fallback decisions. They are especially useful for small
batched BF16 workloads, where the current policy keeps the custom tiled fallback
when `work_items < 16,777,216`.

For CIFAR MiniVGG training, use the script-level profiler:

```powershell
$env:ORTHON_PROFILE_CONV2D="1"
python .\examples\cifar10_cnn.py --model minivgg --epochs 1 --batch-size 128 --device cuda --optimizer adam --lr 0.001 --adam-eps 0.00000001 --weight-decay 0.0005 --augment --max-train-batches 100 --max-test-batches 20 --profile --profile-sync
```

Use `--profile-sync` for accurate CUDA section timing. Without synchronization,
queued GPU work can be charged to a later Python section; one MiniVGG async
profile made optimizer look dominant, while synchronized timing showed backward
was dominant. The CIFAR profiler reports `data_loading`, `label_conversion`,
`input_conversion`, `forward`, `loss`, `backward`, `optimizer`, `metrics`, and
`evaluation`, and stores profile data in the metrics JSON. With
`ORTHON_PROFILE_CONV2D=1`, it also prints Conv2d backward time by shape and by
grad-input, grad-weight, and grad-bias component.

Recorded local full MiniVGG command:

```powershell
python .\examples\cifar10_cnn.py --model minivgg --epochs 30 --batch-size 128 --device cuda --optimizer adam --lr 0.001 --adam-eps 0.00000001 --weight-decay 0.0005 --augment --lr-schedule step --lr-step-epochs 15,25 --lr-gamma 0.1 --max-train-batches 0 --max-test-batches 0
```

A completed run of this command (the script writes its metrics to
`examples/results/cifar10_cnn.json`) reached a best local test accuracy of
84.11% at epoch 27. Final epoch 30 recorded train loss 0.3695, train accuracy
87.09%, test loss 0.4901, test accuracy 83.83%, and 27.3s epoch time. Full
epochs run around 24-29s locally on the RTX 4070 Laptop GPU.

---

## Nsight Systems (timeline + CPU/GPU correlation)

Captures a full timeline: kernel launches, memory copies, and CPU-side
Python/C++ call stacks. Use it to see the overall launch pattern and identify
large gaps between kernels.

```bash
# Tiled matmul — full timeline
nsys profile -o profiles/matmul_tiled \
    python benchmarks/bench_matmul.py --profile --shape 1024

# Softmax + cross entropy — full timeline
nsys profile -o profiles/softmax_ce \
    python benchmarks/bench_softmax_cross_entropy.py --profile --shape 2048 4096

# Elementwise ops
nsys profile -o profiles/elementwise \
    python benchmarks/bench_elementwise.py --profile --shape 16000000

# Reductions
nsys profile -o profiles/reductions \
    python benchmarks/bench_reductions.py --profile --shape 16000000
```

Open the resulting `.nsys-rep` files in the Nsight Systems GUI (`nsys-ui`).

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--trace=cuda,python` | Include Python function trace alongside CUDA |
| `--stats=true` | Print kernel summary to stdout after collection |
| `--delay=2` | Skip 2 s of startup (avoids profiling import overhead) |
| `--capture-range=nvtx` | Record only inside NVTX-marked regions (see below) |

**What to look for in the timeline:**

- **Kernel launch gaps** — idle time between launches indicates driver/CPU overhead.
  Fused kernels should show one launch where the old design showed three.
- **D2H copies** - `cross_entropy` now returns a CUDA scalar loss. D2H copies
  should appear when user code explicitly materializes host values, for example
  `loss.item()`, `Tensor.tolist()`, accuracy calculation, or benchmark reporting.
- **Concurrent streams** — currently all ops use the default stream; no overlap expected.
- **Memory operations** — `cudaMalloc` calls show as purple bars. Repeated allocs
  inside the timed loop indicate missing buffer reuse.

---

## Nsight Compute (per-kernel hardware counters)

Collects detailed SM-level metrics for a single kernel. Much slower than
Nsight Systems — target one script at a time.

```bash
# Full metric set — matmul tiled kernel
ncu --set full -o profiles/matmul_ncu \
    python benchmarks/bench_matmul.py --profile --shape 1024

# Roofline model only (faster collection)
ncu --set roofline -o profiles/matmul_roofline \
    python benchmarks/bench_matmul.py --profile --shape 1024

# Target a specific kernel by name
ncu --kernel-name matmul_tiled_kernel --set full \
    -o profiles/matmul_tiled_kernel \
    python benchmarks/bench_matmul.py --profile --shape 1024

# Softmax fused kernel
ncu --kernel-name softmax_fused_kernel --set full \
    -o profiles/softmax_fused_ncu \
    python benchmarks/bench_softmax_cross_entropy.py --profile --shape 2048 4096

# Cross entropy fused kernel
ncu --kernel-name cross_entropy_fused_kernel --set full \
    -o profiles/ce_fused_ncu \
    python benchmarks/bench_softmax_cross_entropy.py --profile --shape 2048 4096
```

Open `.ncu-rep` files in the Nsight Compute GUI (`ncu-ui`).

---

## Key metrics to inspect

### Occupancy

| Metric | CLI name | Meaning |
|--------|----------|---------|
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | Active warps / maximum possible. Low (<50%) means the kernel is latency-bound and needs more parallelism or fewer resources per block. |
| Theoretical occupancy | reported in "Launch Statistics" section | Ceiling set by register/shared-memory usage. If achieved << theoretical, the kernel stalls frequently. |

### Global memory throughput

| Metric | CLI name | Meaning |
|--------|----------|---------|
| DRAM read bandwidth | `dram__bytes_read.sum.per_second` | Compare to GPU peak (e.g. ~900 GB/s on A100). Elementwise and reduction kernels should approach 70–90 % of peak for large tensors. |
| DRAM write bandwidth | `dram__bytes_write.sum.per_second` | Same benchmark. |
| L1/TEX cache hit rate | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` / total | High hit rate means data reused across threads in the same warp. |
| Overall memory throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization; use together with roofline to determine bound. |

### Shared memory

| Metric | CLI name | Meaning |
|--------|----------|---------|
| Shared memory load transactions | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | High counts with bank conflicts degrade effective bandwidth. |
| Shared memory store transactions | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | |
| Shared memory bank conflicts (load) | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Should be 0 for sequential-access patterns (e.g. smem_m/smem_d in softmax). |

### Warp execution efficiency

| Metric | CLI name | Meaning |
|--------|----------|---------|
| Warp execution efficiency | `smsp__thread_inst_executed_pred_on.ratio` | Fraction of threads active per issued instruction. <75% suggests branch divergence. |
| Issued warp stall — memory | `smsp__average_warp_latency_per_inst_issued.ratio` (see "Warp State Statistics") | High memory stall means latency not hidden by other warps. |

### Memory load/store efficiency (coalescing)

| Metric | CLI name | Meaning |
|--------|----------|---------|
| Global load efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | 100% = fully coalesced 128-byte loads. <50% = strided/random access wasting bandwidth. |
| Global store efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` | Same for stores. |

### Compute throughput

| Metric | CLI name | Meaning |
|--------|----------|---------|
| FP32 FFMA instructions | `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | Actual multiply-add instructions issued. Compare to DRAM traffic to confirm compute vs memory bound. |
| Arithmetic intensity (roofline) | shown in "Roofline Analysis" section | FLOP / byte ratio; where the kernel sits relative to the ridge point. |

---

## Example profiling workflow

```
1.  Run benchmark normally to get baseline numbers:
        python benchmarks/bench_matmul.py

2.  Collect a quick Nsight Systems timeline to see kernel count and gaps:
        nsys profile --stats=true -o profiles/matmul_tiled \
            python benchmarks/bench_matmul.py --profile --shape 1024
    Open in nsys-ui. Confirm one kernel launch per matmul call (not three).

3.  Collect roofline data with Nsight Compute:
        ncu --set roofline --kernel-name matmul_tiled_kernel \
            -o profiles/matmul_roofline \
            python benchmarks/bench_matmul.py --profile --shape 1024
    Open in ncu-ui → "Roofline Analysis". Note whether the kernel is
    memory-bound or compute-bound at this size.

4.  If memory-bound: investigate coalescing and cache usage.
        ncu --set full --kernel-name matmul_tiled_kernel \
            -o profiles/matmul_full \
            python benchmarks/bench_matmul.py --profile --shape 1024
    Check "Memory Workload Analysis" → L1/L2 hit rates and global efficiency.

5.  If compute-bound: check occupancy and warp stalls.
    Look at "Scheduler Statistics" and "Warp State Statistics" in ncu-ui.

6.  After any kernel change, re-run step 1 to confirm the benchmark number
    improved, then re-collect the profile to verify the bottleneck shifted.
```

---

## Conv2d Backward Selectors

The current CUDA Conv2d backward defaults include specialized paths for common
MiniVGG/CIFAR 3x3 stride=1 padding=1 layers:

- grad-input: `conv2d_grad_input_3x3_pad1_kernel`, selected by default for
  eligible shapes after the kernel-position indexing fix and stronger
  MiniVGG-shape parity tests. The generic tiled shared-weight kernel remains
  available through `ORTHON_CUDA_CONV2D_GRAD_INPUT=optimized`.
- grad-weight: `conv2d_grad_weight_3x3_kernel`, selected by default for
  eligible shapes. It computes all nine weights for one
  `(output_channel, input_channel)` pair in one block and avoids atomics.

The fixed grad-input mapping is `oh = ih + pad_h - kh` and
`ow = iw + pad_w - kw`. A faster first version of `optimized_v2` used incorrect
backward-input indexing and hurt training; the current default is covered by
non-uniform and boundary-sensitive parity tests.

Selector examples:

```powershell
$env:ORTHON_CUDA_CONV2D_GRAD_INPUT="naive"        # or optimized, optimized_v2
$env:ORTHON_CUDA_CONV2D_GRAD_WEIGHT="grouped3x3" # or naive, optimized, optimized_v2

Remove-Item Env:\ORTHON_CUDA_CONV2D_GRAD_INPUT -ErrorAction SilentlyContinue
Remove-Item Env:\ORTHON_CUDA_CONV2D_GRAD_WEIGHT -ErrorAction SilentlyContinue
```

`ORTHON_CUDA_CONV2D_GRAD_WEIGHT=optimized_v2` remains experimental; it uses
atomics and was not clearly better than grouped3x3 on the measured MiniVGG
workloads.

---

## NVTX range markers (optional)

Wrap the hot loop with NVTX ranges to let `--capture-range=nvtx` filter the
profile to just the timed portion:

```python
import ctypes
nvtx = ctypes.CDLL("libnvToolsExt.so")   # nvToolsExt64_1.dll on Windows
nvtx.nvtxRangePushA(b"bench_loop")
# ... timed iterations ...
nvtx.nvtxRangePop()
```

Then:
```bash
nsys profile --capture-range=nvtx -o profiles/matmul_nvtx \
    python benchmarks/bench_matmul.py --profile --shape 1024
```

This trims import overhead and warmup iterations from the recorded timeline.
