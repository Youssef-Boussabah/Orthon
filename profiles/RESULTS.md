# Orthon Profiling and Benchmark Results

Recorded results from the benchmark and profiling scripts in `benchmarks/`,
`examples/` and [`profiles/README.md`](README.md). Read them as what they are:
**hardware- and workload-specific microbenchmark measurements**, not a general
claim that Orthon is faster than PyTorch. PyTorch rows are reference
measurements taken by the same script under the same shape, dtype, device,
warmup and synchronization policy.

**Where this file sits.** It is the archive of per-script measurements from the
individual `benchmarks/` and `examples/` drivers, each run under its own
script's timing policy. The dedicated head-to-head comparison against PyTorch —
one harness, one timing contract, numerical equivalence proved before every
row, losses published alongside wins — is
[`PYTORCH_COMPARISON.md`](PYTORCH_COMPARISON.md), and that is where the
framework-versus-framework claims live.

## Systems

Three machines appear in this repository and they are **not** interchangeable:

| System | Role | GPU | Driver | CUDA / build |
| --- | --- | --- | --- | --- |
| Local laptop | **All published performance benchmarking** | NVIDIA GeForce RTX 4070 Laptop GPU | 596.08 | CUDA 13.2, 8 GB VRAM |
| Release validation machine | **Final multi-GPU correctness only — no performance numbers** | 2x NVIDIA A100-SXM4-80GB | 580.126.16 | CUDA 12.8, NCCL 2.27.3 |
| Earlier cloud machine | Historical, earlier project state | NVIDIA A100-SXM4-40GB | - | sm_80 |

The 2x A100-SXM4-80GB release machine is a **correctness** environment
([testing.md](../docs/testing.md), [cuda_runtime.md](../docs/cuda_runtime.md),
[ddp_cloud_validation.md](../docs/ddp_cloud_validation.md)); nothing in this
file or in the README quotes a timing from it. The A100-SXM4-40GB rows below
are a separate, older single-GPU cloud session against an **earlier project
state**, kept as a historical record of BF16 Tensor Core dispatch behaviour on
sm_80. They are not part of the published performance story and were not rerun
on the final tree.

Local environment: Windows 11, Python 3.12.6, PyTorch 2.13.0+cu130 (reference
rows), NumPy 2.5.2, Orthon 0.1.0.

## Summary

| Benchmark | System | Result | Comparison |
| --- | --- | ---: | --- |
| A100-40GB BF16 1024x1024 matmul (historical, earlier project state) | A100-SXM4-40GB | 0.472 ms custom tiled -> 0.030 ms cuBLAS dispatch | 15.7x faster than Orthon custom tiled BF16 kernel on this shape |
| RTX 4070 float32 1024x1024 matmul | RTX 4070 | 1.502 ms CUDA dispatch/custom tiled | 1952.57x vs Orthon CPU |
| RTX 4070 BF16 1024x1024 matmul | RTX 4070 | 1.501 ms custom tiled -> 0.101 ms cuBLAS dispatch | 14.82x faster than Orthon custom tiled BF16 kernel on this shape |
| RTX 4070 softmax 2048x4096 | RTX 4070 | 0.321 ms CUDA | 71.57x vs Orthon CPU |
| RTX 4070 cross entropy 2048x4096 | RTX 4070 | 0.096 ms CUDA | 238.04x vs Orthon CPU |
| RTX 4070 AMP MLP | RTX 4070 | 59.600 ms FP32 -> 6.968 ms AMP | 8.553x faster than Orthon FP32 on this synthetic MLP benchmark |
| MiniVGG CIFAR-10 | RTX 4070 | 84.11% best test accuracy | 23.9-28.9s epoch time |

## A100 BF16 Tensor Core Matmul (historical, earlier project state)

Measured once on the **A100-SXM4-40GB cloud machine** against an earlier
project state, kept for the sm_80 dispatch behaviour it records. It is not the
2x A100-SXM4-80GB release-validation machine, it is not part of the published
performance story, and it was not rerun on the final tree.

Command:

```bash
python benchmarks/bench_matmul.py --dtype bfloat16 --device cuda --clean-dispatch
```

BF16 2D matmul:

These are narrow microbenchmarks, not a general claim that Orthon
outperforms PyTorch. PyTorch timings are included as reference measurements
under the same script, shape, dtype, device, warmup, and synchronization
policy.

| Shape | Orthon custom tiled | Orthon cuBLAS dispatch | PyTorch BF16 CUDA timing from the same benchmark script |
| --- | ---: | ---: | ---: |
| 256x256 | 0.023 ms | 0.019 ms | 0.032 ms |
| 512x512 | 0.075 ms | 0.020 ms | 0.032 ms |
| 1024x1024 | 0.472 ms | 0.030 ms | 0.039 ms |

Small batched BF16 fallback:

| Shape | Orthon custom tiled | Orthon dispatch path | PyTorch BF16 CUDA timing from the same benchmark script |
| --- | ---: | ---: | ---: |
| 32x64x64 batched BF16 | 0.017 ms | 0.016 ms custom fallback | 0.044 ms |

Large BF16 2D GEMMs use cuBLAS Tensor Core dispatch with
`CUBLAS_COMPUTE_32F_FAST_16BF`. The small batched BF16 case stays on the custom
tiled path because it is below the configured work threshold.

Overhead profile for 1024x1024 BF16:

| Row | Time |
| --- | ---: |
| dispatch cuBLAS | 0.030 ms |
| forced cuBLAS | 0.031 ms |
| cuBLAS event-only | 0.019 ms |
| PyTorch BF16 CUDA timing from the same benchmark script | 0.038 ms |

## Local RTX 4070 Benchmarks

| Item | Value |
| --- | --- |
| Orthon import | installed `orthon` 0.1.0 wheel built from this repo |
| `orthon.cuda_enabled()` | `True` |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |

### Matmul

Command:

```powershell
python benchmarks/bench_matmul.py --device cuda
```

1024x1024 float32 matmul:

| Backend | Device | Kernel | Time | Speedup vs Orthon CPU |
| --- | --- | --- | ---: | ---: |
| Orthon | CPU | plain | 2932.746 ms | 1.00x |
| Orthon | CUDA | naive | 8.373 ms | 350.28x |
| Orthon | CUDA | dispatch/custom tiled float32 | 1.502 ms | 1952.57x |
| NumPy | CPU | BLAS | 11.417 ms | 256.87x |
| PyTorch | CPU | BLAS | 3.633 ms | 807.30x |
| PyTorch | CUDA | cuBLAS float32 | 0.264 ms | 11109.73x |

### BF16 Clean Dispatch

Command:

```powershell
python benchmarks/bench_matmul.py --dtype bfloat16 --device cuda --clean-dispatch
```

| Shape | Orthon custom tiled | Orthon cuBLAS dispatch | PyTorch BF16 CUDA timing from the same benchmark script |
| --- | ---: | ---: | ---: |
| 256x256 | 0.041 ms | 0.026 ms | 0.037 ms |
| 512x512 | 0.213 ms | 0.057 ms | 0.047 ms |
| 1024x1024 | 1.501 ms | 0.101 ms | 0.751 ms |
| 32x64x64 batched BF16 | 0.181 ms | 0.182 ms custom fallback | 0.075 ms |

The PyTorch column here is `bench_matmul.py`'s own reference timing, which
clocks each call individually and therefore charges per-call synchronization
and Python overhead to whichever side does more of it. Do not read it as a
framework comparison: under the dedicated harness, which times batches of
repeated calls,
[`PYTORCH_COMPARISON.md`](PYTORCH_COMPARISON.md) measures PyTorch BF16
1024x1024 at ~0.071 ms and Orthon at ~0.071 ms — a tie, not a 7x gap. The
0.751 ms figure is an artefact of this script's timing policy, not a property
of PyTorch.

### Softmax

Command:

```powershell
python benchmarks/bench_softmax_cross_entropy.py --device cuda
```

2048x4096 softmax:

| Backend | Device | Time | Speedup vs Orthon CPU |
| --- | --- | ---: | ---: |
| Orthon | CPU | 22.986 ms | 1.00x |
| Orthon | CUDA | 0.321 ms | 71.57x |
| PyTorch | CPU | 1.544 ms | 14.88x |
| PyTorch | CUDA | 0.331 ms | 69.35x |

### Cross Entropy

2048x4096 cross entropy:

| Backend | Device | Time | Speedup vs Orthon CPU |
| --- | --- | ---: | ---: |
| Orthon | CPU | 22.862 ms | 1.00x |
| Orthon | CUDA | 0.096 ms | 238.04x |
| PyTorch | CPU | 1.774 ms | 12.88x |
| PyTorch | CUDA | 0.342 ms | 66.77x |

### Fused Linear and ReLU

Command:

```powershell
python examples/benchmark_fusion.py --iters 100 --warmup 10
```

Average forward latency:

| Path | Device | Latency |
| --- | --- | ---: |
| separate `matmul + bias + relu` | CPU | 0.1966 ms |
| fused `linear + bias + relu` | CPU | 0.1550 ms |
| torch equivalent | CPU | 0.0247 ms |
| separate `matmul + bias + relu` | CUDA | 0.0473 ms |
| fused `linear + bias + relu` | CUDA | 0.0335 ms |
| torch equivalent | CUDA | 0.0578 ms |

## MiniVGG CIFAR-10

Architecture:

```text
Block 1: Conv2d 3->32, Conv2d 32->32, MaxPool
Block 2: Conv2d 32->64, Conv2d 64->64, MaxPool
Block 3: Conv2d 64->128, Conv2d 128->128, MaxPool
Classifier: Linear(128*4*4 -> hidden), ReLU, Linear(hidden -> 10)
```

Full training command:

```powershell
python .\examples\cifar10_cnn.py --model minivgg --epochs 30 --batch-size 128 --device cuda --optimizer adam --lr 0.001 --adam-eps 0.00000001 --weight-decay 0.0005 --augment --lr-schedule step --lr-step-epochs 15,25 --lr-gamma 0.1 --max-train-batches 0 --max-test-batches 0
```

Best test accuracy was 84.11%, reached at epoch 27. Per-epoch times ranged
from 23.9s to 28.9s over the 30 epochs.

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy | Time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 0.3734 | 86.89% | 0.4893 | 84.11% | 27.7s |
| 30 | 0.3695 | 87.09% | 0.4901 | 83.83% | 27.3s |

100-batch profile command:

```powershell
$env:ORTHON_PROFILE_CONV2D="1"
python .\examples\cifar10_cnn.py --model minivgg --epochs 1 --batch-size 128 --device cuda --optimizer adam --lr 0.001 --adam-eps 0.00000001 --weight-decay 0.0005 --augment --max-train-batches 100 --max-test-batches 20 --profile --profile-sync
```

100-batch profile:

| Section | Time | Percent |
| --- | ---: | ---: |
| data_loading | about 1.6691s | 24.7% |
| label_conversion | about 0.0015s | 0.0% |
| input_conversion | about 0.0627s | 0.9% |
| forward | about 1.6384s | 24.2% |
| loss | about 0.0357s | 0.5% |
| backward | about 2.4906s | 36.8% |
| optimizer | about 0.3171s | 4.7% |
| metrics | about 0.0278s | 0.4% |
| evaluation | about 0.5171s | 7.6% |

Profile summary: 100 train batches, 12,800 train samples, 20 eval batches, and
2,560 eval samples. `seconds_per_train_batch` was `0.062430s`;
`seconds_per_train_sample` was `0.000488s`.

Bottlenecks are backward at about 37%, data loading at about 25%, forward at
about 24%, and optimizer at about 4.7%.

## Conv2d Backward

Command:

```powershell
python examples/benchmark_conv2d.py --forward-iters 50 --backward-iters 100
```

Orthon uses `optimized_v2` grad-input and `grouped3x3` grad-weight for
eligible 3x3 stride=1 padding=1 CUDA workloads.

Average per iteration:

| Config | grad-input default | grad-weight default | grad-bias default | full backward (opt) | Torch CUDA forward | Torch CUDA backward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar_conv1 | 0.027 ms | 0.048 ms | 0.034 ms | 0.153 ms | 0.068 ms | 0.407 ms |
| cifar_conv2 | 0.030 ms | 0.043 ms | 0.028 ms | 0.197 ms | 0.064 ms | 0.452 ms |
| synthetic | 0.598 ms | 0.138 ms | 0.034 ms | 0.455 ms | 0.113 ms | 0.552 ms |
| minivgg_32 | 1.870 ms | 2.940 ms | 0.137 ms | 8.853 ms | 0.345 ms | 1.166 ms |
| minivgg_64 | 1.822 ms | 2.956 ms | 0.071 ms | 7.755 ms | 0.237 ms | 0.819 ms |
| minivgg_128 | 1.836 ms | 3.079 ms | 0.027 ms | 9.787 ms | 0.195 ms | 0.677 ms |

Selector comparison:

| Config | gi naive | gi opt | gi opt_v2 | gw naive | gw opt | gw grouped | gw opt_v2 | bwd naive-gi | bwd naive-gw | bwd fused | bwd opt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar_conv1 | 0.037 ms | 0.037 ms | 0.030 ms | 0.442 ms | 0.083 ms | 0.072 ms | 0.044 ms | 0.193 ms | 0.605 ms | 0.188 ms | 0.153 ms |
| cifar_conv2 | 0.089 ms | 0.069 ms | 0.026 ms | 0.390 ms | 0.109 ms | 0.061 ms | 0.041 ms | 0.266 ms | 0.560 ms | 0.197 ms | 0.197 ms |
| synthetic | 0.914 ms | 0.388 ms | 0.110 ms | 0.940 ms | 0.530 ms | 0.171 ms | 0.127 ms | 0.718 ms | 1.149 ms | 0.822 ms | 0.455 ms |

For MiniVGG-shaped Conv2d workloads, PyTorch CUDA is faster in this benchmark.

MiniVGG 100-batch Conv2d aggregation:

| Shape | Calls | Total | Avg/call | grad-input | grad-weight | grad-bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=128 C=128 H=8 W=8 O=128 K=3x3 | 100 | 0.519332s | 5.193317ms | 0.182050s | 0.328657s | 0.007911s |
| N=128 C=64 H=16 W=16 O=64 K=3x3 | 100 | 0.507695s | 5.076955ms | 0.182123s | 0.310570s | 0.014176s |
| N=128 C=32 H=32 W=32 O=32 K=3x3 | 100 | 0.507288s | 5.072884ms | 0.187794s | 0.304399s | 0.014160s |
| N=128 C=64 H=8 W=8 O=128 K=3x3 | 100 | 0.264913s | 2.649131ms | 0.096833s | 0.160430s | 0.007059s |
| N=128 C=32 H=16 W=16 O=64 K=3x3 | 100 | 0.258372s | 2.583716ms | 0.096923s | 0.154609s | 0.006199s |
| N=128 C=3 H=32 W=32 O=32 K=3x3 | 100 | 0.158812s | 1.588122ms | 0.020343s | 0.138077s | fused |

## AMP vs FP32

Command:

```powershell
python benchmarks/bench_amp_vs_fp32.py --device cuda --preset medium
```

Single-GPU, non-DDP AMP benchmark:

| Metric | FP32 | AMP |
| --- | ---: | ---: |
| ms per step | 59.600 | 6.968 |
| samples/sec | 17,181.16 | 146,952.32 |
| final loss | 0.000025 | 0.000025 |

On this synthetic MLP benchmark, Orthon AMP measured 8.553x faster than
Orthon FP32. The gradient scaler stayed at its initial scale of 128 for the
whole run. This is not a general AMP speedup claim.

## Nsight Profiling

| Tool | Purpose |
| --- | --- |
| Nsight Systems (`nsys`) | kernel timeline and execution tracing |
| Nsight Compute (`ncu`) | hardware counters and roofline analysis |

Nsight reports (`matmul_roofline_one.ncu-rep`, `softmax_fused_ncu_one.ncu-rep`,
`ce_fused_ncu_one.ncu-rep`) are local artifacts produced by the workflows in
[`profiles/README.md`](README.md); they are not tracked in this repository.
Fused softmax and cross-entropy kernels were observed in Nsight Systems traces.

## Scope

Results are benchmark-specific and hardware-specific. PyTorch rows are reference
measurements for the same benchmark commands, taken under each script's own
timing policy — which is not the same thing as the controlled head-to-head in
[`PYTORCH_COMPARISON.md`](PYTORCH_COMPARISON.md). Where the two disagree, the
comparison document is authoritative for framework-versus-framework claims.

The historical A100-SXM4-40GB rows and the RTX 4070 rows were measured on
different systems and different project states. No timing anywhere in this
repository comes from the 2x A100-SXM4-80GB release-validation machine; that
machine produced multi-GPU **correctness** evidence only. DDP/NCCL benchmarks
are not included in this file. Multi-node training is not claimed. Conv2d
MiniVGG-shaped backward is a remaining optimization target.
