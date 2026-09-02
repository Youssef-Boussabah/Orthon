# Nsight Profiling Workflow

How to inspect Orthon's CUDA execution with NVIDIA Nsight Systems (timelines)
and Nsight Compute (per-kernel counters). These runs are for profiling only —
do not use them to compare final model quality; the workloads are
intentionally small and deterministic so captures are easy to reproduce.

Generated reports land in `profiles/` and are gitignored
(`*.nsys-rep`, `*.ncu-rep`, `*.qdrep`, `*.sqlite`); only the written notes in
`profiles/` are tracked. Recorded numbers from these workflows are collected
in [../profiles/RESULTS.md](../profiles/RESULTS.md), and the kernels being
profiled are described in [cuda_runtime.md](cuda_runtime.md).

## Verify CUDA Build

From Windows PowerShell:

```powershell
python -m pip install -e . -C cmake.define.WITH_CUDA=ON
python -c "import orthon; print('cuda_enabled=', orthon.cuda_enabled()); x=orthon.zeros([1]).to('cuda'); orthon.cuda_synchronize(); print('cuda smoke ok')"
```

If the import fails because CUDA DLLs are missing, confirm that `$env:CUDA_PATH`
points at the CUDA Toolkit install and that `$env:CUDA_PATH\bin` exists.

## Workload Driver

Use `examples/profile_cuda_workload.py` directly before opening Nsight:

```powershell
python examples/profile_cuda_workload.py --workload conv2d --iterations 10 --warmup 2
python examples/profile_cuda_workload.py --workload fusion --iterations 10 --warmup 2
python examples/profile_cuda_workload.py --workload training --iterations 5 --warmup 1
```

Supported workloads:

- `conv2d`: optimized CUDA Conv2d forward plus backward.
- `fusion`: separate `matmul + bias + relu`, then `fused_linear_relu`.
- `training`: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> CrossEntropy -> SGD.

The script synchronizes before and after timed iterations and prints the shape
configuration plus a timing summary.

## Nsight Systems

Use Nsight Systems to inspect timeline behavior: CPU launch density, CUDA kernel
sequence, memory copies, and CPU/GPU overlap.

PowerShell examples:

```powershell
New-Item -ItemType Directory -Force profiles
nsys profile -o profiles/orthon_training --trace=cuda,nvtx,osrt python examples/profile_cuda_workload.py --workload training --iterations 20 --warmup 5
nsys profile -o profiles/orthon_conv2d --trace=cuda,nvtx,osrt python examples/profile_cuda_workload.py --workload conv2d --iterations 50 --warmup 10
nsys profile -o profiles/orthon_fusion --trace=cuda,nvtx,osrt python examples/profile_cuda_workload.py --workload fusion --iterations 50 --warmup 10
```

Open the generated `.nsys-rep` file in Nsight Systems.

Screenshots to capture:

- CUDA kernel timeline for the full run.
- Training loop timeline with forward, backward, and optimizer activity visible.
- CPU/GPU overlap view.
- Kernel launch density around Conv2d backward.
- Memory copy events, especially host/device transfers.
- Any long CPU gaps between CUDA kernels.

Metrics and interpretation:

- Many short kernels with CPU gaps suggest launch/autograd overhead.
- For current MiniVGG/CIFAR-style 3x3 stride=1 padding=1 layers, expect
  `conv2d_grad_input_3x3_pad1_kernel` and grouped
  `conv2d_grad_weight_3x3_kernel` by default.
- Frequent memcpy events inside the timed region suggest accidental host/device reads.
- Low GPU occupancy across the timeline can be normal for these small workloads, but it explains weak throughput scaling.
- For end-to-end training attribution, prefer the CIFAR script's `--profile
  --profile-sync` path before collecting a full Nsight timeline; async timing
  can otherwise misattribute queued CUDA work to later Python sections.

## Nsight Compute

Use Nsight Compute for kernel-level analysis. Start with Conv2d because it has
the optimized forward, grad-weight, and grad-bias kernels.

PowerShell examples:

```powershell
New-Item -ItemType Directory -Force profiles
ncu --set full -o profiles/orthon_conv2d python examples/profile_cuda_workload.py --workload conv2d --iterations 100 --warmup 10
ncu --set full -o profiles/orthon_fusion python examples/profile_cuda_workload.py --workload fusion --iterations 100 --warmup 10
```

If the report is too large or slow, reduce the captured kernels with Nsight
Compute's UI filters after collecting one full run.

Screenshots to capture:

- Kernel duration table.
- Achieved occupancy.
- Memory throughput.
- Global load/store efficiency.
- Shared memory usage for optimized Conv2d forward and reduction kernels.
- Top bottleneck section.
- Source or SASS view for the longest Orthon kernels if available.

Metrics and interpretation:

- `conv2d_forward_tiled_kernel`: check shared memory usage, global load efficiency, and occupancy.
- `conv2d_grad_input_3x3_pad1_kernel`: default grad-input path for 3x3 stride=1 padding=1. It owns one grad-input element per thread, caches `C_out * 9` weights in shared memory, unrolls the 3x3 accumulation (mapping `oh = ih + pad_h - kh`, `ow = iw + pad_w - kw`), and avoids atomics.
- `conv2d_grad_input_tiled_kernel`: generic optimized grad-input path selected with `ORTHON_CUDA_CONV2D_GRAD_INPUT=optimized`; check shared filter-cache behavior, global store efficiency, and branch cost from stride/padding checks.
- `conv2d_grad_weight_3x3_kernel`: default grad-weight path for 3x3 stride=1 padding=1. It computes all nine weights for one `(output_channel, input_channel)` pair in one block and avoids atomics.
- `conv2d_grad_weight_reduction_kernel`: generic optimized grad-weight path selected with `ORTHON_CUDA_CONV2D_GRAD_WEIGHT=optimized`; check reduction efficiency and memory throughput.
- `conv2d_grad_bias_reduction_kernel`: standalone optimized bias path; if it appears high, inspect occupancy and launch overhead.
- `conv2d_grad_weight_bias_reduction_kernel`: reduced-launch path selected with `ORTHON_CUDA_CONV2D_BIAS_GRAD=fused` and used by default for small CIFAR-style channel counts; inspect whether the saved launch offsets extra work inside the weight kernel.
- Compare default Conv2d backward kernels against selectors (`ORTHON_CUDA_CONV2D_GRAD_INPUT=naive|optimized|optimized_v2`, `ORTHON_CUDA_CONV2D_GRAD_WEIGHT=naive|optimized|grouped3x3|optimized_v2`, `ORTHON_CUDA_CONV2D_BIAS_GRAD=naive|optimized|fused`) to separate launch overhead from kernel work.
- Fused Linear+Bias+ReLU should show fewer Orthon kernels than the separate path, but this prototype is not expected to beat mature PyTorch kernels.

## NVTX

The commands include `--trace=cuda,nvtx,osrt`, but Orthon does not currently
emit NVTX ranges. Adding NVTX markers would require a small optional dependency
or CUDA runtime annotations in the framework. That is future work; for now use
kernel names and timeline regions.

## Reporting Template

When saving screenshots, include:

- Hardware: GPU name, driver version, CUDA Toolkit version.
- Build command used for Orthon.
- Exact profile command.
- Workload config printed by `profile_cuda_workload.py`.
- Systems observations: launch density, CPU gaps, memcpy events.
- Compute observations: longest kernels, occupancy, memory throughput, bottleneck section.
- Recommendation: continue kernel work, reduce launch/autograd overhead, or stop and document current performance.

## Limitations

- Workloads are synthetic and intentionally small.
- Timings include Python and dynamic-autograd overhead.
- Nsight Compute can significantly slow execution; compare relative kernel behavior, not wall time from profiled runs.
- No NVTX ranges are emitted yet, so captures rely on kernel names and Python process regions.
