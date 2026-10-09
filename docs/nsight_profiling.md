# Nsight profiling

Use NVIDIA Nsight Systems for launch timelines and Nsight Compute for kernel
counters. Run from the repository root against an intentionally selected CUDA
package and a usable physical GPU. These are external profiler tools; building
CUDA support alone does not make a capture executable.

## Prepare and smoke

```bash
python -m pip install . -C cmake.define.WITH_CUDA=ON -C cmake.define.WITH_NCCL=OFF
python -c "import orthon as o; print(o.cuda_enabled()); print(o.zeros([1]).to('cuda:0').to('cpu').item())"
python -m benchmarks.profile_cuda_workload --help
python -m benchmarks.profile_cuda_workload --workload conv2d --iterations 10 --warmup 2
python -m benchmarks.profile_cuda_workload --workload fusion --iterations 10 --warmup 2
python -m benchmarks.profile_cuda_workload --workload training --iterations 5 --warmup 1
```

`benchmarks/profile_cuda_workload.py` runs fixed deterministic hot loops, then
emits workload/config/provenance metadata through the shared JSON writer. Conv2d
runs forward/backward, fusion runs separate and fused Linear/ReLU, and training
runs a small synthetic CNN. The driver synchronizes the workload at boundaries;
it does not print normal benchmark latency or a comparison verdict. `--quick`
is bounded smoke with `representative=false`.

## Capture

These commands require the corresponding installed Nsight tool and CUDA hardware:

```bash
nsys profile --trace=cuda,osrt -o /tmp/orthon-training python -m benchmarks.profile_cuda_workload --workload training --iterations 20 --warmup 5
nsys profile --trace=cuda,osrt -o /tmp/orthon-conv2d python -m benchmarks.profile_cuda_workload --workload conv2d --iterations 50 --warmup 10
ncu --set full -o /tmp/orthon-conv2d python -m benchmarks.profile_cuda_workload --workload conv2d --iterations 20 --warmup 5
ncu --set full -o /tmp/orthon-fusion python -m benchmarks.profile_cuda_workload --workload fusion --iterations 20 --warmup 5
```

Nsight Systems owns CPU launch density, compute/copy/communication queues, gaps
and completion ordering. Inspect host materializations, allocation boundaries and
short-kernel gaps; correct ordering is not proof of useful compute overlap.
Nsight Compute owns kernel duration, occupancy, throughput, shared-memory usage
and warp stalls. Instrumentation changes execution; do not compare profiled wall
time with ordinary benchmark samples.

## Kernel and source attribution

Native CUDA kernels live in `src/backends/cuda/`. Semantic dispatch/graph creation
lives in `src/ops/`; queue and transfer ownership is in [CUDA runtime](cuda_runtime.md).
For Conv2d, useful names include `conv2d_forward_tiled_kernel`,
`conv2d_grad_input_3x3_pad1_kernel`, `conv2d_grad_input_tiled_kernel`,
`conv2d_grad_weight_3x3_kernel`, `conv2d_grad_weight_reduction_kernel`,
`conv2d_grad_bias_reduction_kernel` and `conv2d_grad_weight_bias_reduction_kernel`.
Shape-specific paths differ from generic paths; verify selected kernels in the
actual capture rather than assuming a name from a previous run.

Private `orthon._diagnostics` and supported Conv2d selector environments can force
routes for attribution. Benchmark support scopes and restores selector values.
Forced routes bypass ordinary selection and must be labeled diagnostic.
Application section timing (for example CIFAR profile/profile-sync) is an attribution
experiment with changing training state, not the shared latency protocol.
No NVTX ranges are currently emitted; use kernel names and process timelines.

## Record and interpret

Keep the exact command, workload configuration, commit/dirty state, package/extension
paths, CUDA/NCCL flags, hardware/driver/toolkit, external profiler version and
artifact paths. Shared metadata records available environment fields; absent
fields remain null. State profiler overhead/filters alongside observations.
Generated `.nsys-rep`, `.ncu-rep`, `.qdrep` and `.sqlite` files are untracked.

For ordinary completed-operation latency, use [benchmarking](benchmarking.md).
Capture on the hardware and backend configuration being studied. NCCL timelines
require an NCCL-enabled multi-GPU environment.
