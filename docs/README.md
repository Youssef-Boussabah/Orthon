# Documentation index

Current source and executable tests define behavior. These guides describe
Orthon 1.0.0rc1 APIs, architecture and development workflows.

Repository information: [contributing](../CONTRIBUTING.md) and
[MIT license](../LICENSE).

## Current documentation

Start with the [project README](../README.md). The current guides are:

- [architecture.md](architecture.md) — Tensor ownership, dispatch, planning and runtime services.
- [autograd.md](autograd.md) — GraphTask execution, mutation watches and gradient commit boundaries.
- [cuda_runtime.md](cuda_runtime.md) — Explicit queues, completion, memory reuse and transfers.
- [distributed.md](distributed.md) — ProcessGroup, Reducer, Work and NCCL requirements.
- [checkpoint_format.md](checkpoint_format.md) — Single-file serialization format and restore transaction.
- [api.md](api.md) — Python exports and signatures.
- [training_stack.md](training_stack.md) — Module, optimizer and AMP owner/state protocols.
- [training_workloads.md](training_workloads.md) — Applications, dataset recipes and measurement-tool boundaries.
- [testing.md](testing.md) — Five test layers, package profiles, CI and hardware gates.
- [benchmarking.md](benchmarking.md) — Correctness-first latency, provenance, JSON and stateful regions.
- [nsight_profiling.md](nsight_profiling.md) — Fixed hot loops and external kernel/timeline attribution.
- [cuda_coverage.md](cuda_coverage.md) — Supported dtype/operator matrix and restrictions.
