# Training workloads

`examples/` contains runnable applications and training demos. `benchmarks/`
contains measurement/profiling/reporting modules. Run from the repository root
against an installed Orthon package or explicit complete package stage. These
repository tools are not installed in the runtime wheel.

[Training stack](training_stack.md) explains the APIs; [benchmarking](benchmarking.md)
defines measurement semantics. Record model quality and timing with the
workload configuration; example output is not a performance contract.

## Applications

| File | Purpose | Requirements |
| --- | --- | --- |
| `examples/mnist_mlp.py` | CPU MLP with train_step/eval_step | MNIST; torch/torchvision data loading |
| `examples/cifar10_cnn.py` | Small/MiniVGG CNN, optimizer/schedule and accuracy controls | CPU/CUDA; CIFAR-10 or synthetic data |
| `examples/tiny_transformer_classifier.py` | Synthetic encoder classifier | Supported CPU/CUDA primitive profiles |
| `examples/tiny_causal_lm.py` | Causal additive mask, training and greedy generation | Synthetic; no sampling/top-k/beam API |
| `examples/tiny_text_classifier.py` | Embedding/attention classifier | Synthetic |
| `examples/nn_api_demo.py` | NN surface demonstration | CPU or available CUDA |
| `examples/cuda_amp_mlp.py` | Explicit F16 model, F32 masters and scaling | CUDA only |
| `examples/synthetic_mlp_checkpoint.py` | Synthetic training and checkpoint resume | CPU; local files |
| `examples/ddp_local_demo.py` | Local multiprocessing DDP | CPU and working local IPC |

Torch-suffixed application variants provide reference recipes. Dataset-driven
examples use torch/torchvision for data loading; comparisons additionally use
PyTorch compute. Core Orthon runtime and the README synthetic loop need neither.
Use each script's `--help` for controls and defaults; output metric files under
`examples/results/` are generated/gitignored, not current validation records.

```bash
python examples/mnist_mlp.py --epochs 1
python examples/cifar10_cnn.py --synthetic-data --epochs 1 --max-train-batches 2 --max-test-batches 2
python examples/tiny_transformer_classifier.py --device cpu --epochs 1
python examples/tiny_causal_lm.py --device cpu --epochs 1
python examples/cuda_amp_mlp.py --steps 5 --device cuda
python examples/synthetic_mlp_checkpoint.py --steps 5 --resume-smoke
python examples/ddp_local_demo.py
```

MNIST/CIFAR batch caps bound initial runs; zero caps request full epochs where
supported. Real datasets can download; synthetic CIFAR is the dataset-free smoke
route. A successful command is an application smoke, not a reproducible latency
comparison or a model-quality guarantee.

## Application diagnostics

CIFAR supports Small/MiniVGG models, Adam/SGD, augmentation and step/cosine
learning-rate schedules. Use `--synthetic-data --no-download` and batch caps for
repeatable smoke runs. Measurement and kernel attribution belong in the tools
below.

Tiny attention models reshape 3-D linear projections through 2-D Linear and
materialize after transpose as needed. They use composed norms/attention and
equal-batch 3-D matmul; they are not fused Transformer or FlashAttention kernels.
Causal generation uses device argmax and explicit no_grad/eval; it is greedy only.

## Measurement tools

Use module invocation for the benchmark package:

```bash
python -m benchmarks.bench_tiny_mlp_training --help
python -m benchmarks.bench_model_training --help
python -m benchmarks.bench_amp_vs_fp32 --help
python -m benchmarks.bench_checkpoint_overhead --help
python -m benchmarks.bench_ddp_vs_single --help
python -m benchmarks.benchmark_training --help
python -m benchmarks.compare_mnist_results --help
python -m benchmarks.profile_cuda_workload --workload training --quick
```

Latency comparisons use the shared timer; training measures continuous changing
state and reports that honestly. Legacy application-result ingestion requires
`--include-legacy` where supported. Local DDP timing measures host transport, not
NCCL throughput. Physical NCCL validation requires an NCCL-enabled CUDA build and multiple
suitable GPUs; see [distributed](distributed.md).

Private kernel selectors/forced diagnostic paths belong to benchmark experiments,
with scoped setting/restoration, not ordinary training quick starts. See
[Nsight profiling](nsight_profiling.md) for fixed hot-loop kernel attribution and
[checkpoint format](checkpoint_format.md) for local-file restore semantics.
