"""Compare FP32 DDP against explicit FP16 AMP DDP on the large MLP workload.

Run with the distributed launcher for multi-rank NCCL measurements, for example:
    python -m orthon.distributed.run --nproc_per_node 2 --backend nccl \
        benchmarks/bench_amp_ddp_vs_fp32_ddp.py
"""

from benchmarks._support.training import update_metadata, measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer

import argparse
import os

import orthon as ortho
import orthon.distributed as ddist
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.distributed import DistributedDataParallel as DDP
from orthon.nn import Module, Parameter, ReLU, Sequential
from orthon.optim import SGD


class ConstantLinear(Module):
    def __init__(self, in_features, out_features, weight_value):
        super().__init__()
        self.weight = Parameter(ortho.full([in_features, out_features], weight_value))
        self.bias = Parameter(ortho.zeros([out_features]))

    def forward(self, x):
        return x @ self.weight + self.bias


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def local_rank():
    return env_int("LOCAL_RANK", env_int("RANK", 0))


def make_model(args, device, dtype):
    modules = []
    in_features = args.input_dim
    weight_value = 1.0 / max(float(args.input_dim), float(args.hidden_size))
    for _ in range(args.layers):
        modules.append(ConstantLinear(in_features, args.hidden_size, weight_value))
        modules.append(ReLU())
        in_features = args.hidden_size
    modules.append(ConstantLinear(in_features, args.output_dim, weight_value))
    model = Sequential(*modules).to(device)
    if dtype == "float16":
        model.half()
    return model


def make_batch(args, device, dtype):
    x = ortho.full([args.batch_size, args.input_dim], args.input_value).to(device)
    if dtype != "float32":
        x = ortho.cast(x, dtype)
    return x


def loss_fn(model, x):
    out = model(x)
    return ortho.mean(out * out)


def sync(device):
    synchronizer(device, "orthon")()


def scalar_float(tensor):
    return float(ortho.cast(tensor.to("cpu"), "float32").item())


def average_scalar(value, device):
    if ddist.get_world_size() == 1:
        return value
    t = ortho.full([1], float(value)).to(device)
    ddist.all_reduce(t, op="sum")
    return float(t.to("cpu").tolist()[0]) / float(ddist.get_world_size())


def run_fp32_ddp(args, device):
    model = make_model(args, device, "float32")
    ddp_model = DDP(
        model,
        sync_grads=True,
        bucket_size_mb=args.bucket_size_mb,
        overlap_grad_sync=args.overlap,
    )
    optimizer = SGD(ddp_model.parameters(), lr=args.lr)
    x = make_batch(args, device, "float32")

    for _ in range(args.warmup_steps):
        ddp_model.zero_grad()
        loss = loss_fn(ddp_model, x)
        loss.backward()
        if args.overlap:
            ddp_model.wait_for_gradient_sync()
        optimizer.step()
    sync(device)
    ddist.barrier()

    last_loss = None
    with measured("bench_amp_ddp_vs_fp32_ddp:start:run_fp32_ddp", lambda: sync(device), calls=args.steps, device=device, implementation="orthon", dtype="float32", config=vars(args)) as _region_start:
        for _ in range(args.steps):
            ddp_model.zero_grad()
            last_loss = loss_fn(ddp_model, x)
            last_loss.backward()
            if args.overlap:
                ddp_model.wait_for_gradient_sync()
            optimizer.step()
        sync(device)
        ddist.barrier()
    elapsed = _region_start["seconds"]
    return {
        "ms_per_step": elapsed * 1000.0 / args.steps,
        "samples_per_sec": args.batch_size * ddist.get_world_size() * args.steps / elapsed,
        "final_loss": average_scalar(scalar_float(last_loss), device),
    }


def run_amp_ddp(args, device):
    model = make_model(args, device, "float16")
    ddp_model = DDP(model, sync_grads=False)
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=args.lr)
    scaler = GradScaler(init_scale=args.init_scale, growth_interval=args.growth_interval)
    x = make_batch(args, device, "float16")

    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        loss = loss_fn(ddp_model, x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    sync(device)
    ddist.barrier()

    last_loss = None
    with measured("bench_amp_ddp_vs_fp32_ddp:start:run_amp_ddp", lambda: sync(device), calls=args.steps, device=device, implementation="orthon", dtype="float16", config=vars(args)) as _region_start:
        for _ in range(args.steps):
            optimizer.zero_grad()
            last_loss = loss_fn(ddp_model, x)
            scaler.scale(last_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        sync(device)
        ddist.barrier()
    elapsed = _region_start["seconds"]
    return {
        "ms_per_step": elapsed * 1000.0 / args.steps,
        "samples_per_sec": args.batch_size * ddist.get_world_size() * args.steps / elapsed,
        "final_loss": average_scalar(scalar_float(last_loss), device),
        "scaler_final_scale": scaler.scale_factor,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Orthon FP32 DDP vs FP16 AMP DDP benchmark.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-value", type=float, default=0.01)
    parser.add_argument("--bucket-size-mb", type=float, default=25.0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--init-scale", type=float, default=128.0)
    parser.add_argument("--growth-interval", type=int, default=2000)
    options(parser)
    args = configure(parser.parse_args())
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    return args


def main():
    args = parse_args()
    ddist.init_process_group()
    try:
        rank = ddist.get_rank()
        world_size = ddist.get_world_size()
        update_metadata(backend="nccl",world_size=world_size,device_profile="cuda",
            bucket_mode="bucketed",overlap=getattr(args,"overlap",args.mode=="overlap" if hasattr(args,"mode") else False),
            batch_size=args.batch_size,warmup_steps=getattr(args,"warmup_steps",getattr(args,"warmup_iters",0)),
            measured_steps=getattr(args,"steps",getattr(args,"timed_iters",0)))
        device = f"cuda:{local_rank()}"
        fp32 = run_fp32_ddp(args, device)
        amp = run_amp_ddp(args, device)
        speedup = fp32["ms_per_step"] / amp["ms_per_step"] if amp["ms_per_step"] > 0 else 0.0
        if rank == 0:
            print("benchmark_name: amp_ddp_vs_fp32_ddp")
            print("backend: nccl")
            print(f"world_size: {world_size}")
            print(f"model: mlp_{args.input_dim}_{args.hidden_size}_{args.output_dim}")
            print(f"local_batch_size: {args.batch_size}")
            print(f"global_batch_size: {args.batch_size * world_size}")
            print(f"warmup_steps: {args.warmup_steps}")
            print(f"measured_steps: {args.steps}")
            print(f"fp32_ddp_ms_per_step: {fp32['ms_per_step']:.3f}")
            print(f"amp_ddp_ms_per_step: {amp['ms_per_step']:.3f}")
            print(f"fp32_ddp_samples_per_sec: {fp32['samples_per_sec']:.2f}")
            print(f"amp_ddp_samples_per_sec: {amp['samples_per_sec']:.2f}")
            print(f"unverified_region_ratio: {speedup:.3f}")
            print(f"fp32_ddp_final_loss: {fp32['final_loss']:.6f}")
            print(f"amp_ddp_final_loss: {amp['final_loss']:.6f}")
            print(f"scaler_final_scale: {amp['scaler_final_scale']:.1f}")
            print(
                "notes: AMP DDP syncs FP32 master gradients only; "
                "low-precision DDP gradient sync remains disabled"
            )
            print(
                "csv_summary,"
                "benchmark_name=amp_ddp_vs_fp32_ddp,"
                f"world_size={world_size},"
                f"model=mlp_{args.input_dim}_{args.hidden_size}_{args.output_dim},"
                f"local_batch_size={args.batch_size},"
                f"global_batch_size={args.batch_size * world_size},"
                f"warmup_steps={args.warmup_steps},"
                f"measured_steps={args.steps},"
                f"fp32_ddp_ms_per_step={fp32['ms_per_step']:.3f},"
                f"amp_ddp_ms_per_step={amp['ms_per_step']:.3f},"
                f"fp32_ddp_samples_per_sec={fp32['samples_per_sec']:.2f},"
                f"amp_ddp_samples_per_sec={amp['samples_per_sec']:.2f},"
                f"unverified_region_ratio={speedup:.3f},"
                f"fp32_ddp_final_loss={fp32['final_loss']:.6f},"
                f"amp_ddp_final_loss={amp['final_loss']:.6f},"
                f"scaler_final_scale={amp['scaler_final_scale']:.1f}"
            )
    finally:
        ddist.destroy_process_group()


if __name__ == "__main__":
    run_measurement(main, "bench_amp_ddp_vs_fp32_ddp")
