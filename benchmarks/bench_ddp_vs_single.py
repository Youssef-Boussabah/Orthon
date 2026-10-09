"""Cloud-ready Orthon DDP-vs-single synthetic training benchmark.

Local direct DDP mode runs a world_size=1 smoke with the local backend. Real
multi-GPU scaling should be measured with the Orthon launcher and NCCL.
"""

from benchmarks._support.training import update_metadata, measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer, cuda_available

import argparse
import os

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.distributed.local_backend import create_store
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import SGD


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def local_rank():
    return env_int("LOCAL_RANK", env_int("RANK", 0))


def env_world_size():
    return env_int("WORLD_SIZE", 1)


def env_backend():
    return os.environ.get("ORTHON_DISTRIBUTED_BACKEND")


def sync(device):
    synchronizer(device, "orthon")()


def make_model(input_dim, hidden_size, num_classes, device):
    model = Sequential(
        Linear(input_dim, hidden_size),
        ReLU(),
        Linear(hidden_size, hidden_size),
        ReLU(),
        Linear(hidden_size, num_classes),
    )
    for index, param in enumerate(model.parameters()):
        param.fill_(0.01 * (index + 1))
    return model.to(device)


def make_batch(batch_size, input_dim, num_classes, device, rank=0):
    import numpy as np

    x = np.zeros((batch_size, input_dim), dtype=np.float32)
    targets = []
    for row in range(batch_size):
        label = (rank * batch_size + row) % num_classes
        targets.append(label)
        x[row, label % input_dim] = 1.0
        x[row, (label + num_classes) % input_dim] = 0.5
    tensor = ortho.from_numpy(x)
    return (tensor.to(device) if device.startswith("cuda") else tensor), targets


def train_step(model, optimizer, x, targets):
    optimizer.zero_grad()
    logits = model(x)
    loss = ortho.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss


def train_step_ddp(ddp_model, optimizer, x, targets):
    ddp_model.zero_grad()
    logits = ddp_model(x)
    loss = ortho.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss


def average_loss_across_ranks(loss_value, device, world_size):
    if world_size == 1:
        return loss_value
    t = ortho.full([1], loss_value).to(device)
    ddist.all_reduce(t, op="sum")
    return float(t.to("cpu").tolist()[0]) / float(world_size)


def run_single(args):
    device = args.device
    if device == "cuda" and not cuda_available("cuda:0"):
        return {"skipped": "cuda_unavailable"}
    model = make_model(args.input_dim, args.hidden_size, args.num_classes, device)
    optimizer = SGD(model.parameters(), lr=args.lr)
    x, targets = make_batch(args.batch_size, args.input_dim, args.num_classes, device)

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, x, targets)
    sync(device)

    last_loss = None
    with measured("bench_ddp_vs_single:start:run_single", lambda: sync(device), calls=args.steps, device=args.device, implementation="orthon", dtype="float32", config=vars(args)) as _region_start:
        for _ in range(args.steps):
            last_loss = train_step(model, optimizer, x, targets)
        sync(device)
    elapsed = _region_start["seconds"]
    return {
        "backend": "none",
        "world_size": 1,
        "rank": 0,
        "device": device,
        "local_batch_size": args.batch_size,
        "global_batch_size": args.batch_size,
        "ms_per_step": elapsed * 1000.0 / args.steps,
        "samples_per_sec": args.batch_size * args.steps / elapsed,
        "final_loss": float(last_loss.item()),
    }


def init_ddp_process_group(args):
    backend = env_backend()
    world_size = env_world_size()
    rank = env_int("RANK", 0)
    if backend is None and world_size > 1:
        backend = "nccl"

    if backend == "nccl" or world_size > 1:
        if backend != "nccl":
            raise RuntimeError(
                "DDP multi-rank benchmark requires the Orthon launcher with "
                "--backend nccl"
            )
        if not cuda_available("cuda:0") or not ortho.nccl_enabled():
            return None, {
                "skipped": "cuda_or_nccl_unavailable",
                "backend": "nccl",
                "world_size": world_size,
                "rank": rank,
            }
        ddist.init_process_group()
        return "nccl", None

    manager, store = create_store()
    ddist.init_process_group(backend="local", rank=0, world_size=1, store=store)
    return ("local", manager), None


def destroy_ddp_process_group(backend_info):
    try:
        ddist.destroy_process_group()
    finally:
        if isinstance(backend_info, tuple) and backend_info[0] == "local":
            backend_info[1].shutdown()


def run_ddp(args):
    backend_info, skipped = init_ddp_process_group(args)
    if skipped is not None:
        return skipped
    try:
        rank = ddist.get_rank()
        world_size = ddist.get_world_size()
        backend = backend_info[0] if isinstance(backend_info, tuple) else backend_info
        device = f"cuda:{local_rank()}" if backend == "nccl" else args.device
        if args.device == "cuda" and not cuda_available("cuda:0"):
            return {
                "skipped": "cuda_unavailable",
                "backend": backend,
                "world_size": world_size,
                "rank": rank,
            }

        model = make_model(args.input_dim, args.hidden_size, args.num_classes, device)
        ddp_model = DDP(
            model,
            sync_grads=True,
            bucket_size_mb=args.bucket_size_mb,
            overlap_grad_sync=args.overlap,
        )
        optimizer = SGD(ddp_model.parameters(), lr=args.lr)
        x, targets = make_batch(
            args.batch_size,
            args.input_dim,
            args.num_classes,
            device,
            rank=rank,
        )

        for _ in range(args.warmup_steps):
            train_step_ddp(ddp_model, optimizer, x, targets)
        sync(device)
        ddist.barrier()

        last_loss = None
        with measured("bench_ddp_vs_single:start:run_ddp", lambda: sync(device), calls=args.steps, device=args.device, implementation="orthon", dtype="float32", config=vars(args)) as _region_start:
            for _ in range(args.steps):
                last_loss = train_step_ddp(ddp_model, optimizer, x, targets)
            sync(device)
            ddist.barrier()
        elapsed = _region_start["seconds"]

        final_loss = average_loss_across_ranks(
            float(last_loss.item()),
            device,
            world_size,
        )
        return {
            "backend": backend,
            "world_size": world_size,
            "rank": rank,
            "device": device,
            "local_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "ms_per_step": elapsed * 1000.0 / args.steps,
            "samples_per_sec": args.batch_size * world_size * args.steps / elapsed,
            "final_loss": final_loss,
        }
    finally:
        destroy_ddp_process_group(backend_info)


def format_optional(value, precision=3):
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def print_summary(args, result):
    update_metadata(backend=result.get("backend"),world_size=result.get("world_size",1),
        device=result.get("device",args.device),transport="host/local transport" if result.get("backend")=="local" else result.get("backend"),
        bucket_mode="bucketed",overlap=args.overlap,batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,measured_steps=args.steps)
    if "skipped" in result:
        print(
            "benchmark_name=ddp_vs_single "
            f"mode={args.mode} "
            f"status=skipped "
            f"reason={result['skipped']} "
            f"backend={result.get('backend', 'n/a')} "
            f"world_size={result.get('world_size', 'n/a')}"
        )
        return

    speedup = None
    efficiency = None
    can_report_scaling = (
        args.mode == "ddp"
        and result["backend"] == "nccl"
        and result["world_size"] > 1
    )
    if can_report_scaling:
        if (
            args.baseline_samples_per_sec is not None
            and args.baseline_samples_per_sec > 0
        ):
            speedup = result["samples_per_sec"] / args.baseline_samples_per_sec
            efficiency = speedup / float(result["world_size"])
        elif args.baseline_ms_per_step is not None and args.baseline_ms_per_step > 0:
            speedup = (
                float(result["world_size"])
                * args.baseline_ms_per_step
                / result["ms_per_step"]
            )
            efficiency = speedup / float(result["world_size"])

    model_name = f"mlp_{args.input_dim}_{args.hidden_size}_{args.num_classes}"
    notes = (
        "single-process synthetic baseline"
        if args.mode == "single"
        else "DDP gradient sync is float32 only; real scaling requires NCCL cloud runs"
    )

    print("benchmark_name: ddp_vs_single")
    print(f"mode: {args.mode}")
    print(f"backend: {result['backend']}")
    print(f"world_size: {result['world_size']}")
    print(f"device: {result['device']}")
    print(f"model: {model_name}")
    print(f"local_batch_size: {result['local_batch_size']}")
    print(f"global_batch_size: {result['global_batch_size']}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"measured_steps: {args.steps}")
    print(f"ms_per_step: {result['ms_per_step']:.3f}")
    print(f"samples_per_sec: {result['samples_per_sec']:.2f}")
    print(f"final_loss: {result['final_loss']:.6f}")
    print(f"unverified_region_ratio: {format_optional(speedup)}")
    print(f"scaling_efficiency: {format_optional(efficiency)}")
    print(f"notes: {notes}")
    print(
        "csv_summary,"
        "benchmark_name=ddp_vs_single,"
        f"mode={args.mode},"
        f"backend={result['backend']},"
        f"world_size={result['world_size']},"
        f"device={result['device']},"
        f"model={model_name},"
        f"local_batch_size={result['local_batch_size']},"
        f"global_batch_size={result['global_batch_size']},"
        f"warmup_steps={args.warmup_steps},"
        f"measured_steps={args.steps},"
        f"ms_per_step={result['ms_per_step']:.3f},"
        f"samples_per_sec={result['samples_per_sec']:.2f},"
        f"final_loss={result['final_loss']:.6f},"
        f"unverified_region_ratio={format_optional(speedup)},"
        f"scaling_efficiency={format_optional(efficiency)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Orthon DDP-vs-single benchmark.")
    parser.add_argument("--mode", choices=["single", "ddp"], default="single")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--bucket-size-mb", type=float, default=25.0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--baseline-ms-per-step", type=float, default=None)
    parser.add_argument("--baseline-samples-per-sec", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    options(parser)
    args = configure(parser.parse_args())
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.bucket_size_mb <= 0:
        parser.error("--bucket-size-mb must be positive")
    return args


def main():
    args = parse_args()
    _ = args.seed  # Synthetic data is deterministic by construction.
    result = run_single(args) if args.mode == "single" else run_ddp(args)
    print_summary(args, result)


if __name__ == "__main__":
    run_measurement(main, "bench_ddp_vs_single")
