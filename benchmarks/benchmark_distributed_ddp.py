from benchmarks._support.training import update_metadata, measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer, cuda_available
import argparse
import os
from benchmarks._support.timing import quartiles

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import SGD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Focused single-node NCCL DDP benchmark for Orthon."
    )
    parser.add_argument(
        "--mode",
        choices=("blocking", "bucketed", "overlap"),
        default="bucketed",
        help="DDP synchronization mode to benchmark.",
    )
    parser.add_argument("--bucket-size-mb", type=float, default=4.0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    options(parser)
    args = configure(parser.parse_args())
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be >= 0")
    if args.timed_iters < 1:
        parser.error("--timed-iters must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.hidden_size < 1:
        parser.error("--hidden-size must be >= 1")
    if args.depth < 1:
        parser.error("--depth must be >= 1")
    if args.bucket_size_mb <= 0:
        parser.error("--bucket-size-mb must be positive")
    return args


def local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def sync_cuda():
    synchronizer("cuda:0", "orthon")()


def make_model(hidden_size, depth, device):
    layers = []
    for index in range(depth):
        layers.append(Linear(hidden_size, hidden_size))
        if index != depth - 1:
            layers.append(ReLU())
    model = Sequential(*layers)
    for param_index, parameter in enumerate(model.parameters()):
        parameter.fill_(0.01 * (param_index + 1))
    return model.to(device)


def make_ddp(model, args):
    if args.mode == "blocking":
        return DDP(model, sync_grads=True)
    if args.mode == "bucketed":
        return DDP(model, sync_grads=True, bucket_size_mb=args.bucket_size_mb)
    return DDP(
        model,
        sync_grads=True,
        bucket_size_mb=args.bucket_size_mb,
        overlap_grad_sync=True,
    )


def run_step(ddp_model, optimizer, x, target_neg):
    pred = ddp_model(x)
    diff = pred + target_neg
    loss = ortho.mean(diff * diff)
    sync_cuda()
    with measured("benchmark_distributed_ddp:backward_start", lambda: sync_cuda(), device="cuda:0", calls=1) as _region_backward_start:
        loss.backward()
        sync_cuda()
    backward_elapsed = _region_backward_start["seconds"]
    optimizer.step()
    optimizer.zero_grad()
    return backward_elapsed


def main():
    args = parse_args()
    if not cuda_available("cuda:0") or not ortho.nccl_enabled():
        raise RuntimeError("benchmark_distributed_ddp requires a CUDA+NCCL Orthon build")

    ddist.init_process_group()
    try:
        rank = ddist.get_rank()
        world_size = ddist.get_world_size()
        update_metadata(backend="nccl",world_size=world_size,device_profile="cuda",
            bucket_mode="bucketed",overlap=getattr(args,"overlap",args.mode=="overlap" if hasattr(args,"mode") else False),
            batch_size=args.batch_size,warmup_steps=getattr(args,"warmup_steps",getattr(args,"warmup_iters",0)),
            measured_steps=getattr(args,"steps",getattr(args,"timed_iters",0)))
        device = f"cuda:{local_rank()}"

        model = make_model(args.hidden_size, args.depth, device)
        ddp_model = make_ddp(model, args)
        optimizer = SGD(ddp_model.parameters(), lr=args.lr)
        x = ortho.full([args.batch_size, args.hidden_size], float(rank + 1)).to(device)
        target_neg = ortho.full([args.batch_size, args.hidden_size], -0.5).to(device)

        for _ in range(args.warmup_iters):
            run_step(ddp_model, optimizer, x, target_neg)
        sync_cuda()

        step_times = []
        backward_times = []
        for _ in range(args.timed_iters):
            sync_cuda()
            with measured("benchmark_distributed_ddp:step_start:main", lambda: sync_cuda(), calls=1, device=device, implementation="orthon", dtype="float32", config=vars(args)) as _region_step_start:
                backward_elapsed = run_step(ddp_model, optimizer, x, target_neg)
                sync_cuda()
            step_elapsed = _region_step_start["seconds"]
            step_times.append(step_elapsed)
            backward_times.append(backward_elapsed)

        median_continuous_segment_ms = 1000.0 * quartiles(step_times)[1]
        median_continuous_backward_ms = 1000.0 * quartiles(backward_times)[1]
        if rank == 0:
            print(
                "benchmark_distributed_ddp "
                f"backend=nccl "
                f"world_size={world_size} "
                f"mode={args.mode} "
                f"bucket_size_mb={args.bucket_size_mb:g} "
                f"warmup_iters={args.warmup_iters} "
                f"timed_iters={args.timed_iters} "
                f"batch_size={args.batch_size} "
                f"hidden_size={args.hidden_size} "
                f"depth={args.depth} "
                f"median_continuous_segment_ms={median_continuous_segment_ms:.3f} "
                f"median_continuous_backward_ms={median_continuous_backward_ms:.3f}",
                flush=True,
            )
    finally:
        ddist.destroy_process_group()


if __name__ == "__main__":
    run_measurement(main, "benchmark_distributed_ddp")
