import argparse
import os
import time

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
    args = parser.parse_args()
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
    if ortho.cuda_enabled():
        ortho.cuda_synchronize()


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
    backward_start = time.perf_counter()
    loss.backward()
    sync_cuda()
    backward_elapsed = time.perf_counter() - backward_start
    optimizer.step()
    optimizer.zero_grad()
    return backward_elapsed


def main():
    args = parse_args()
    if not ortho.cuda_enabled() or not ortho.nccl_enabled():
        raise RuntimeError("benchmark_distributed_ddp requires a CUDA+NCCL Orthon build")

    ddist.init_process_group()
    try:
        rank = ddist.get_rank()
        world_size = ddist.get_world_size()
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
            step_start = time.perf_counter()
            backward_elapsed = run_step(ddp_model, optimizer, x, target_neg)
            sync_cuda()
            step_elapsed = time.perf_counter() - step_start
            step_times.append(step_elapsed)
            backward_times.append(backward_elapsed)

        avg_step_ms = 1000.0 * sum(step_times) / len(step_times)
        avg_backward_ms = 1000.0 * sum(backward_times) / len(backward_times)
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
                f"avg_step_ms={avg_step_ms:.3f} "
                f"avg_backward_ms={avg_backward_ms:.3f}",
                flush=True,
            )
    finally:
        ddist.destroy_process_group()


if __name__ == "__main__":
    main()
