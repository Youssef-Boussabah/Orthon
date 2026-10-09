"""Tiny synthetic MLP training comparison.

This is a smoke-friendly benchmark. It reports rough step time and loss
trajectory only; it is not a performance claim.
"""

from benchmarks._support.training import measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer, cuda_available

import argparse

import orthon as ortho
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam


def sync_device(device):
    synchronizer(device, "orthon")()


def make_orthon_model(hidden, device):
    model = Sequential(Linear(8, hidden), ReLU(), Linear(hidden, 2))
    for param in model.parameters():
        param.fill_(0.03)
    model.to(device)
    return model


def orthon_step(model, optimizer, x):
    optimizer.zero_grad()
    out = model(x)
    loss = ortho.sum(out * out)
    value = float(loss.item())
    loss.backward()
    optimizer.step()
    return value


def run_orthon(args):
    if args.device == "cuda" and not cuda_available("cuda:0"):
        return None
    model = make_orthon_model(args.hidden, args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    x = ortho.full([args.batch_size, 8], 0.2)
    if args.device == "cuda":
        x = x.to("cuda")

    losses = []
    for _ in range(args.warmup):
        orthon_step(model, optimizer, x)
    sync_device(args.device)
    with measured("bench_tiny_mlp_training:start:run_orthon", lambda: sync_device(args.device), calls=args.steps, device=args.device, implementation="orthon", dtype="float32", config=vars(args)) as _region_start:
        for _ in range(args.steps):
            losses.append(orthon_step(model, optimizer, x))
        sync_device(args.device)
    elapsed = _region_start["seconds"]
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "ms_per_step": elapsed * 1000.0 / args.steps,
    }


def run_torch(args):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    if args.device == "cuda" and not torch.cuda.is_available():
        return None

    device = torch.device(args.device)
    model = nn.Sequential(nn.Linear(8, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 2)).to(device)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.03)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x = torch.full((args.batch_size, 8), 0.2, device=device)

    def step():
        optimizer.zero_grad()
        out = model(x)
        loss = torch.sum(out * out)
        value = float(loss.item())
        loss.backward()
        optimizer.step()
        return value

    losses = []
    for _ in range(args.warmup):
        step()
    if args.device == "cuda":
        synchronizer(args.device,"torch")()
    with measured("bench_tiny_mlp_training:torch-start:run_torch", synchronizer(args.device, 'torch'), calls=args.steps, device=args.device, implementation="torch", dtype="float32", config=vars(args)) as _region_start:
        for _ in range(args.steps):
            losses.append(step())
        if args.device == "cuda":
            synchronizer(args.device,"torch")()
    elapsed = _region_start["seconds"]
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "ms_per_step": elapsed * 1000.0 / args.steps,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny Orthon/PyTorch MLP training comparison.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    options(parser)
    return configure(parser.parse_args())


def print_result(name, result):
    if result is None:
        print(f"{name}: skipped")
        return
    print(
        f"{name}: first_loss={result['first_loss']:.6f} "
        f"last_loss={result['last_loss']:.6f} "
        f"ms_per_step={result['ms_per_step']:.3f}"
    )


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    print(
        "tiny MLP training comparison "
        f"(device={args.device}, steps={args.steps}, batch={args.batch_size})"
    )
    da_result = run_orthon(args)
    torch_result = run_torch(args)
    print_result("Orthon", da_result)
    print_result("PyTorch", torch_result)


if __name__ == "__main__":
    run_measurement(main, "bench_tiny_mlp_training")
