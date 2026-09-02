"""Tiny CUDA fp16 AMP training example.

Smoke run:
    python examples/cuda_amp_mlp.py --steps 5 --device cuda
"""

import argparse

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam


def make_model(hidden, device):
    model = Sequential(Linear(8, hidden), ReLU(), Linear(hidden, 2))
    for param in model.parameters():
        param.fill_(0.03)
    model.to(device).half()
    return model


def make_batch(batch_size, device):
    x = ortho.full([batch_size, 8], 0.2).to(device)
    return ortho.cast(x, "float16")


def loss_fn(model, x):
    out = model(x)
    return ortho.sum(out * out)


def parse_args():
    parser = argparse.ArgumentParser(description="Orthon CUDA AMP MLP smoke.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--init-scale", type=float, default=128.0)
    parser.add_argument("--growth-interval", type=int, default=2)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not ortho.cuda_enabled():
        raise RuntimeError("Orthon CUDA is not available in this build")

    model = make_model(args.hidden, args.device)
    optimizer = MixedPrecisionOptimizer(model, Adam, lr=args.lr)
    scaler = GradScaler(
        init_scale=args.init_scale,
        growth_interval=args.growth_interval,
    )
    x = make_batch(args.batch_size, args.device)

    losses = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        loss = loss_fn(model, x)
        loss_value = float(ortho.cast(loss.to("cpu"), "float32").item())
        if not ortho.isfinite_all(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ortho.cuda_synchronize()
        losses.append(loss_value)
        print(
            f"step {step}: loss={loss_value:.6f} "
            f"scale={scaler.scale_factor:.1f}"
        )

    if losses[-1] > losses[0] * 10.0:
        raise RuntimeError(
            f"loss exploded: first={losses[0]:.6f} last={losses[-1]:.6f}"
        )
    print(
        "amp smoke complete: "
        f"first_loss={losses[0]:.6f} last_loss={losses[-1]:.6f} "
        f"final_scale={scaler.scale_factor:.1f}"
    )


if __name__ == "__main__":
    main()
