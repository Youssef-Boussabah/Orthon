"""Tiny synthetic MLP training example with checkpoint resume.

Smoke run:
    python examples/synthetic_mlp_checkpoint.py --steps 5 --resume-smoke
"""

import argparse
from pathlib import Path

import orthon as ortho
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam


def make_model(hidden):
    model = Sequential(Linear(8, hidden), ReLU(), Linear(hidden, 2))
    for param in model.parameters():
        param.fill_(0.03)
    return model


def make_batch(batch_size):
    return ortho.full([batch_size, 8], 0.2)


def loss_fn(model, x):
    out = model(x)
    return ortho.sum(out * out)


def train_steps(model, optimizer, x, steps):
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model, x)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
    return losses


def flat_params(model):
    values = []
    for param in model.parameters():
        values.extend(ortho.cast(param.to("cpu"), "float32").tolist())
    return [float(value) for value in values]


def max_abs_diff(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Orthon MLP checkpoint smoke.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--resume-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint",
        default="examples/results/synthetic_mlp_checkpoint.ckpt",
    )
    parser.add_argument(
        "--resume-smoke",
        action="store_true",
        help="Validate that continuous and resumed training end at the same parameters.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the checkpoint file after --resume-smoke.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    x = make_batch(args.batch_size)

    model = make_model(args.hidden)
    optimizer = Adam(model.parameters(), lr=args.lr)
    initial_loss = float(loss_fn(model, x).item())
    first_losses = train_steps(model, optimizer, x, args.steps)
    ortho.save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        metadata={"steps": args.steps, "loss": first_losses[-1]},
    )
    print(
        f"saved checkpoint: {checkpoint} "
        f"initial_loss={initial_loss:.6f} step_loss={first_losses[-1]:.6f}"
    )

    if not args.resume_smoke:
        return

    continuous = model
    continuous_optimizer = optimizer
    continuous_losses = train_steps(
        continuous,
        continuous_optimizer,
        x,
        args.resume_steps,
    )

    resumed = make_model(args.hidden)
    resumed_optimizer = Adam(resumed.parameters(), lr=args.lr)
    metadata = ortho.load_checkpoint(checkpoint, model=resumed, optimizer=resumed_optimizer)
    resumed_losses = train_steps(resumed, resumed_optimizer, x, args.resume_steps)
    diff = max_abs_diff(flat_params(continuous), flat_params(resumed))
    print(
        "resume smoke: "
        f"metadata={metadata} "
        f"continuous_loss={continuous_losses[-1]:.6f} "
        f"resumed_loss={resumed_losses[-1]:.6f} "
        f"max_param_diff={diff:.6e}"
    )
    if diff > 1e-5:
        raise RuntimeError(f"resume mismatch: max_param_diff={diff:.6e}")
    if not args.keep_checkpoint:
        checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
