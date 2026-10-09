"""Measure Orthon local checkpoint save/load overhead on a synthetic MLP."""

from benchmarks._support.training import mark_correctness, measured, configure, options, run as run_measurement

import argparse
import tempfile
from pathlib import Path

import orthon as ortho
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam


def make_model(input_dim, hidden, output_dim):
    model = Sequential(
        Linear(input_dim, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, output_dim),
    )
    for index, param in enumerate(model.parameters()):
        param.fill_(0.01 * (index + 1))
    return model


def parameter_count(model):
    return sum(_numel(param) for param in model.parameters())


def _numel(tensor):
    total = 1
    for dim in tensor.shape:
        total *= int(dim)
    return total


def make_batch(batch_size, input_dim):
    return ortho.full([batch_size, input_dim], 0.125)


def loss_fn(model, x):
    out = model(x)
    return ortho.mean(out * out)


def train_steps(model, optimizer, x, steps):
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model, x)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
    return losses


def max_param_diff(left, right):
    diffs = []
    for lp, rp in zip(left.parameters(), right.parameters()):
        lv = ortho.cast(lp.to("cpu"), "float32").tolist()
        rv = ortho.cast(rp.to("cpu"), "float32").tolist()
        diffs.extend(abs(float(a) - float(b)) for a, b in zip(_flatten(lv), _flatten(rv)))
    return max(diffs) if diffs else 0.0


def _flatten(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    out = []
    for item in value:
        out.extend(_flatten(item))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Orthon checkpoint overhead benchmark.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--resume-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--keep-checkpoint", action="store_true")
    options(parser)
    args = configure(parser.parse_args())
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.resume_steps < 1:
        parser.error("--resume-steps must be at least 1")
    return args


def main():
    args = parse_args()
    x = make_batch(args.batch_size, args.input_dim)

    model = make_model(args.input_dim, args.hidden, args.output_dim)
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_steps(model, optimizer, x, args.steps)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    temp_dir = None
    if checkpoint_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        checkpoint_path = Path(temp_dir.name) / "checkpoint.ckpt"
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with measured("bench_checkpoint_overhead:save_start:main", lambda: None, calls=1, device="cpu", implementation="orthon", dtype="float32", config=vars(args)) as _region_save_start:
        ortho.save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            metadata={"steps": args.steps, "benchmark": "checkpoint_overhead"},
        )
    save_ms = (_region_save_start["seconds"]) * 1000.0
    checkpoint_size_kb = checkpoint_path.stat().st_size / 1024.0

    restored = make_model(args.input_dim, args.hidden, args.output_dim)
    restored_optimizer = Adam(restored.parameters(), lr=args.lr)
    with measured("bench_checkpoint_overhead:load_start:main", lambda: None, calls=1, device="cpu", implementation="orthon", dtype="float32", config=vars(args)) as _region_load_start:
        metadata = ortho.load_checkpoint(
            checkpoint_path,
            model=restored,
            optimizer=restored_optimizer,
        )
    load_ms = (_region_load_start["seconds"]) * 1000.0

    continuous_losses = train_steps(model, optimizer, x, args.resume_steps)
    resumed_losses = train_steps(restored, restored_optimizer, x, args.resume_steps)
    max_loss_delta = abs(continuous_losses[-1] - resumed_losses[-1])
    max_parameter_delta = max_param_diff(model, restored)
    resume_match = max_loss_delta <= 1e-6 and max_parameter_delta <= 1e-5
    mark_correctness("pass" if resume_match else "fail", max_loss_delta=max_loss_delta,max_parameter_delta=max_parameter_delta)
    if not resume_match:
        raise RuntimeError("checkpoint resume correctness failed")
    params = parameter_count(model)

    print("benchmark_name: checkpoint_overhead")
    print(f"model: mlp_{args.input_dim}_{args.hidden}_{args.output_dim}")
    print(f"parameter_count: {params}")
    print(f"checkpoint_size_kb: {checkpoint_size_kb:.2f}")
    print(f"save_ms: {save_ms:.3f}")
    print(f"load_ms: {load_ms:.3f}")
    print(f"resume_match: {resume_match}")
    print(f"max_loss_delta: {max_loss_delta:.6e}")
    print(f"max_parameter_delta: {max_parameter_delta:.6e}")
    print(f"metadata_steps: {metadata.get('steps')}")
    print("notes: local ml-runtime-checkpoint archive (ZIP + JSON manifest + raw "
          "tensor bytes, no pickle); not sharded, distributed, or cloud-native")
    print(
        "csv_summary,"
        "benchmark_name=checkpoint_overhead,"
        f"model=mlp_{args.input_dim}_{args.hidden}_{args.output_dim},"
        f"parameter_count={params},"
        f"checkpoint_size_kb={checkpoint_size_kb:.2f},"
        f"save_ms={save_ms:.3f},"
        f"load_ms={load_ms:.3f},"
        f"resume_match={resume_match},"
        f"max_loss_delta={max_loss_delta:.6e},"
        f"max_parameter_delta={max_parameter_delta:.6e}"
    )

    if not args.keep_checkpoint:
        checkpoint_path.unlink(missing_ok=True)
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    run_measurement(main, "bench_checkpoint_overhead")
