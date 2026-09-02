"""Train a small MNIST MLP with Orthon.

Run from the repository root:
    python examples/mnist_mlp.py --epochs 1
"""

import argparse
import json
import random
import time
from pathlib import Path

import orthon
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam, SGD
from orthon.training import eval_step, train_step, weighted_metric_average

from utils.mnist_data import build_mnist_loaders, flatten_images, labels_to_list


def orthon_tensor_from_batch(flat_images):
    return orthon.from_numpy(flat_images.contiguous().numpy())


def make_model(hidden):
    return Sequential(
        Linear(784, hidden),
        ReLU(),
        Linear(hidden, 10),
    )


def make_optimizer(name, params, lr):
    if name == "sgd":
        return SGD(params, lr=lr)
    if name == "adam":
        return Adam(params, lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def run_epoch(model, loader, optimizer, max_batches=None):
    steps = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        targets = labels_to_list(labels)
        x = orthon_tensor_from_batch(flatten_images(images))
        steps.append(train_step(model, x, targets, orthon.cross_entropy, optimizer))

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
    }


def evaluate(model, loader, max_batches=None):
    steps = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        targets = labels_to_list(labels)
        x = orthon_tensor_from_batch(flatten_images(images))
        steps.append(eval_step(model, x, targets, orthon.cross_entropy))

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MNIST MLP with Orthon.")
    parser.add_argument("--data-dir", default="data", help="MNIST data directory.")
    parser.add_argument("--output", default="examples/results/mnist_mlp.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=100,
        help="Limit training batches per epoch; use 0 for the full train set.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=50,
        help="Limit evaluation batches per epoch; use 0 for the full test set.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST.")
    return parser.parse_args()


def limit_from_arg(value):
    return None if value == 0 else value


def main():
    args = parse_args()
    # random.seed is kept because --seed is this script's documented
    # Python-side seed, but Orthon initialization is now governed by
    # manual_seed: the framework RNG is separate from Python's.
    random.seed(args.seed)
    orthon.manual_seed(args.seed)

    train_loader, test_loader = build_mnist_loaders(
        args.data_dir,
        args.batch_size,
        download=not args.no_download,
    )

    model = make_model(args.hidden)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)

    metrics = {
        "framework": "orthon",
        "config": vars(args),
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train = run_epoch(model, train_loader, optimizer, max_train_batches)
        test = evaluate(model, test_loader, max_test_batches)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "train_samples": train["samples"],
            "test_samples": test["samples"],
            "seconds": elapsed,
        }
        metrics["epochs"].append(row)
        print(
            f"epoch {epoch}: "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
            f"test loss={row['test_loss']:.4f} acc={row['test_accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics to {output}")


if __name__ == "__main__":
    main()
