"""Train the matching MNIST MLP baseline with PyTorch."""

import argparse
import json
import time
from pathlib import Path

from utils.mnist_data import (
    build_mnist_loaders,
    flatten_images,
    require_torchvision_mnist,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MNIST MLP with PyTorch.")
    parser.add_argument("--data-dir", default="data", help="MNIST data directory.")
    parser.add_argument("--output", default="examples/results/mnist_mlp_torch.json")
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


def make_model(torch, nn, hidden):
    return nn.Sequential(
        nn.Linear(784, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )


def make_optimizer(torch, name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def run_epoch(torch, model, loader, optimizer, max_batches=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = flatten_images(images)
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "samples": total_seen,
    }


def evaluate(torch, model, loader, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = flatten_images(images)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += batch_size

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "samples": total_seen,
    }


def limit_from_arg(value):
    return None if value == 0 else value


def main():
    args = parse_args()
    torch, _, _, _ = require_torchvision_mnist()
    import torch.nn as nn

    torch.manual_seed(args.seed)
    train_loader, test_loader = build_mnist_loaders(
        args.data_dir,
        args.batch_size,
        download=not args.no_download,
    )

    model = make_model(torch, nn, args.hidden)
    optimizer = make_optimizer(torch, args.optimizer, model.parameters(), args.lr)
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)

    metrics = {
        "framework": "pytorch",
        "config": vars(args),
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train = run_epoch(torch, model, train_loader, optimizer, max_train_batches)
        test = evaluate(torch, model, test_loader, max_test_batches)
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
