"""Train the matching CIFAR-10 CNN baseline with PyTorch."""

import argparse
import json
import time
from pathlib import Path

from utils.cifar10_data import build_cifar10_loaders, require_torchvision_cifar10


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 CNN with PyTorch.")
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 data directory.")
    parser.add_argument("--output", default="examples/results/cifar10_cnn_torch.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.5,
        help="Scale initial weights to match the Orthon CIFAR smoke setup.",
    )
    parser.add_argument("--debug-logits", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=5,
        help="Limit training batches per epoch; use 0 for the full train set.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=5,
        help="Limit evaluation batches per epoch; use 0 for the full test set.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not download CIFAR-10.")
    return parser.parse_args()


def make_model(torch, nn, hidden, channels=16, init_scale=0.5):
    model = nn.Sequential(
        nn.Conv2d(3, channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(channels * 8 * 8, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )
    if init_scale != 1.0:
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(init_scale)
    return model


def make_optimizer(torch, name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def print_first_batch_logit_stats(torch, model, loader):
    images, _ = next(iter(loader))
    logits = model(images.float())
    print(
        "first-batch logits: "
        f"min={float(logits.min().item()):.4f} "
        f"max={float(logits.max().item()):.4f} "
        f"mean={float(logits.mean().item()):.4f} "
        f"std={float(logits.std(unbiased=False).item()):.4f}"
    )


def run_epoch(torch, model, loader, optimizer, max_batches=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        optimizer.zero_grad()
        logits = model(images.float())
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

            logits = model(images.float())
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
    torch, _, _, _ = require_torchvision_cifar10()
    import torch.nn as nn

    torch.manual_seed(args.seed)
    train_loader, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        download=not args.no_download,
        seed=args.seed,
    )

    model = make_model(torch, nn, args.hidden, channels=args.channels, init_scale=args.init_scale)
    optimizer = make_optimizer(torch, args.optimizer, model.parameters(), args.lr)
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)

    metrics = {
        "framework": "pytorch",
        "task": "cifar10_cnn",
        "config": vars(args),
        "epochs": [],
    }

    if args.debug_logits:
        print_first_batch_logit_stats(torch, model, train_loader)

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
