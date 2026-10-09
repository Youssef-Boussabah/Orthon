"""Train a CIFAR-10 CNN; use --synthetic-data for a bounded dataset-free run."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import orthon
from orthon.nn import Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential
from orthon.optim import Adam, SGD
from orthon.training import eval_step, train_step, weighted_metric_average
from utils.cifar10_data import (
    build_cifar10_loaders, build_synthetic_cifar10_loaders, labels_to_list,
)


class CIFAR10CNN(Module):
    def __init__(self, hidden, channels=16, init_scale=0.5):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, channels, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(channels, channels, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        self.classifier = Sequential(
            Linear(channels * 8 * 8, hidden),
            ReLU(),
            Linear(hidden, 10),
        )
        self._flat_features = channels * 8 * 8
        self._scale_parameters(init_scale)

    def forward(self, x):
        features = self.features(x)
        flat = features.reshape([x.shape[0], self._flat_features])
        return self.classifier(flat)

    def _scale_parameters(self, scale):
        if scale == 1.0:
            return
        for param in self.parameters():
            param.mul_(scale)


class CIFAR10MiniVGG(Module):
    def __init__(self, hidden, base_channels=32, init_scale=0.5):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.features = Sequential(
            Conv2d(3, c1, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(c1, c1, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(c1, c2, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(c2, c2, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(c2, c3, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(c3, c3, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        self._flat_features = c3 * 4 * 4
        self.classifier = Sequential(
            Linear(self._flat_features, hidden),
            ReLU(),
            Linear(hidden, 10),
        )
        self._scale_parameters(init_scale)

    def forward(self, x):
        features = self.features(x)
        flat = features.reshape([x.shape[0], self._flat_features])
        return self.classifier(flat)

    def _scale_parameters(self, scale):
        if scale == 1.0:
            return
        for param in self.parameters():
            param.mul_(scale)


def make_model(args):
    if args.model == "small":
        return CIFAR10CNN(args.hidden, channels=args.channels, init_scale=args.init_scale)
    if args.model == "minivgg":
        return CIFAR10MiniVGG(
            args.hidden,
            base_channels=args.vgg_base_channels,
            init_scale=args.init_scale,
        )
    raise ValueError(f"unsupported model: {args.model}")


def orthon_tensor_from_images(images, device="cpu"):
    tensor = orthon.from_numpy(images.contiguous().numpy())
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def make_optimizer(
    name,
    params,
    lr,
    adam_eps=1e-8,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.0,
    decay_bias=False,
):
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, decay_bias=decay_bias)
    if name == "adam":
        return Adam(
            params,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
            decay_bias=decay_bias,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def parse_lr_step_epochs(value):
    if value.strip() == "":
        return set()
    epochs = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        epoch = int(part)
        if epoch <= 0:
            raise ValueError("--lr-step-epochs values must be positive")
        epochs.add(epoch)
    return epochs


def apply_lr_schedule(optimizer, args, epoch, step_epochs):
    if args.lr_schedule == "none":
        return optimizer.get_lr()
    if args.lr_schedule == "step":
        if epoch in step_epochs:
            optimizer.set_lr(optimizer.get_lr() * args.lr_gamma)
            print(f"lr schedule: epoch {epoch} lr={optimizer.get_lr():.6g}")
        return optimizer.get_lr()
    if args.lr_schedule == "cosine":
        progress = (epoch - 1) / max(1, args.epochs - 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        optimizer.set_lr(lr)
        return lr
    raise ValueError(f"unsupported lr schedule: {args.lr_schedule}")


def warn_if_unstable_loss(loss, context):
    if not math.isfinite(loss):
        raise RuntimeError(f"{context} produced non-finite loss: {loss}")


def run_epoch(model, loader, optimizer, max_batches=None, device="cpu"):
    steps = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        targets = labels_to_list(labels)
        x = orthon_tensor_from_images(images, device)
        steps.append(train_step(model, x, targets, orthon.cross_entropy, optimizer))

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
    }


def evaluate(model, loader, max_batches=None, device="cpu"):
    steps = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        targets = labels_to_list(labels)
        x = orthon_tensor_from_images(images, device)
        with orthon.no_grad():
            step = eval_step(model, x, targets, orthon.cross_entropy)
        warn_if_unstable_loss(step["loss"], f"evaluation batch {batch_idx}")
        steps.append(step)

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
        "batches": len(steps),
    }


def limit_from_arg(value):
    return None if value == 0 else value


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CIFAR-10 CNN with Orthon.')
    parser.add_argument('--data-dir', default='data', help='CIFAR-10 data directory.')
    parser.add_argument('--output', default='examples/results/cifar10_cnn.json')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model', choices=['small', 'minivgg'], default='small')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--vgg-base-channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--adam-eps', type=float, default=1e-08)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--decay-bias', action='store_true')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument(
        '--init-scale', type=float, default=0.5,
        help='Scale initial weights in this CIFAR example only.',
    )
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--lr-schedule', choices=['none', 'step', 'cosine'], default='none')
    parser.add_argument('--lr-step-epochs', default='')
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--min-lr', type=float, default=0.0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--disable-shuffle', action='store_true')
    parser.add_argument(
        '--max-train-batches', type=int, default=5,
        help='Limit training batches per epoch; use 0 for the full train set.',
    )
    parser.add_argument(
        '--max-test-batches', type=int, default=5,
        help='Limit evaluation batches per epoch; use 0 for the full test set.',
    )
    parser.add_argument('--no-download', action='store_true', help='Do not download CIFAR-10.')
    parser.add_argument(
        '--synthetic-data', action='store_true',
        help='Use tiny generated CIFAR-shaped batches for smoke tests; no dataset files needed.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    orthon.manual_seed(args.seed)
    if args.device == "cuda" and not orthon.cuda_enabled():
        print("CUDA device requested, but this Orthon build has CUDA disabled.")
        return
    step_epochs = parse_lr_step_epochs(args.lr_step_epochs)
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)
    if args.synthetic_data:
        train_loader, test_loader = build_synthetic_cifar10_loaders(
            args.batch_size,
            train_batches=max_train_batches or 2,
            test_batches=max_test_batches or 1,
            seed=args.seed,
        )
    else:
        train_loader, test_loader = build_cifar10_loaders(
            args.data_dir, args.batch_size, download=not args.no_download,
            seed=args.seed, shuffle_train=not args.disable_shuffle,
            train_augment=args.augment,
        )
    model = make_model(args)
    model.to(args.device)
    optimizer = make_optimizer(
        args.optimizer, model.parameters(), args.lr,
        adam_eps=args.adam_eps, adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2, weight_decay=args.weight_decay,
        decay_bias=args.decay_bias,
    )
    metrics = {
        "framework": "orthon",
        "task": "cifar10_cnn",
        "config": vars(args),
        "epochs": [],
    }
    for epoch in range(1, args.epochs + 1):
        lr = apply_lr_schedule(optimizer, args, epoch, step_epochs)
        start = time.monotonic()
        train = run_epoch(model, train_loader, optimizer, max_train_batches, args.device)
        test = evaluate(model, test_loader, max_test_batches, args.device)
        row = {
            "epoch": epoch,
            "lr": lr,
            "seconds": time.monotonic() - start,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "eval_loss": test["loss"],
            "eval_accuracy": test["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "train_samples": train["samples"],
            "eval_samples": test["samples"],
            "test_samples": test["samples"],
        }
        metrics["epochs"].append(row)
        print(f"epoch {epoch}: lr={lr:.6g} | "
              f"train loss={train['loss']:.4f} acc={train['accuracy']:.4f} | "
              f"test loss={test['loss']:.4f} acc={test['accuracy']:.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics to {output}")


if __name__ == "__main__":
    main()
