"""Training benchmarks for Orthon and PyTorch.

Runs two workloads:
  - MNIST MLP
  - CIFAR-10 CNN

Presets separate correctness smoke checks from overhead and throughput runs.
"""

from benchmarks._support.training import measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer

import argparse
import random

from examples.utils.cifar10_data import build_cifar10_loaders, labels_to_list as cifar_labels
from examples.utils.mnist_data import build_mnist_loaders, flatten_images, labels_to_list as mnist_labels


PRESETS = {
    "smoke": {
        "epochs": 1,
        "max_train_batches": 2,
        "max_test_batches": 1,
        "mnist_batch_size": 32,
        "cifar_batch_size": 4,
        "hidden": 32,
        "channels": 4,
        "warmup": 1,
    },
    "small": {
        "epochs": 2,
        "max_train_batches": 10,
        "max_test_batches": 3,
        "mnist_batch_size": 64,
        "cifar_batch_size": 8,
        "hidden": 64,
        "channels": 4,
        "warmup": 1,
    },
    "medium": {
        "epochs": 3,
        "max_train_batches": 50,
        "max_test_batches": 10,
        "mnist_batch_size": 128,
        "cifar_batch_size": 16,
        "hidden": 128,
        "channels": 8,
        "warmup": 2,
    },
}


def limit_from_arg(value):
    return None if value == 0 else value


def sync_device(framework, device):
    synchronizer(device, framework)()


def weighted_average(rows):
    total = sum(row["samples"] for row in rows)
    if total == 0:
        raise RuntimeError("benchmark produced no samples")
    return {
        "loss": sum(row["loss"] * row["samples"] for row in rows) / total,
        "accuracy": sum(row["accuracy"] * row["samples"] for row in rows) / total,
        "samples": total,
    }


def orthon_tensor_from_torch(tensor, device):
    import orthon

    out = orthon.from_numpy(tensor.contiguous().numpy())
    return out.to(device) if device == "cuda" else out


def orthon_accuracy(logits, targets):
    values = logits.tolist()
    classes = int(logits.shape[1])
    correct = 0
    for row, target in enumerate(targets):
        offset = row * classes
        pred = max(range(classes), key=lambda i: values[offset + i])
        correct += int(pred == target)
    return correct / len(targets)


def run_orthon_epoch(model, loader, workload, optimizer, max_batches, device, training):
    import orthon

    rows = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if workload == "mnist_mlp":
            x = orthon_tensor_from_torch(flatten_images(images), device)
            targets = mnist_labels(labels)
        else:
            x = orthon_tensor_from_torch(images.float(), device)
            targets = cifar_labels(labels)

        if training:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        logits = model(x)
        loss = orthon.cross_entropy(logits, targets)
        if training:
            loss.backward()
            optimizer.step()

        rows.append({
            "loss": float(loss.item()),
            "accuracy": orthon_accuracy(logits, targets),
            "samples": len(targets),
        })

    sync_device("orthon", device)
    return weighted_average(rows)


def make_orthon_mnist(hidden, device):
    from orthon.nn import Linear, ReLU, Sequential
    from orthon.optim import SGD

    model = Sequential(Linear(784, hidden), ReLU(), Linear(hidden, 10))
    model.to(device)
    return model, SGD(model.parameters(), lr=0.01)


def make_orthon_cifar(hidden, channels, device):
    import orthon
    from orthon.nn import Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential
    from orthon.optim import SGD

    class SmallCIFAR(Module):
        def __init__(self):
            super().__init__()
            self.features = Sequential(
                Conv2d(3, channels, kernel_size=3, padding=1),
                ReLU(),
                MaxPool2d(kernel_size=2),
            )
            self.classifier = Linear(channels * 16 * 16, 10)

        def forward(self, x):
            out = self.features(x)
            out = orthon.reshape(out, [x.shape[0], channels * 16 * 16])
            return self.classifier(out)

    model = SmallCIFAR()
    model.to(device)
    return model, SGD(model.parameters(), lr=0.01)


def torch_accuracy(torch, logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def run_torch_epoch(torch, model, loader, workload, optimizer, max_batches, device, training):
    rows = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        labels = labels.to(device)
        x = flatten_images(images).to(device) if workload == "mnist_mlp" else images.float().to(device)

        if training:
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, labels)

        rows.append({
            "loss": float(loss.item()),
            "accuracy": torch_accuracy(torch, logits, labels),
            "samples": int(labels.shape[0]),
        })

    sync_device("torch", device)
    return weighted_average(rows)


def make_torch_mnist(torch, nn, hidden, device):
    model = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10)).to(device)
    return model, torch.optim.SGD(model.parameters(), lr=0.01)


def make_torch_cifar(torch, nn, hidden, channels, device):
    model = nn.Sequential(
        nn.Conv2d(3, channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(channels * 16 * 16, 10),
    ).to(device)
    return model, torch.optim.SGD(model.parameters(), lr=0.01)


def run_workload(args, workload):
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)

    if workload == "mnist_mlp":
        train_loader, test_loader = build_mnist_loaders(
            args.data_dir, args.mnist_batch_size, download=not args.no_download)
    else:
        train_loader, test_loader = build_cifar10_loaders(
            args.data_dir, args.cifar_batch_size, download=not args.no_download, seed=args.seed)

    if args.framework == "orthon":
        import orthon

        if args.device == "cuda" and not orthon.cuda_enabled():
            raise RuntimeError("Orthon CUDA is unavailable in this build")
        if workload == "mnist_mlp":
            model, optimizer = make_orthon_mnist(args.hidden, args.device)
        else:
            model, optimizer = make_orthon_cifar(args.hidden, args.channels, args.device)
        run_epoch = run_orthon_epoch
    else:
        import torch
        import torch.nn as nn

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.manual_seed(args.seed)
        if workload == "mnist_mlp":
            model, optimizer = make_torch_mnist(torch, nn, args.hidden, args.device)
        else:
            model, optimizer = make_torch_cifar(torch, nn, args.hidden, args.channels, args.device)

        def run_epoch(model, loader, workload, optimizer, max_batches, device, training):
            return run_torch_epoch(torch, model, loader, workload, optimizer, max_batches, device, training)

    initial = run_epoch(model, train_loader, workload, optimizer, max_train_batches, args.device, False)
    for warmup_idx in range(args.warmup):
        run_epoch(model, train_loader, workload, optimizer, max_train_batches, args.device, True)
        sync_device(args.framework, args.device)
        print(f"{workload} warmup {warmup_idx + 1}/{args.warmup} complete")

    epochs = []
    for epoch in range(1, args.epochs + 1):
        sync_device(args.framework, args.device)
        with measured("benchmark_training:start:run_workload", lambda: sync_device(args.framework, args.device), calls=1, device=args.device, implementation=args.framework, dtype="float32", config=vars(args)) as _region_start:
            train = run_epoch(model, train_loader, workload, optimizer, max_train_batches, args.device, True)
            test = run_epoch(model, test_loader, workload, optimizer, max_test_batches, args.device, False)
            _region_start.update(samples_processed=train['samples'], units='ms/epoch',
                notes='one continuous train+evaluation epoch; samples/sec denominator includes evaluation',
                metrics=dict(workload=workload,train_loss=train['loss'],train_accuracy=train['accuracy'],
                             test_loss=test['loss'],test_accuracy=test['accuracy'],
                             train_samples=train['samples'],test_samples=test['samples']))
            sync_device(args.framework, args.device)
        seconds = _region_start["seconds"]
        samples_per_sec = train["samples"] / seconds if seconds > 0 else 0.0
        row = {
            "epoch": epoch,
            "epoch_time_sec": seconds,
            "samples_per_sec": samples_per_sec,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "train_samples": train["samples"],
            "test_samples": test["samples"],
        }
        epochs.append(row)
        print(
            f"{workload} epoch {epoch}: "
            f"loss={row['train_loss']:.4f} acc={row['test_accuracy']:.4f} "
            f"time={seconds:.4f}s samples/sec={samples_per_sec:.2f}"
        )

    final = epochs[-1]
    return {
        "workload": workload,
        "preset": args.preset,
        "batch_size": args.mnist_batch_size if workload == "mnist_mlp" else args.cifar_batch_size,
        "epochs_requested": args.epochs,
        "max_train_batches": args.max_train_batches,
        "max_test_batches": args.max_test_batches,
        "warmup": args.warmup,
        "initial_train_loss": initial["loss"],
        "final_train_loss": final["train_loss"],
        "loss_decrease": initial["loss"] - final["train_loss"],
        "final_accuracy": final["test_accuracy"],
        "epoch_time_sec": final["epoch_time_sec"],
        "samples_per_sec": final["samples_per_sec"],
        "epochs": epochs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Orthon/PyTorch training benchmarks.")
    parser.add_argument("--framework", choices=["orthon", "torch"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--preset", choices=["smoke", "small", "medium"], default="smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--mnist-batch-size", type=int, default=None)
    parser.add_argument("--cifar-batch-size", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/orthon-benchmarks")
    options(parser)
    return configure(parser.parse_args())


def apply_preset(args):
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main():
    args = apply_preset(parse_args())
    random.seed(args.seed)
    if args.framework == "orthon":
        # Imported lazily and only for this framework, matching how the rest of
        # this script defers the import so --framework torch runs without
        # orthon installed.
        import orthon
        orthon.manual_seed(args.seed)

    # Dataset epochs remain continuous training regions, distinct from synthetic models.
    for workload in ['mnist_mlp','cifar_cnn']:
        run_workload(args,workload)


if __name__ == '__main__':
    run_measurement(main,'benchmark_training')
