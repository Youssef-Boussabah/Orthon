"""Train a small CIFAR-10 CNN with Orthon.

Run from the repository root:
    python examples/cifar10_cnn.py --epochs 1
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

import orthon
from orthon.nn import Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential
from orthon.optim import Adam, SGD
from orthon.training import (
    classification_accuracy,
    eval_step,
    train_step,
    weighted_metric_average,
)

from utils.cifar10_data import (
    build_cifar10_loaders,
    build_synthetic_cifar10_loaders,
    labels_to_list,
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


def flatten_nested(values):
    flat = []
    if isinstance(values, (list, tuple)):
        for value in values:
            flat.extend(flatten_nested(value))
    else:
        flat.append(float(values))
    return flat


def _abs_stats(values):
    if not values:
        return {"max": 0.0, "mean": 0.0}
    abs_values = [abs(float(v)) for v in values]
    return {
        "max": max(abs_values),
        "mean": sum(abs_values) / len(abs_values),
    }


def tensor_stats(tensor):
    return _abs_stats(flatten_nested(tensor.tolist()))


def diff_stats(after_values, before_values):
    if not before_values:
        return {"max": 0.0, "mean": 0.0}
    diffs = [abs(float(a) - float(b)) for a, b in zip(after_values, before_values)]
    return {
        "max": max(diffs),
        "mean": sum(diffs) / len(diffs),
    }


def _ratio(numerator, denominator):
    return float("inf") if denominator == 0.0 and numerator != 0.0 else (
        0.0 if denominator == 0.0 else numerator / denominator
    )


def snapshot_parameters(model):
    snapshots = []
    for index, param in enumerate(model.parameters()):
        values = flatten_nested(param.tolist())
        snapshots.append({
            "index": index,
            "shape": list(param.shape),
            "values": values,
            "stats": _abs_stats(values),
        })
    return snapshots


def debug_parameter_updates(model, before_values):
    print("debug all parameter updates:")
    print(
        "  idx shape              param_max  param_mean   grad_max  grad_mean "
        "update_max update_mean upd_max/param upd_mean/param"
    )
    rows = []
    for param, before in zip(model.parameters(), before_values):
        after_values = flatten_nested(param.tolist())
        grad_stats = (
            tensor_stats(param.grad)
            if param.grad is not None
            else {"max": 0.0, "mean": 0.0}
        )
        update_stats = diff_stats(after_values, before["values"])
        param_stats = before["stats"]
        row = {
            "index": before["index"],
            "shape": before["shape"],
            "param_max": param_stats["max"],
            "param_mean": param_stats["mean"],
            "grad_max": grad_stats["max"],
            "grad_mean": grad_stats["mean"],
            "update_max": update_stats["max"],
            "update_mean": update_stats["mean"],
            "update_max_param_max_ratio": _ratio(update_stats["max"], param_stats["max"]),
            "update_mean_param_mean_ratio": _ratio(update_stats["mean"], param_stats["mean"]),
        }
        rows.append(row)
        print(
            f"  {row['index']:>3} {str(row['shape']):<18} "
            f"{row['param_max']:.3e} {row['param_mean']:.3e} "
            f"{row['grad_max']:.3e} {row['grad_mean']:.3e} "
            f"{row['update_max']:.3e} {row['update_mean']:.3e} "
            f"{row['update_max_param_max_ratio']:.3e} "
            f"{row['update_mean_param_mean_ratio']:.3e}"
        )
    return rows


def grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        for value in flatten_nested(param.grad.tolist()):
            total += float(value) * float(value)
    return total ** 0.5


def _tensor_from_flat_values(values, shape, device):
    array = np.array(values, dtype=np.float32).reshape(tuple(shape))
    tensor = orthon.from_numpy(array)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def clip_gradients(parameters, clip_norm=None, clip_value=None):
    params = list(parameters)
    result = {"global_norm": grad_norm(params), "scale": 1.0}
    with orthon.no_grad():
        if clip_value is not None:
            limit = abs(float(clip_value))
            for param in params:
                if param.grad is None:
                    continue
                values = flatten_nested(param.grad.tolist())
                clipped = [max(-limit, min(limit, value)) for value in values]
                param.set_grad(_tensor_from_flat_values(clipped, list(param.grad.shape), param.grad.device))
            result["global_norm"] = grad_norm(params)

        if clip_norm is not None:
            limit = float(clip_norm)
            if limit <= 0.0:
                raise ValueError("--grad-clip-norm must be positive")
            current_norm = result["global_norm"]
            if current_norm > limit:
                scale = limit / (current_norm + 1e-12)
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(scale)
                result["scale"] = scale
                result["global_norm"] = grad_norm(params)
    return result


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


def logit_stats(logits):
    values = logits.tolist()
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": var ** 0.5,
    }


def warn_if_unstable_loss(loss, context):
    if not math.isfinite(loss):
        raise RuntimeError(f"{context} produced non-finite loss: {loss}")


def warn_if_suspicious_initial_values(loss, logits):
    warn_if_unstable_loss(loss, "initial evaluation")
    stats = logit_stats(logits)
    max_abs_logit = max(abs(stats["min"]), abs(stats["max"]))
    if loss > 10.0:
        print(
            "warning: initial CIFAR-10 loss is much larger than the expected "
            f"untrained range near 2.3: loss={loss:.4f}. "
            "This suggests an initialization or logit-scale issue."
        )
    if max_abs_logit > 20.0:
        print(
            "warning: initial logits have very large magnitude "
            f"(max_abs={max_abs_logit:.4f}). "
            "This can make cross entropy unstable."
        )


def print_first_batch_logit_stats(model, loader, device="cpu"):
    images, _ = next(iter(loader))
    x = orthon_tensor_from_images(images, device)
    stats = logit_stats(model(x))
    print(
        "first-batch logits: "
        f"min={stats['min']:.4f} max={stats['max']:.4f} "
        f"mean={stats['mean']:.4f} std={stats['std']:.4f}"
    )


def collect_fixed_batches(loader, num_batches):
    fixed_batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        fixed_batches.append(batch)
    if len(fixed_batches) != num_batches:
        raise RuntimeError(
            f"requested {num_batches} overfit batches, but loader only produced "
            f"{len(fixed_batches)}"
        )
    return fixed_batches


def num_examples(batches):
    return sum(int(labels.shape[0]) for _, labels in batches)


def batch_fingerprints(batches):
    fingerprints = []
    for images, labels in batches:
        fingerprints.append({
            "shape": list(images.shape),
            "labels": labels_to_list(labels),
            "image_sum": float(images.sum().item()),
        })
    return fingerprints


PROFILE_KEYS = [
    "data_loading",
    "label_conversion",
    "input_conversion",
    "forward",
    "loss",
    "backward",
    "optimizer",
    "metrics",
    "evaluation",
]


def maybe_synchronize(device):
    if device == "cuda" and hasattr(orthon, "cuda_synchronize"):
        orthon.cuda_synchronize()


def conv2d_profile_enabled():
    return os.environ.get("ORTHON_PROFILE_CONV2D", "0") != "0"


def reset_conv2d_profile():
    reset = getattr(orthon, "cuda_conv2d_profile_reset", None)
    if callable(reset):
        reset()


def conv2d_profile_summary():
    summary = getattr(orthon, "cuda_conv2d_profile_summary", None)
    if callable(summary):
        return summary()
    return ""


def profile_percent(profile):
    total = sum(profile.values())
    return {
        key: (value / total * 100.0) if total > 0 else 0.0
        for key, value in profile.items()
    }


def _profile_start(device, sync=False):
    if sync:
        maybe_synchronize(device)
    return time.perf_counter()


def _add_time(profile, key, start, device="cpu", sync=False):
    if sync:
        maybe_synchronize(device)
    profile[key] += time.perf_counter() - start


def _print_profile(profile, train_batches, train_samples, eval_batches=None, eval_samples=None):
    percentages = profile_percent(profile)
    train_total = sum(profile[key] for key in PROFILE_KEYS if key != "evaluation")
    seconds_per_batch = train_total / train_batches if train_batches else 0.0
    seconds_per_sample = train_total / train_samples if train_samples else 0.0
    print("profile:")
    for key in PROFILE_KEYS:
        value = profile[key]
        print(f"  {key:<17} {value:>8.4f}s  {percentages[key]:>5.1f}%")
    print(
        "profile summary: "
        f"train_batches={train_batches} train_samples={train_samples} "
        f"seconds_per_train_batch={seconds_per_batch:.6f} "
        f"seconds_per_train_sample={seconds_per_sample:.6f}"
    )
    if eval_batches is not None or eval_samples is not None:
        print(
            "profile eval: "
            f"eval_batches={eval_batches or 0} eval_samples={eval_samples or 0}"
        )


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


def run_epoch_fixed_batches(
    model,
    batches,
    optimizer,
    device="cpu",
    debug_train=False,
    debug_all_params=False,
    grad_clip_norm=None,
    grad_clip_value=None,
):
    steps = []
    for batch_idx, (images, labels) in enumerate(batches):
        targets = labels_to_list(labels)
        x = orthon_tensor_from_images(images, device)

        model.train()
        optimizer.zero_grad()
        first_param = next(model.parameters())
        first_param_before = (
            flatten_nested(first_param.tolist())
            if debug_train and batch_idx == 0
            else None
        )
        all_params_before = (
            snapshot_parameters(model)
            if debug_all_params and batch_idx == 0
            else None
        )

        logits = model(x)
        loss = orthon.cross_entropy(logits, targets)
        loss_value = float(loss.item())
        warn_if_unstable_loss(loss_value, f"training batch {batch_idx}")
        loss.backward()

        if grad_clip_norm is not None or grad_clip_value is not None:
            clip_result = clip_gradients(
                model.parameters(),
                clip_norm=grad_clip_norm,
                clip_value=grad_clip_value,
            )
            if debug_train and batch_idx == 0:
                print(
                    "debug grad clip: "
                    f"global_norm={clip_result['global_norm']:.6e} "
                    f"scale={clip_result['scale']:.6e}"
                )

        if debug_train and batch_idx == 0:
            param_stats = tensor_stats(first_param)
            print(
                "debug first parameter before update: "
                f"max={param_stats['max']:.6e} mean={param_stats['mean']:.6e}"
            )
            if first_param.grad is not None:
                grad_stats = tensor_stats(first_param.grad)
                print(
                    "debug first gradient: "
                    f"max={grad_stats['max']:.6e} mean={grad_stats['mean']:.6e}"
                )
            else:
                print("debug first gradient: None")

        optimizer.step()

        if debug_train and batch_idx == 0 and first_param_before is not None:
            update_stats = diff_stats(flatten_nested(first_param.tolist()), first_param_before)
            print(
                "debug first update: "
                f"max={update_stats['max']:.6e} mean={update_stats['mean']:.6e}"
            )
        if all_params_before is not None:
            debug_parameter_updates(model, all_params_before)

        steps.append({
            "loss": loss_value,
            "accuracy": classification_accuracy(logits, targets),
            "num_examples": len(targets),
        })

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
    }


def run_adam_sweep(args, fixed_batches):
    lrs = [1e-5, 3e-5, 1e-4, 3e-4]
    eps_values = [1e-8, 1e-7, 1e-6, 1e-5]
    results = []
    print(f"Adam sweep: {len(lrs) * len(eps_values)} configs, {args.sweep_epochs} epochs each")
    for lr in lrs:
        for eps in eps_values:
            random.seed(args.seed)
            orthon.manual_seed(args.seed)   # The framework RNG is separate from Python's global random.
            model = make_model(args)
            model.to(args.device)
            optimizer = make_optimizer(
                "adam",
                model.parameters(),
                lr,
                adam_eps=eps,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                weight_decay=args.weight_decay,
                decay_bias=args.decay_bias,
            )
            spiked = False
            train = None
            for _ in range(args.sweep_epochs):
                train = run_epoch_fixed_batches(
                    model,
                    fixed_batches,
                    optimizer,
                    args.device,
                    grad_clip_norm=args.grad_clip_norm,
                    grad_clip_value=args.grad_clip_value,
                )
                if train["loss"] > 10.0:
                    spiked = True
            final_eval = evaluate_fixed_batches(model, fixed_batches, args.device)
            row = {
                "lr": lr,
                "eps": eps,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "eval_loss": final_eval["loss"],
                "eval_accuracy": final_eval["accuracy"],
                "spiked_above_10": spiked,
            }
            results.append(row)
            print(
                "adam sweep: "
                f"lr={lr:.0e} eps={eps:.0e} "
                f"loss={row['eval_loss']:.4f} acc={row['eval_accuracy']:.4f} "
                f"spike_gt_10={spiked}"
            )
    return results


def run_epoch_profiled(
    model,
    loader,
    optimizer,
    max_batches=None,
    device="cpu",
    profile_sync=False,
):
    profile = {key: 0.0 for key in PROFILE_KEYS}
    steps = []
    iterator = iter(loader)
    batch_idx = 0

    while max_batches is None or batch_idx < max_batches:
        start = _profile_start(device, sync=False)
        try:
            images, labels = next(iterator)
        except StopIteration:
            break
        _add_time(profile, "data_loading", start)

        start = _profile_start(device, sync=False)
        targets = labels_to_list(labels)
        _add_time(profile, "label_conversion", start)

        start = _profile_start(device, sync=profile_sync)
        x = orthon_tensor_from_images(images, device)
        _add_time(profile, "input_conversion", start, device, profile_sync)

        model.train()
        optimizer.zero_grad()
        start = _profile_start(device, sync=profile_sync)
        logits = model(x)
        _add_time(profile, "forward", start, device, profile_sync)

        start = _profile_start(device, sync=profile_sync)
        loss = orthon.cross_entropy(logits, targets)
        loss_value = float(loss.item())
        warn_if_unstable_loss(loss_value, f"training batch {batch_idx}")
        _add_time(profile, "loss", start, device, profile_sync)

        start = _profile_start(device, sync=profile_sync)
        loss.backward()
        _add_time(profile, "backward", start, device, profile_sync)

        start = _profile_start(device, sync=profile_sync)
        optimizer.step()
        _add_time(profile, "optimizer", start, device, profile_sync)

        start = _profile_start(device, sync=profile_sync)
        steps.append({
            "loss": loss_value,
            "accuracy": classification_accuracy(logits, targets),
            "num_examples": len(targets),
        })
        _add_time(profile, "metrics", start, device, profile_sync)
        batch_idx += 1

    summary = weighted_metric_average(steps)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
        "batches": len(steps),
    }, profile


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


def evaluate_fixed_batches(model, batches, device="cpu", check_initial=False):
    steps = []
    first_logits = None
    first_loss = None
    for batch_idx, (images, labels) in enumerate(batches):
        targets = labels_to_list(labels)
        x = orthon_tensor_from_images(images, device)
        with orthon.no_grad():
            model.eval()
            logits = model(x)
            loss = orthon.cross_entropy(logits, targets)
        loss_value = float(loss.item())
        warn_if_unstable_loss(loss_value, f"evaluation batch {batch_idx}")
        if batch_idx == 0:
            first_logits = logits
            first_loss = loss_value
        steps.append({
            "loss": loss_value,
            "accuracy": classification_accuracy(logits, targets),
            "num_examples": len(targets),
        })

    summary = weighted_metric_average(steps)
    if check_initial and first_logits is not None:
        warn_if_suspicious_initial_values(first_loss, first_logits)
    return {
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "samples": summary["num_examples"],
        "batches": len(steps),
    }


def evaluate_profiled(model, loader, max_batches=None, device="cpu", profile_sync=False):
    start = _profile_start(device, sync=profile_sync)
    result = evaluate(model, loader, max_batches, device)
    if profile_sync:
        maybe_synchronize(device)
    return result, time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 CNN with Orthon.")
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 data directory.")
    parser.add_argument("--output", default="examples/results/cifar10_cnn.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", choices=["small", "minivgg"], default="small")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--vgg-base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--decay-bias", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.5,
        help="Scale initial weights in this CIFAR example only.",
    )
    parser.add_argument("--debug-logits", action="store_true")
    parser.add_argument("--debug-train", action="store_true")
    parser.add_argument("--debug-all-params", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--grad-clip-value", type=float, default=None)
    parser.add_argument("--sweep-adam", action="store_true")
    parser.add_argument("--sweep-epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--lr-schedule", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--lr-step-epochs", default="")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-sync",
        action="store_true",
        help="Synchronize CUDA around profiled regions for accurate GPU timing; slower but more accurate.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overfit-small", action="store_true")
    parser.add_argument("--overfit-batches", type=int, default=2)
    parser.add_argument(
        "--overfit-seed",
        type=int,
        default=None,
        help="Seed used when collecting fixed overfit batches; defaults to --seed.",
    )
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument(
        "--overfit-eval-test",
        action="store_true",
        help="In overfit mode, also evaluate the normal test loader each epoch.",
    )
    parser.add_argument(
        "--compare-cpu-cuda-step",
        action="store_true",
        help="Compare one CPU and CUDA forward/backward pass on the same batch.",
    )
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
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use tiny generated CIFAR-shaped batches for smoke tests; no dataset files needed.",
    )
    return parser.parse_args()


def limit_from_arg(value):
    return None if value == 0 else value


def compare_cpu_cuda_step(args, fixed_batches):
    if not orthon.cuda_enabled():
        print("CPU/CUDA step comparison skipped: CUDA is disabled in this build.")
        return None

    images, labels = fixed_batches[0]
    targets = labels_to_list(labels)
    results = {}

    for device in ["cpu", "cuda"]:
        random.seed(args.seed)
        orthon.manual_seed(args.seed)   # The framework RNG is separate from Python's global random.
        model = make_model(args)
        model.to(device)
        x = orthon_tensor_from_images(images, device)
        model.train()
        model.zero_grad()
        logits = model(x)
        loss = orthon.cross_entropy(logits, targets)
        loss_value = float(loss.item())
        warn_if_unstable_loss(loss_value, f"{device} comparison")
        loss.backward()
        results[device] = {
            "loss": loss_value,
            "grad_norm": grad_norm(model.parameters()),
        }

    loss_diff = abs(results["cpu"]["loss"] - results["cuda"]["loss"])
    grad_diff = abs(results["cpu"]["grad_norm"] - results["cuda"]["grad_norm"])
    close = loss_diff < 1e-4 and grad_diff < 1e-3
    print(
        "CPU/CUDA one-step comparison: "
        f"cpu_loss={results['cpu']['loss']:.6f} "
        f"cuda_loss={results['cuda']['loss']:.6f} "
        f"loss_diff={loss_diff:.6e} | "
        f"cpu_grad_norm={results['cpu']['grad_norm']:.6f} "
        f"cuda_grad_norm={results['cuda']['grad_norm']:.6f} "
        f"grad_diff={grad_diff:.6e} | close={close}"
    )
    return {**results, "loss_diff": loss_diff, "grad_norm_diff": grad_diff, "close": close}


def main():
    args = parse_args()
    if args.overfit_seed is None:
        args.overfit_seed = args.seed
    if args.overfit_small and args.augment:
        print("warning: --augment is disabled in --overfit-small mode so fixed batches stay fixed.")
        args.augment = False
    step_epochs = parse_lr_step_epochs(args.lr_step_epochs)
    # random.seed is kept because --seed is this script's documented
    # Python-side seed, but Orthon initialization is now governed by
    # manual_seed: the framework RNG is separate from Python's.
    random.seed(args.seed)
    orthon.manual_seed(args.seed)

    if args.device == "cuda" and not orthon.cuda_enabled():
        print("CUDA device requested, but this Orthon build has CUDA disabled.")
        return
    if args.profile and args.device == "cuda" and not args.profile_sync:
        print("Profile note: CUDA timings are asynchronous; use --profile-sync for accurate GPU section timing.")

    if args.synthetic_data:
        train_batch_count = max(
            1,
            args.overfit_batches if args.overfit_small else limit_from_arg(args.max_train_batches) or 2,
        )
        test_batch_count = max(1, limit_from_arg(args.max_test_batches) or 1)
        train_loader, test_loader = build_synthetic_cifar10_loaders(
            args.batch_size,
            train_batches=train_batch_count,
            test_batches=test_batch_count,
            seed=args.overfit_seed if args.overfit_small else args.seed,
        )
        print(
            "using synthetic CIFAR-shaped data: "
            f"train_batches={train_batch_count} test_batches={test_batch_count}"
        )
    else:
        train_loader, test_loader = build_cifar10_loaders(
            args.data_dir,
            args.batch_size,
            download=not args.no_download,
            seed=args.overfit_seed if args.overfit_small else args.seed,
            shuffle_train=not args.disable_shuffle,
            train_augment=args.augment,
        )

    model = make_model(args)
    model.to(args.device)
    optimizer = make_optimizer(
        args.optimizer,
        model.parameters(),
        args.lr,
        adam_eps=args.adam_eps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        decay_bias=args.decay_bias,
    )
    max_train_batches = limit_from_arg(args.max_train_batches)
    max_test_batches = limit_from_arg(args.max_test_batches)

    metrics = {
        "framework": "orthon",
        "task": "cifar10_cnn",
        "config": vars(args),
        "epochs": [],
    }

    if args.debug_logits:
        print_first_batch_logit_stats(model, train_loader, args.device)
    if args.grad_clip_norm is not None or args.grad_clip_value is not None:
        print(
            "gradient clipping active: "
            f"norm={args.grad_clip_norm} value={args.grad_clip_value}"
        )

    fixed_batches = None
    fixed_fingerprints = None
    initial = None
    if args.overfit_small:
        fixed_batches = collect_fixed_batches(train_loader, args.overfit_batches)
        fixed_fingerprints = batch_fingerprints(fixed_batches)
        print(
            "Overfit mode: training on fixed "
            f"{len(fixed_batches)} batches / {num_examples(fixed_batches)} examples"
        )
        initial = evaluate_fixed_batches(
            model, fixed_batches, args.device, check_initial=True)
        metrics["initial"] = initial
        metrics["fixed_batch_fingerprints"] = fixed_fingerprints
        print(
            "initial fixed batches: "
            f"loss={initial['loss']:.4f} acc={initial['accuracy']:.4f}"
        )
        if args.compare_cpu_cuda_step:
            metrics["cpu_cuda_step_comparison"] = compare_cpu_cuda_step(args, fixed_batches)
    elif args.compare_cpu_cuda_step:
        comparison_batches = collect_fixed_batches(train_loader, 1)
        metrics["cpu_cuda_step_comparison"] = compare_cpu_cuda_step(args, comparison_batches)

    if args.sweep_adam:
        if fixed_batches is None:
            fixed_batches = collect_fixed_batches(train_loader, args.overfit_batches)
            print(
                "Overfit mode: training on fixed "
                f"{len(fixed_batches)} batches / {num_examples(fixed_batches)} examples"
            )
        metrics["adam_sweep"] = run_adam_sweep(args, fixed_batches)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote metrics to {output}")
        return

    for epoch in range(1, args.epochs + 1):
        current_lr = apply_lr_schedule(optimizer, args, epoch, step_epochs)
        if conv2d_profile_enabled():
            reset_conv2d_profile()
        start = time.time()
        profile = None
        if args.overfit_small:
            if batch_fingerprints(fixed_batches) != fixed_fingerprints:
                raise RuntimeError("fixed overfit batches changed unexpectedly")
            train = run_epoch_fixed_batches(
                model,
                fixed_batches,
                optimizer,
                args.device,
                args.debug_train,
                args.debug_all_params,
                args.grad_clip_norm,
                args.grad_clip_value,
            )
            test = evaluate_fixed_batches(model, fixed_batches, args.device)
            if args.overfit_eval_test:
                normal_test = evaluate(model, test_loader, max_test_batches, args.device)
                metrics.setdefault("test_epochs", []).append({
                    "epoch": epoch,
                    "loss": normal_test["loss"],
                    "accuracy": normal_test["accuracy"],
                    "samples": normal_test["samples"],
                })
        elif args.profile:
            train, profile = run_epoch_profiled(
                model,
                train_loader,
                optimizer,
                max_train_batches,
                args.device,
                args.profile_sync,
            )
            test, eval_seconds = evaluate_profiled(
                model,
                test_loader,
                max_test_batches,
                args.device,
                args.profile_sync,
            )
            profile["evaluation"] = eval_seconds
        else:
            train = run_epoch(model, train_loader, optimizer, max_train_batches, args.device)
            test = evaluate(model, test_loader, max_test_batches, args.device)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "eval_loss": test["loss"],
            "eval_accuracy": test["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "train_samples": train["samples"],
            "eval_samples": test["samples"],
            "test_samples": test["samples"],
            "lr": current_lr,
            "seconds": elapsed,
        }
        metrics["epochs"].append(row)
        eval_label = "fixed eval" if args.overfit_small else "test"
        print(
            f"epoch {epoch}: "
            f"lr={current_lr:.6g} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
            f"{eval_label} loss={row['eval_loss']:.4f} acc={row['eval_accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )
        if args.overfit_small and args.overfit_eval_test:
            print(
                "  normal test: "
                f"loss={normal_test['loss']:.4f} acc={normal_test['accuracy']:.4f}"
            )
        if profile is not None:
            percentages = profile_percent(profile)
            train_batches = train.get("batches", 0)
            eval_batches = test.get("batches", 0)
            train_total = sum(profile[key] for key in PROFILE_KEYS if key != "evaluation")
            row["profile"] = profile
            row["profile_percent"] = percentages
            row["seconds_per_train_batch"] = (
                train_total / train_batches if train_batches else 0.0
            )
            row["seconds_per_train_sample"] = (
                train_total / train["samples"] if train["samples"] else 0.0
            )
            _print_profile(
                profile,
                train_batches=train_batches,
                train_samples=train["samples"],
                eval_batches=eval_batches,
                eval_samples=test["samples"],
            )
        if conv2d_profile_enabled():
            summary = conv2d_profile_summary()
            if summary:
                print(summary, end="" if summary.endswith("\n") else "\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics to {output}")


if __name__ == "__main__":
    main()