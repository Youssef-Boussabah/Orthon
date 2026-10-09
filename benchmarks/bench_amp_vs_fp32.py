"""Benchmark Orthon FP32 training against explicit FP16 AMP training on CUDA.

The benchmark runs FP32 and AMP in the same process on the same synthetic
workload, then reports separate continuous-region observations without a statistical winner.  AMP here
means explicit FP16 model/input tensors plus MixedPrecisionOptimizer and
GradScaler; this benchmark intentionally does not use autocast.
"""

from benchmarks._support.training import measured, configure, options, run as run_measurement
from benchmarks._support.environment import synchronizer, cuda_available

import argparse

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.nn import Module, Parameter, ReLU, Sequential
from orthon.optim import Adam, SGD


PRESETS = {
    "smoke": {
        "batch_size": 256,
        "input_dim": 512,
        "hidden_size": 1024,
        "layers": 3,
        "steps": 10,
        "warmup_steps": 2,
    },
    "medium": {
        "batch_size": 1024,
        "input_dim": 1024,
        "hidden_size": 2048,
        "layers": 4,
        "steps": 30,
        "warmup_steps": 5,
    },
    "large": {
        "batch_size": 1024,
        "input_dim": 2048,
        "hidden_size": 4096,
        "layers": 4,
        "steps": 50,
        "warmup_steps": 10,
    },
}

OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
}


class ConstantLinear(Module):
    """Linear layer with fast constant initialization for large benchmarks."""

    def __init__(self, in_features, out_features, weight_value):
        super().__init__()
        self.weight = Parameter(ortho.full([in_features, out_features], weight_value))
        self.bias = Parameter(ortho.zeros([out_features]))

    def forward(self, x):
        return x @ self.weight + self.bias


class ConstantConv2d(Module):
    """Conv2d layer with fast constant initialization for optional CNN runs."""

    def __init__(self, in_channels, out_channels, kernel_size, padding, weight_value):
        super().__init__()
        self.weight = Parameter(
            ortho.full([out_channels, in_channels, kernel_size, kernel_size], weight_value)
        )
        self.bias = Parameter(ortho.zeros([out_channels]))
        self._padding = padding

    def forward(self, x):
        return ortho.conv2d(x, self.weight, self.bias, 1, 1, self._padding, self._padding)


def preset_defaults(preset):
    return dict(PRESETS[preset])


def resolve_config(args):
    config = preset_defaults(args.preset)
    overridden = False
    for key in ("batch_size", "input_dim", "hidden_size", "layers", "steps", "warmup_steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
            overridden = True
    config["preset_label"] = "custom" if overridden else args.preset
    return config


def make_mlp(input_dim, hidden_size, layers, output_dim, device, dtype):
    modules = []
    in_features = input_dim
    weight_value = 1.0 / max(float(input_dim), float(hidden_size))
    for _ in range(layers):
        modules.append(ConstantLinear(in_features, hidden_size, weight_value))
        modules.append(ReLU())
        in_features = hidden_size
    modules.append(ConstantLinear(in_features, output_dim, weight_value))
    model = Sequential(*modules).to(device)
    if dtype == "float16":
        model.half()
    return model


def make_cnn(input_dim, hidden_size, layers, output_dim, device, dtype):
    modules = []
    in_channels = 3
    channels = hidden_size
    weight_value = 1.0 / max(64.0, float(channels * 9))
    for _ in range(layers):
        modules.append(ConstantConv2d(in_channels, channels, 3, 1, weight_value))
        modules.append(ReLU())
        in_channels = channels
    modules.append(ConstantConv2d(in_channels, output_dim, 1, 0, weight_value))
    model = Sequential(*modules).to(device)
    if dtype == "float16":
        model.half()
    return model


def make_model(args, config, dtype):
    if args.model == "mlp":
        return make_mlp(
            config["input_dim"],
            config["hidden_size"],
            config["layers"],
            args.output_dim,
            args.device,
            dtype,
        )
    return make_cnn(
        config["input_dim"],
        config["hidden_size"],
        config["layers"],
        args.output_dim,
        args.device,
        dtype,
    )


def make_batch(args, config, dtype):
    if args.model == "mlp":
        shape = [config["batch_size"], config["input_dim"]]
    else:
        shape = [config["batch_size"], 3, config["input_dim"], config["input_dim"]]
    x = ortho.full(shape, args.input_value).to(args.device)
    if dtype != "float32":
        x = ortho.cast(x, dtype)
    return x


def loss_fn(model, x):
    out = model(x)
    return ortho.mean(out * out)


def sync(device):
    synchronizer(device, "orthon")()


def scalar_float(tensor):
    return float(ortho.cast(tensor.to("cpu"), "float32").item())


def optimizer_kwargs(args):
    kwargs = {"lr": args.lr}
    if args.optimizer == "adam":
        kwargs["eps"] = args.adam_eps
    return kwargs


def run_fp32(args, config):
    model = make_model(args, config, "float32")
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), **optimizer_kwargs(args))
    x = make_batch(args, config, "float32")

    for _ in range(config["warmup_steps"]):
        optimizer.zero_grad()
        loss = loss_fn(model, x)
        loss.backward()
        optimizer.step()
    sync(args.device)

    last_loss = None
    with measured("bench_amp_vs_fp32:start:run_fp32", lambda: sync(args.device), calls=config['steps'], device=args.device, implementation="orthon", dtype="float32", config=dict(vars(args), **config)) as _region_start:
        for _ in range(config["steps"]):
            optimizer.zero_grad()
            last_loss = loss_fn(model, x)
            last_loss.backward()
            optimizer.step()
        sync(args.device)
    elapsed = _region_start["seconds"]
    return {
        "ms_per_step": elapsed * 1000.0 / config["steps"],
        "samples_per_sec": config["batch_size"] * config["steps"] / elapsed,
        "final_loss": scalar_float(last_loss),
    }


def run_amp(args, config):
    model = make_model(args, config, "float16")
    optimizer = MixedPrecisionOptimizer(
        model,
        OPTIMIZERS[args.optimizer],
        **optimizer_kwargs(args),
    )
    scaler = GradScaler(
        init_scale=args.init_scale,
        growth_interval=args.growth_interval,
    )
    x = make_batch(args, config, "float16")

    for _ in range(config["warmup_steps"]):
        optimizer.zero_grad()
        loss = loss_fn(model, x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    sync(args.device)

    last_loss = None
    with measured("bench_amp_vs_fp32:start:run_amp", lambda: sync(args.device), calls=config['steps'], device=args.device, implementation="orthon", dtype="float16", config=dict(vars(args), **config)) as _region_start:
        for _ in range(config["steps"]):
            optimizer.zero_grad()
            last_loss = loss_fn(model, x)
            scaler.scale(last_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        sync(args.device)
    elapsed = _region_start["seconds"]
    return {
        "ms_per_step": elapsed * 1000.0 / config["steps"],
        "samples_per_sec": config["batch_size"] * config["steps"] / elapsed,
        "final_loss": scalar_float(last_loss),
        "scaler_final_scale": scaler.scale_factor,
    }


def positive_int(parser, name, value):
    if value < 1:
        parser.error(f"--{name.replace('_', '-')} must be at least 1")


def parse_args():
    parser = argparse.ArgumentParser(description="Orthon FP32 vs FP16 AMP benchmark.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup", dest="warmup_steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--hidden", dest="hidden_size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--input-value", type=float, default=0.01)
    parser.add_argument("--init-scale", type=float, default=128.0)
    parser.add_argument("--growth-interval", type=int, default=2000)
    options(parser)
    args = configure(parser.parse_args())

    config = resolve_config(args)
    for key in ("steps", "batch_size", "input_dim", "hidden_size", "layers"):
        positive_int(parser, key, config[key])
    if config["warmup_steps"] < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.output_dim < 1:
        parser.error("--output-dim must be at least 1")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.input_value == 0.0:
        parser.error("--input-value must be non-zero")
    if args.model == "cnn" and config["input_dim"] > 256:
        parser.error("--model cnn uses --input-dim as image size; choose <= 256")
    return args, config


def print_summary(args, config, fp32, amp):
    speedup = fp32["ms_per_step"] / amp["ms_per_step"] if amp["ms_per_step"] > 0 else 0.0
    limitations = (
        "explicit FP16 AMP only; no autocast; FP32 master weights; "
        "GradScaler CUDA unscale/finite checks are fused; master sync is in-place; "
        "2D FP16 matmul uses cuBLAS"
    )
    model_name = (
        f"mlp_{config['input_dim']}_{config['hidden_size']}_{args.output_dim}"
        if args.model == "mlp"
        else f"cnn_{config['input_dim']}_{config['hidden_size']}_{args.output_dim}"
    )

    print("benchmark_name: amp_vs_fp32")
    print(f"preset: {config['preset_label']}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"optimizer: {args.optimizer}")
    print(f"batch_size: {config['batch_size']}")
    print(f"input_dim: {config['input_dim']}")
    print(f"hidden_size: {config['hidden_size']}")
    print(f"layers: {config['layers']}")
    print(f"warmup_steps: {config['warmup_steps']}")
    print(f"measured_steps: {config['steps']}")
    print(f"fp32_ms_per_step: {fp32['ms_per_step']:.3f}")
    print(f"amp_ms_per_step: {amp['ms_per_step']:.3f}")
    print(f"fp32_samples_per_sec: {fp32['samples_per_sec']:.2f}")
    print(f"amp_samples_per_sec: {amp['samples_per_sec']:.2f}")
    print(f"unverified_region_ratio: {speedup:.3f}")
    print(f"fp32_final_loss: {fp32['final_loss']:.6f}")
    print(f"amp_final_loss: {amp['final_loss']:.6f}")
    print(f"scaler_initial_scale: {args.init_scale:.1f}")
    print(f"scaler_final_scale: {amp['scaler_final_scale']:.1f}")
    print(f"notes: {limitations}")
    print(
        "csv_summary,"
        "benchmark_name=amp_vs_fp32,"
        f"preset={config['preset_label']},"
        f"model={args.model},"
        f"model_name={model_name},"
        f"device={args.device},"
        f"optimizer={args.optimizer},"
        f"batch_size={config['batch_size']},"
        f"input_dim={config['input_dim']},"
        f"hidden_size={config['hidden_size']},"
        f"layers={config['layers']},"
        f"warmup_steps={config['warmup_steps']},"
        f"measured_steps={config['steps']},"
        f"fp32_ms_per_step={fp32['ms_per_step']:.3f},"
        f"amp_ms_per_step={amp['ms_per_step']:.3f},"
        f"fp32_samples_per_sec={fp32['samples_per_sec']:.2f},"
        f"amp_samples_per_sec={amp['samples_per_sec']:.2f},"
        f"unverified_region_ratio={speedup:.3f},"
        f"fp32_final_loss={fp32['final_loss']:.6f},"
        f"amp_final_loss={amp['final_loss']:.6f},"
        f"scaler_initial_scale={args.init_scale:.1f},"
        f"scaler_final_scale={amp['scaler_final_scale']:.1f},"
        f"notes={limitations}"
    )


def main():
    args, config = parse_args()
    if not cuda_available("cuda:0"):
        print("benchmark_name=amp_vs_fp32,status=skipped,reason=cuda_unavailable")
        return

    fp32 = run_fp32(args, config)
    amp = run_amp(args, config)
    print_summary(args, config, fp32, amp)


if __name__ == "__main__":
    run_measurement(main, "bench_amp_vs_fp32")
