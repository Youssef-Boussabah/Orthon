#!/usr/bin/env python3
"""Compare Orthon and PyTorch eager workloads with shared completed-call timing.

Case setup precedes timing; numerical checks gate claims. See docs/benchmarking.md
for methodology and use --help for the comparison matrix and output options."""

import argparse
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from typing import Optional

# Imports. Both frameworks are mandatory here: this script exists only to
# compare them, so a missing one is an error, not a skipped row.
try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("compare_pytorch.py needs NumPy: pip install numpy")

try:
    import orthon as ortho
except ImportError:
    sys.exit("compare_pytorch.py needs Orthon: pip install -e .")

try:
    import torch
except ImportError:
    sys.exit('compare_pytorch.py needs PyTorch: pip install "torch>=2.0"')


from benchmarks._support import comparison_cases
from benchmarks._support.comparison_cases import BENCHMARK_ORDER, BENCHMARKS, Case
from benchmarks._support.environment import (
    cuda_available,
    global_warmup as device_warmup,
    provenance,
    synchronizer,
)
from benchmarks._support.results import case_result, document, write_json as serialize_json
from benchmarks._support.timing import (
    Statistics,
    TimingConfig,
    comparison,
    measure,
    measure_pair,
    quartiles as _quartiles,
)

ORTHON_DEVICE = "cuda:0"
TORCH_DEVICE = "cuda:0"


# Environment
def _nvidia_smi(query: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except Exception:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _orthon_version() -> str:
    try:
        return version("orthon")
    except Exception:
        return "unknown"


def _orthon_sm_targets() -> str:
    """Which SM architectures the installed Orthon extension actually carries.

    Reported because a PTX-JIT fallback and a native cubin are not the same
    measurement. Best-effort: cuobjdump is not required to run the benchmark.
    """
    try:
        import orthon._C as _c
        path = _c.__file__
    except Exception:
        return "unavailable"
    try:
        out = subprocess.run(["cuobjdump", "--list-elf", path],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return "unavailable (cuobjdump not found)"
    if out.returncode != 0:
        return "unavailable"
    seen = []
    for line in out.stdout.splitlines():
        for token in line.replace(".", " ").split():
            if token.startswith("sm_") and token not in seen:
                seen.append(token)
    return ", ".join(seen) if seen else "none (PTX only)"


def collect_environment() -> dict:
    if not ORTHON_DEVICE.startswith("cuda"):
        return provenance()
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": f"sm_{props.major}{props.minor}",
        "gpu_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        "gpu_l2_cache_mib": round(getattr(props, "L2_cache_size", 0) / (1024 ** 2), 1),
        "gpu_sm_count": props.multi_processor_count,
        "driver": _nvidia_smi("driver_version") or "unknown",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "orthon_version": _orthon_version(),
        "orthon_cuda_enabled": bool(ortho.cuda_enabled()),
        "orthon_extension_sm": _orthon_sm_targets(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
        "torch_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def print_environment(env: dict) -> None:
    print("=" * 86)
    print("  Environment")
    print("=" * 86)
    order = [
        ("GPU", "gpu"),
        ("Compute capability", "gpu_capability"),
        ("GPU memory (GiB)", "gpu_memory_gb"),
        ("GPU L2 cache (MiB)", "gpu_l2_cache_mib"),
        ("SM count", "gpu_sm_count"),
        ("NVIDIA driver", "driver"),
        ("Platform", "platform"),
        ("Python", "python"),
        ("NumPy", "numpy"),
        ("PyTorch", "torch_version"),
        ("PyTorch CUDA", "torch_cuda_version"),
        ("cuDNN", "torch_cudnn_version"),
        ("Orthon", "orthon_version"),
        ("Orthon CUDA build", "orthon_cuda_enabled"),
        ("Orthon extension SM", "orthon_extension_sm"),
        ("torch matmul TF32", "torch_matmul_allow_tf32"),
        ("torch cuDNN TF32", "torch_cudnn_allow_tf32"),
        ("torch fp32 matmul precision", "torch_float32_matmul_precision"),
    ]
    for label, key in order:
        value = env.get(key)
        if value is None:
            continue
        print(f"  {label:<30} {value}")
    print()


# Timing
def ortho_sync() -> None:
    synchronizer(ORTHON_DEVICE)()


def torch_sync() -> None:
    synchronizer(TORCH_DEVICE, "torch")()


@dataclass
class Result:
    case: Case
    inner: int
    orthon_samples: list
    torch_samples: list
    check: dict
    compiled_samples: Optional[list] = None
    compile_error: Optional[str] = None

    # Derived, filled in by finalize().
    orthon_median: float = 0.0
    orthon_iqr: float = 0.0
    torch_median: float = 0.0
    torch_iqr: float = 0.0
    compiled_median: Optional[float] = None
    speedup: float = 0.0
    band: tuple = (0.0, 0.0)
    verdict: str = ""
    stability: str = ""

    def finalize(self) -> "Result":
        o = Statistics(tuple(self.orthon_samples), self.inner)
        pt = Statistics(tuple(self.torch_samples), self.inner)
        self.orthon_median, self.orthon_iqr = o.as_dict()['median'], o.as_dict()['iqr']
        self.torch_median, self.torch_iqr = pt.as_dict()['median'], pt.as_dict()['iqr']
        decision = comparison(o, pt, self.check.get('ok', False))
        self.speedup, self.band = decision['speedup'], decision['band']
        self.verdict, self.stability = decision['verdict'], decision['stability']
        if self.compiled_samples:
            self.compiled_median = _quartiles(self.compiled_samples)[1]
        return self


VERDICT_TEXT = {
    "orthon": "Orthon faster",
    "pytorch": "PyTorch faster",
    "tie": "similar",
    "unreliable": "ordering unreliable",
    "invalid": "NUMERICAL CHECK FAILED",
}


# Runner
def global_warmup(quick=False):
    device_warmup(ORTHON_DEVICE, quick)


def run_case(case: Case, cfg: TimingConfig, want_compile: bool) -> Result:
    check = case.check()
    o, pt = measure_pair(case.orthon_fn, case.torch_fn, ortho_sync, torch_sync, cfg)
    compiled_samples = None
    compile_error = None
    if want_compile:
        try:
            if case.torch_compile_factory is None:
                raise ValueError('no compiled variant for this case')
            compiled_fn = case.torch_compile_factory()
            torch_sync()
            compiled = measure(
                compiled_fn, torch_sync, replace(cfg, fixed_inner=o.inner)
            )
            compiled_samples = list(compiled.samples)
        except Exception as exc:
            compile_error = str(exc)
    return Result(
        case, o.inner, list(o.samples), list(pt.samples), check,
        compiled_samples, compile_error,
    ).finalize()


def select_cases(args) -> list:
    names = BENCHMARK_ORDER if args.benchmark is None else list(args.benchmark)
    for name in names:
        if name not in BENCHMARKS:
            sys.exit(f"unknown benchmark {name!r}; try --list")
    wanted_shape = tuple(args.shape) if args.shape else None

    cases = []
    for name in names:
        spec = BENCHMARKS[name]
        for shape in spec["shapes"]:
            if wanted_shape is not None and tuple(shape) != wanted_shape:
                continue
            cases.append((name, shape))
    if wanted_shape is not None and not cases:
        sys.exit(f"no case matches --shape {' '.join(str(s) for s in wanted_shape)}; "
                 f"try --list")
    return cases


# Reporting
def print_check_table(results: list) -> None:
    print("=" * 86)
    print("  Numerical equivalence (PyTorch eager is the reference)")
    print("=" * 86)
    header = f"  {'benchmark':<24}{'shape':<30}{'max rel diff':>14}  {'tol':>9}  status"
    print(header)
    print("  " + "-" * 82)
    for result in results:
        c = result.check
        status = "PASS" if c.get("ok") else "FAIL"
        relative = "unavailable" if c['rel'] is None else format(c['rel'], '.2e')
        print(f"  {result.case.benchmark:<24}{result.case.label:<30}"
              f"{relative:>14}  {c['tol']:>9.1e}  {status}")
    print()


def print_results(results: list) -> None:
    by_benchmark = {}
    for result in results:
        by_benchmark.setdefault(result.case.benchmark, []).append(result)

    for name in BENCHMARK_ORDER:
        group = by_benchmark.get(name)
        if not group:
            continue
        spec = BENCHMARKS[name]
        first = group[0].case
        print("=" * 86)
        print(f"  {spec['title']}  |  {first.dtype}  |  {first.mode}")
        print(f"  {first.note}")
        print("=" * 86)
        print(f"  {'shape':<30}{'Orthon ms':>11}{'IQR':>9}"
              f"{'PyTorch ms':>12}{'IQR':>9}{'speedup':>10}  verdict")
        print("  " + "-" * 82)
        for result in group:
            speedup = (
                "invalid" if result.speedup is None else format(result.speedup, ".2f") + "x"
            )
            print(f"  {result.case.label:<30}"
                  f"{result.orthon_median:>11.4f}{result.orthon_iqr:>9.4f}"
                  f"{result.torch_median:>12.4f}{result.torch_iqr:>9.4f}"
                  f"{speedup:>10}  {VERDICT_TEXT[result.verdict]}"
                  + ("" if result.stability == "stable" else "  [noisy]"))
            if result.compiled_median is not None:
                print(f"  {'  (torch.compile)':<30}{'':>11}{'':>9}"
                      f"{result.compiled_median:>12.4f}"
                      f"  observation; compiled correctness not checked; compile time excluded")
            elif result.compile_error:
                print(f"  {'  (torch.compile)':<30}unavailable: {result.compile_error}")
        print()


def print_summary(results: list) -> None:
    buckets = {"orthon": [], "pytorch": [], "tie": [], "unreliable": [], "invalid": []}
    for result in results:
        buckets[result.verdict].append(result)

    print("=" * 86)
    print("  Summary")
    print("=" * 86)
    print(f"  cases measured             {len(results)}")
    print(f"  Orthon >= +10% faster      {len(buckets['orthon'])}")
    print(f"  PyTorch >= +10% faster     {len(buckets['pytorch'])}")
    print(f"  within +/-10% (similar)    {len(buckets['tie'])}")
    print(f"  ordering unreliable        {len(buckets['unreliable'])}")
    print(f"  numerical check failed     {len(buckets['invalid'])}")
    print()

    if buckets["orthon"]:
        print("  Orthon rows (>= 10%, nonoverlapping IQR intervals):")
        for result in sorted(buckets["orthon"], key=lambda r: -r.speedup):
            flag = "" if result.stability == "stable" else "   [noisy]"
            print(f"    {result.speedup:>6.2f}x  {result.case.benchmark} "
                  f"{result.case.label}{flag}")
        print()
    if buckets["invalid"]:
        print("  INVALID rows carry no performance claim:")
        for result in buckets["invalid"]:
            relative = (
                'unavailable' if result.check['rel'] is None
                else format(result.check['rel'], '.2e')
            )
            print(f"    {result.case.benchmark} {result.case.label}: "
                  f"rel {relative} > tol {result.check['tol']:.1e}")
        print()


def print_highlight(result: Result, env: dict) -> None:
    case = result.case
    title = BENCHMARKS[case.benchmark]["title"]
    width = 60
    print()
    print("=" * width)
    print("  Orthon vs. PyTorch eager")
    print(f"  {title}  |  {case.label}  |  {case.dtype}")
    print(f"  Device: {ORTHON_DEVICE}")
    if ORTHON_DEVICE.startswith('cuda'):
        print(f"  GPU: {env['gpu']}")
    print("=" * width)
    print(f"  Orthon      {result.orthon_median:8.3f} ms")
    print(f"  PyTorch     {result.torch_median:8.3f} ms")
    speedup = 'invalid' if result.speedup is None else format(result.speedup, '.2f') + 'x'
    print(f"  Speedup     {speedup:>9}")
    print("=" * width)
    print(f"  median of {len(result.orthon_samples)} alternating samples, "
          f"{result.inner} iterations each")
    relative = (
        'unavailable' if result.check['rel'] is None
        else format(result.check['rel'], '.2e')
    )
    print(f"  max relative difference vs PyTorch: {relative} "
          f"(tolerance {result.check['tol']:.0e})")
    print("=" * width)
    print()


def write_markdown(path: str, env: dict, results: list) -> None:
    lines = ["# Orthon vs. PyTorch - generated result matrix", ""]
    lines.append(f"Generated by `benchmarks/compare_pytorch.py` on "
                 f"{provenance()['timestamp_utc']}.")
    lines.append("")
    lines.append("| Environment | Value |")
    lines.append("| --- | --- |")
    for key, value in env.items():
        if value is None:
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")

    by_benchmark = {}
    for result in results:
        by_benchmark.setdefault(result.case.benchmark, []).append(result)

    for name in BENCHMARK_ORDER:
        group = by_benchmark.get(name)
        if not group:
            continue
        spec = BENCHMARKS[name]
        first = group[0].case
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append(f"dtype `{first.dtype}`, {first.mode}. {first.note}")
        lines.append("")
        lines.append("| Shape | Orthon median ms | Orthon IQR | PyTorch median ms "
                     "| PyTorch IQR | Speedup | Verdict | Max rel diff |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for result in group:
            verdict = VERDICT_TEXT[result.verdict]
            if result.stability != "stable":
                verdict += " (noisy)"
            speedup = (
                'invalid' if result.speedup is None else format(result.speedup, '.2f') + 'x'
            )
            relative = (
                'unavailable' if result.check['rel'] is None
                else format(result.check['rel'], '.2e')
            )
            lines.append(
                f"| {result.case.label} | {result.orthon_median:.4f} | "
                f"{result.orthon_iqr:.4f} | {result.torch_median:.4f} | "
                f"{result.torch_iqr:.4f} | {speedup} | {verdict} | "
                f"{relative} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  markdown written to {path}")


def write_json(path, env, results, cfg, quick):
    cases = []
    for r in results:
        for implementation, samples in [
            ('orthon', r.orthon_samples),
            ('pytorch-eager', r.torch_samples),
            ('pytorch-compiled', r.compiled_samples),
        ]:
            if samples is None:
                continue
            cases.append(case_result(
                r.case.benchmark, Statistics(tuple(samples), r.inner),
                shape=list(r.case.shape), implementation=implementation,
                device=ORTHON_DEVICE, dtype=r.case.dtype, mode=r.case.mode,
                warmup=cfg.warmup,
                correctness=(
                    'not-compared' if implementation == 'pytorch-compiled'
                    else 'pass' if r.check['ok'] else 'fail'
                ),
                notes=r.case.note, numerical_check=r.check,
                speedup=None if implementation == 'pytorch-compiled' else r.speedup,
                verdict='observation' if implementation == 'pytorch-compiled' else r.verdict,
                stability=r.stability, compile_error=r.compile_error,
            ))
    serialize_json(path, document(
        'compare_pytorch', cases,
        dict(
            representative=not quick, timing=asdict(cfg),
            ordinal=0 if ORTHON_DEVICE.startswith('cuda') else None,
            global_warmup='Orthon fixed 100 matmuls (8 in quick)', torch_policy=env,
        ),
    ))


def print_list() -> None:
    print("Available benchmarks (--benchmark NAME, --shape follows the shape form):")
    print()
    for name in BENCHMARK_ORDER:
        spec = BENCHMARKS[name]
        print(f"  {name}")
        print(f"      {spec['title']}   --shape {spec['shape_help']}")
        for shape in spec["shapes"]:
            print(f"        --shape {' '.join(str(s) for s in shape)}")
        print()


# Entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_pytorch.py",
        description=("Reproducible Orthon vs. PyTorch eager comparison on the "
                     "selected CPU or CUDA device. Inputs are pre-placed; every case "
                     "proves numerical equivalence before it is timed."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m benchmarks.compare_pytorch\n"
            "  python -m benchmarks.compare_pytorch --benchmark cross_entropy\n"
            "  python -m benchmarks.compare_pytorch --benchmark bf16_matmul "
            "--shape 1024\n"
            "  python -m benchmarks.compare_pytorch --highlight cross_entropy "
            "--shape 2048 4096\n"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--list", action="store_true",
                        help="list benchmarks and their shapes, then exit")
    parser.add_argument("--benchmark", nargs="+", metavar="NAME",
                        help="run only these benchmarks (default: all)")
    parser.add_argument("--shape", nargs="+", type=int, metavar="INT",
                        help="run only the case with exactly this shape")
    parser.add_argument("--highlight", metavar="NAME",
                        help="print the screenshot-ready card for one benchmark "
                             "(combine with --shape to pick the case)")
    parser.add_argument("--check-only", action="store_true",
                        help="run the numerical equivalence checks and stop")
    parser.add_argument("--samples", type=int, default=11,
                        help="independent samples per framework (default: 11)")
    parser.add_argument("--warmup", type=int, default=25,
                        help="warmup iterations per framework per case (default: 25)")
    parser.add_argument("--target-ms", type=float, default=40.0,
                        help="work per sample for the faster framework, ms "
                             "(default: 40)")
    parser.add_argument("--max-sample-ms", type=float, default=500.0,
                        help="cap on a single sample for the slower framework, ms "
                             "(default: 500)")
    parser.add_argument("--inner", type=int, default=None,
                        help="fix the per-sample iteration count instead of "
                             "deriving it")
    parser.add_argument("--tf32", action="store_true",
                        help="allow PyTorch TF32 for float32 matmul/conv; rows are "
                             "labelled accordingly (default: strict float32)")
    parser.add_argument("--torch-compile", action="store_true",
                        help="additionally time torch.compile where a case "
                             "provides a compiled variant; never merged into the "
                             "eager row")
    parser.add_argument("--markdown", metavar="PATH",
                        help="write the full matrix as a Markdown table")
    parser.add_argument("--json", metavar="PATH",
                        help="write the full matrix, including raw samples, as JSON")
    return parser


def main(argv=None) -> int:
    global ORTHON_DEVICE, TORCH_DEVICE
    parser = build_parser()
    args = parser.parse_args(argv)
    ORTHON_DEVICE = TORCH_DEVICE = args.device
    comparison_cases.ORTHON_DEVICE = comparison_cases.TORCH_DEVICE = args.device
    if args.quick:
        args.samples, args.warmup, args.inner = 3, 1, 1
        if args.benchmark is None:
            args.benchmark = ["elementwise_add", "reduce_sum"]
    print(f"representative={str(not args.quick).lower()}")

    if args.list:
        print_list()
        return 0

    if args.device.startswith("cuda") and not cuda_available(args.device):
        sys.exit("this comparison needs a CUDA-enabled Orthon build "
                 "(orthon.cuda_enabled() is False)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("this comparison needs a CUDA-enabled PyTorch build")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    env = collect_environment()
    print()
    print_environment(env)

    if args.highlight:
        args.benchmark = [args.highlight]

    selected = select_cases(args)

    if args.quick:
        seen = set()
        selected = [(n, s) for n, s in selected if not (n in seen or seen.add(n))]

    if args.check_only and not selected:
        parser.error("no numerical checks selected")

    cfg = TimingConfig(warmup=args.warmup, samples=args.samples,
                       target_ms=args.target_ms,
                       max_sample_ms=args.max_sample_ms,
                       fixed_inner=args.inner)

    if not args.check_only:
        print(f"  warming up {ORTHON_DEVICE} ...")
        global_warmup(args.quick)

    results = []
    failed_checks = False
    for name, shape in selected:
        case = BENCHMARKS[name]["build"](tuple(shape))
        if args.check_only:
            check = case.check()
            status = "PASS" if check["ok"] else "FAIL"
            if not check["ok"]:
                failed_checks = True
            relative = 'unavailable' if check['rel'] is None else format(check['rel'], '.2e')
            print(f"  {name:<24}{case.label:<30}rel {relative:>10}  "
                  f"tol {check['tol']:.1e}  {status}")
            continue
        print(f"  measuring {name} {case.label} ...", flush=True)
        results.append(run_case(case, cfg, args.torch_compile))

    if args.check_only:
        return 1 if failed_checks else 0

    print()
    print_check_table(results)

    if args.highlight:
        if len(results) == 1:
            print_highlight(results[0], env)
        else:
            print_results(results)
            print("  --highlight matched more than one case; add --shape to pick "
                  "exactly one.")
            return 0
    else:
        print_results(results)
        print_summary(results)

    if args.markdown:
        write_markdown(args.markdown, env, results)
    if args.json:
        write_json(args.json, env, results, cfg, args.quick)

    failed = [r for r in results if r.verdict == "invalid"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
