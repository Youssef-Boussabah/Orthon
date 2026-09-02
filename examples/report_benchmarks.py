"""Report training benchmark JSON files in grouped tables."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path("examples/results")


def load_training_rows(results_dir, include_legacy=False):
    rows = []
    for path in sorted(results_dir.glob("training_benchmark_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        preset = data.get("preset") or config.get("preset", "unknown")
        if preset == "unknown" and not include_legacy:
            continue
        for workload in data.get("workloads", []):
            row = {
                "preset": workload.get("preset", preset),
                "workload": workload["workload"],
                "framework": data["framework"],
                "device": data["device"],
                "accuracy": workload["final_accuracy"],
                "epoch_time_sec": workload["epoch_time_sec"],
                "samples_per_sec": workload["samples_per_sec"],
                "loss_decrease": workload["loss_decrease"],
                "batch_size": workload.get("batch_size"),
                "epochs": workload.get("epochs_requested", config.get("epochs")),
                "max_train_batches": workload.get("max_train_batches", config.get("max_train_batches")),
                "max_test_batches": workload.get("max_test_batches", config.get("max_test_batches")),
                "warmup": workload.get("warmup", config.get("warmup")),
                "path": str(path),
            }
            row["config_key"] = config_key(row)
            rows.append(row)
    return rows


def config_key(row):
    return (
        row["preset"],
        row["workload"],
        row["batch_size"],
        row["epochs"],
        row["max_train_batches"],
        row["max_test_batches"],
        row["warmup"],
    )


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def find_row(rows, key, framework, device):
    return next(
        (
            row for row in rows
            if row["config_key"] == key
            and row["framework"] == framework
            and row["device"] == device
        ),
        None,
    )


def speedup_against_orthon_cpu(row, rows):
    baseline = find_row(rows, row["config_key"], "orthon", "cpu")
    if baseline is None or row["epoch_time_sec"] <= 0:
        return None
    return baseline["epoch_time_sec"] / row["epoch_time_sec"]


def grouped_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["preset"]].append(row)
    preset_order = {"smoke": 0, "small": 1, "medium": 2}
    return sorted(groups.items(), key=lambda item: (preset_order.get(item[0], 99), item[0]))


def render_table(rows):
    headers = [
        "workload",
        "framework",
        "device",
        "batch",
        "batches",
        "accuracy",
        "epoch time",
        "samples/sec",
        "speedup",
    ]
    table_rows = []
    for row in sorted(rows, key=lambda r: (r["workload"], r["framework"], r["device"])):
        table_rows.append([
            row["workload"],
            row["framework"],
            row["device"],
            str(row["batch_size"]),
            str(row["max_train_batches"]),
            fmt(row["accuracy"]),
            fmt(row["epoch_time_sec"]),
            fmt(row["samples_per_sec"], 2),
            fmt(speedup_against_orthon_cpu(row, rows), 2),
        ])

    widths = [
        max(len(str(value)) for value in [header] + [row[i] for row in table_rows])
        for i, header in enumerate(headers)
    ]
    lines = [
        " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        " | ".join("-" * width for width in widths),
    ]
    for row in table_rows:
        lines.append(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
    return "\n".join(lines)


def compute_pair_speedups(rows):
    lines = []
    keys = sorted(set(row["config_key"] for row in rows))
    for key in keys:
        preset, workload, batch_size, epochs, train_batches, _, warmup = key
        label = (
            f"{preset}/{workload} "
            f"(batch={batch_size}, epochs={epochs}, train_batches={train_batches}, warmup={warmup})"
        )

        for framework in ["orthon", "torch"]:
            cpu = find_row(rows, key, framework, "cpu")
            cuda = find_row(rows, key, framework, "cuda")
            if cpu and cuda and cuda["epoch_time_sec"] > 0:
                lines.append(
                    f"{label}: {framework} CUDA vs CPU = "
                    f"{cpu['epoch_time_sec'] / cuda['epoch_time_sec']:.2f}x"
                )

        for device in ["cpu", "cuda"]:
            orthon = find_row(rows, key, "orthon", device)
            torch = find_row(rows, key, "torch", device)
            if orthon and torch and torch["epoch_time_sec"] > 0:
                lines.append(
                    f"{label}: Orthon/PyTorch time on {device} = "
                    f"{orthon['epoch_time_sec'] / torch['epoch_time_sec']:.2f}x; "
                    f"Orthon speedup over PyTorch = "
                    f"{torch['epoch_time_sec'] / orthon['epoch_time_sec']:.2f}x"
                )
    return lines


def render_console(rows):
    parts = []
    for preset, preset_rows in grouped_rows(rows):
        parts.append(f"Preset: {preset}")
        if preset == "smoke":
            parts.append("  note: smoke results validate execution and are not representative throughput numbers")
        parts.append(render_table(preset_rows))
        parts.append("")
    speedups = compute_pair_speedups(rows)
    parts.append("Speedups for matching configs only:")
    parts.extend(f"- {line}" for line in speedups)
    return "\n".join(parts), speedups


def render_markdown(rows, speedups):
    lines = [
        "# Benchmark Report",
        "",
        "Smoke results validate execution and should not be read as representative throughput.",
        "Small and medium presets reduce overhead dominance by increasing batch counts and batch sizes.",
        "",
    ]
    for preset, preset_rows in grouped_rows(rows):
        lines.extend([
            f"## Preset: {preset}",
            "",
            "| workload | framework | device | batch | train batches | accuracy | epoch time | samples/sec | speedup vs Orthon CPU |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in sorted(preset_rows, key=lambda r: (r["workload"], r["framework"], r["device"])):
            lines.append(
                f"| {row['workload']} | {row['framework']} | {row['device']} | "
                f"{row['batch_size']} | {row['max_train_batches']} | "
                f"{fmt(row['accuracy'])} | {fmt(row['epoch_time_sec'])} | "
                f"{fmt(row['samples_per_sec'], 2)} | "
                f"{fmt(speedup_against_orthon_cpu(row, preset_rows), 2)} |"
            )
        lines.append("")
    lines.extend(["## Speedups", ""])
    lines.extend(f"- {line}" for line in speedups)
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Tiny CUDA runs are dominated by launch overhead, synchronization, Python dispatch, and data conversion. "
        "Orthon can look competitive on smoke workloads because its measured work is very small. "
        "PyTorch is expected to win at scale as larger batches amortize overhead and exercise optimized kernels.",
        "",
    ])
    return "\n".join(lines)


def write_csv(rows, path):
    fieldnames = [
        "preset",
        "workload",
        "framework",
        "device",
        "batch_size",
        "epochs",
        "max_train_batches",
        "max_test_batches",
        "warmup",
        "accuracy",
        "epoch_time_sec",
        "samples_per_sec",
        "loss_decrease",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["preset"], r["workload"], r["framework"], r["device"])):
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON outputs.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include older training_benchmark_*.json files without preset metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = load_training_rows(results_dir, include_legacy=args.include_legacy)
    if not rows:
        raise SystemExit(f"no training_benchmark_*.json files found in {results_dir}")

    report, speedups = render_console(rows)
    print(report)

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(rows, speedups), encoding="utf-8")
        print(f"wrote markdown report to {path}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, path)
        print(f"wrote csv report to {path}")


if __name__ == "__main__":
    main()
