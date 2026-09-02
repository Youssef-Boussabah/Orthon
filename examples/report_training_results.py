"""Summarize MNIST and CIFAR-10 training result JSON files."""

import argparse
import json
import math
from pathlib import Path


WORKLOADS = [
    {
        "name": "MNIST MLP",
        "orthon": Path("examples/results/mnist_mlp.json"),
        "torch": Path("examples/results/mnist_mlp_torch.json"),
        "accuracy_threshold": 0.80,
    },
    {
        "name": "CIFAR-10 CNN",
        "orthon": Path("examples/results/cifar10_cnn.json"),
        "torch": Path("examples/results/cifar10_cnn_torch.json"),
        "accuracy_threshold": 0.15,
    },
]


def load_result(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("epochs"):
        raise ValueError(f"Result file has no epoch rows: {path}")
    return data


def loss_decrease(rows):
    return rows[0]["train_loss"] - rows[-1]["train_loss"]


def avg_seconds(rows):
    return sum(row["seconds"] for row in rows) / len(rows)


def has_nan(rows):
    keys = ["train_loss", "train_accuracy", "test_loss", "test_accuracy", "seconds"]
    for row in rows:
        for key in keys:
            value = row[key]
            if isinstance(value, float) and math.isnan(value):
                return True
    return False


def summarize_workload(spec):
    orthon = load_result(spec["orthon"])
    torch = load_result(spec["torch"])
    if orthon is None or torch is None:
        return {
            "name": spec["name"],
            "missing": True,
            "orthon_path": spec["orthon"],
            "torch_path": spec["torch"],
        }

    d_rows = orthon["epochs"]
    t_rows = torch["epochs"]
    d_final = d_rows[-1]
    t_final = t_rows[-1]
    d_loss_drop = loss_decrease(d_rows)
    t_loss_drop = loss_decrease(t_rows)
    d_avg = avg_seconds(d_rows)
    t_avg = avg_seconds(t_rows)
    speedup = d_avg / t_avg if t_avg > 0 else float("inf")
    d_has_nan = has_nan(d_rows)

    passed = (
        d_final["test_accuracy"] > spec["accuracy_threshold"]
        and d_loss_drop > 0.0
        and not d_has_nan
    )

    return {
        "name": spec["name"],
        "missing": False,
        "threshold": spec["accuracy_threshold"],
        "orthon_final_accuracy": d_final["test_accuracy"],
        "torch_final_accuracy": t_final["test_accuracy"],
        "orthon_loss_decrease": d_loss_drop,
        "torch_loss_decrease": t_loss_drop,
        "orthon_avg_seconds": d_avg,
        "torch_avg_seconds": t_avg,
        "torch_speedup": speedup,
        "orthon_has_nan": d_has_nan,
        "passed": passed,
    }


def print_summary(rows):
    header = (
        "workload      d_acc   t_acc   d_loss_drop  t_loss_drop  "
        "d_sec  t_sec  speedup  pass"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        if row["missing"]:
            print(f"{row['name']:<12} missing result file(s)")
            continue
        print(
            f"{row['name']:<12} "
            f"{row['orthon_final_accuracy']:>6.3f} "
            f"{row['torch_final_accuracy']:>7.3f} "
            f"{row['orthon_loss_decrease']:>13.4f} "
            f"{row['torch_loss_decrease']:>12.4f} "
            f"{row['orthon_avg_seconds']:>6.2f} "
            f"{row['torch_avg_seconds']:>6.2f} "
            f"{row['torch_speedup']:>8.2f}x "
            f"{'yes' if row['passed'] else 'no'}"
        )

    print("\ncriteria")
    print("--------")
    print("MNIST: Orthon final accuracy > 0.80 and train loss decreases.")
    print("CIFAR-10: Orthon final accuracy > 0.15, train loss decreases, no NaN.")


def markdown_report(rows):
    lines = [
        "# Training Report",
        "",
        "| Workload | Orthon final acc | PyTorch final acc | Orthon loss drop | PyTorch loss drop | Orthon sec/epoch | PyTorch sec/epoch | PyTorch speedup | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        if row["missing"]:
            lines.append(
                f"| {row['name']} | missing | missing | missing | missing | missing | missing | missing | no |"
            )
            continue
        lines.append(
            f"| {row['name']} | "
            f"{row['orthon_final_accuracy']:.4f} | "
            f"{row['torch_final_accuracy']:.4f} | "
            f"{row['orthon_loss_decrease']:.4f} | "
            f"{row['torch_loss_decrease']:.4f} | "
            f"{row['orthon_avg_seconds']:.2f} | "
            f"{row['torch_avg_seconds']:.2f} | "
            f"{row['torch_speedup']:.2f}x | "
            f"{'yes' if row['passed'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Criteria",
        "",
        "- MNIST passes when Orthon final accuracy is above 80% and train loss decreases.",
        "- CIFAR-10 passes when Orthon final accuracy is above 15%, train loss decreases, and no NaN is present.",
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Report Orthon training workload results.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown output path, e.g. examples/results/training_report.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [summarize_workload(spec) for spec in WORKLOADS]
    print_summary(rows)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(rows), encoding="utf-8")
        print(f"\nwrote markdown report to {args.markdown}")


if __name__ == "__main__":
    main()
