"""Compare Orthon and PyTorch training metric JSON files."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ORTHON = Path("examples/results/mnist_mlp.json")
DEFAULT_TORCH = Path("examples/results/mnist_mlp_torch.json")
DEFAULT_CSV = Path("examples/results/mnist_comparison.csv")

TASK_DEFAULTS = {
    "mnist": {
        "orthon": DEFAULT_ORTHON,
        "torch": DEFAULT_TORCH,
        "csv": DEFAULT_CSV,
    },
    "cifar10": {
        "orthon": Path("examples/results/cifar10_cnn.json"),
        "torch": Path("examples/results/cifar10_cnn_torch.json"),
        "csv": Path("examples/results/cifar10_comparison.csv"),
    },
}


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {path}. Run the matching training script first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    epochs = data.get("epochs", [])
    if not epochs:
        raise ValueError(f"Metrics file has no epoch rows: {path}")
    return data


def loss_decrease(rows):
    return rows[0]["train_loss"] - rows[-1]["train_loss"]


def avg_seconds(rows):
    return sum(row["seconds"] for row in rows) / len(rows)


def comparison_rows(orthon_rows, torch_rows):
    count = min(len(orthon_rows), len(torch_rows))
    rows = []
    for i in range(count):
        d_row = orthon_rows[i]
        t_row = torch_rows[i]
        rows.append({
            "epoch": i + 1,
            "orthon_train_loss": d_row["train_loss"],
            "orthon_test_accuracy": d_row["test_accuracy"],
            "orthon_seconds": d_row["seconds"],
            "torch_train_loss": t_row["train_loss"],
            "torch_test_accuracy": t_row["test_accuracy"],
            "torch_seconds": t_row["seconds"],
        })
    return rows


def print_table(rows):
    header = (
        "epoch  "
        "orthon_loss  orthon_acc  orthon_sec  "
        "torch_loss  torch_acc  torch_sec"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['epoch']:>5}  "
            f"{row['orthon_train_loss']:>13.4f}  "
            f"{row['orthon_test_accuracy']:>12.4f}  "
            f"{row['orthon_seconds']:>12.2f}  "
            f"{row['torch_train_loss']:>10.4f}  "
            f"{row['torch_test_accuracy']:>9.4f}  "
            f"{row['torch_seconds']:>9.2f}"
        )


def print_summary(orthon_rows, torch_rows):
    d_final = orthon_rows[-1]
    t_final = torch_rows[-1]
    d_loss_drop = loss_decrease(orthon_rows)
    t_loss_drop = loss_decrease(torch_rows)
    d_avg = avg_seconds(orthon_rows)
    t_avg = avg_seconds(torch_rows)
    speedup = d_avg / t_avg if t_avg > 0 else float("inf")

    print("\nsummary")
    print("-------")
    print(f"final Orthon test accuracy: {d_final['test_accuracy']:.4f}")
    print(f"final PyTorch test accuracy:  {t_final['test_accuracy']:.4f}")
    print(f"Orthon train loss decrease: {d_loss_drop:.4f}")
    print(f"PyTorch train loss decrease:  {t_loss_drop:.4f}")
    print(f"Orthon avg epoch seconds:   {d_avg:.2f}")
    print(f"PyTorch avg epoch seconds:    {t_avg:.2f}")
    print(f"PyTorch speedup over Orthon: {speedup:.2f}x")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote CSV comparison to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Orthon and PyTorch metric JSON files.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_DEFAULTS.keys()),
        default="mnist",
        help="Use default metric paths for a known workload.",
    )
    parser.add_argument("--orthon", type=Path, default=None)
    parser.add_argument("--torch", type=Path, default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = TASK_DEFAULTS[args.task]
    orthon_path = args.orthon or defaults["orthon"]
    torch_path = args.torch or defaults["torch"]

    orthon = load_metrics(orthon_path)
    torch = load_metrics(torch_path)

    d_rows = orthon["epochs"]
    t_rows = torch["epochs"]
    rows = comparison_rows(d_rows, t_rows)

    if len(d_rows) != len(t_rows):
        print(
            "warning: metric files have different epoch counts; "
            f"comparing first {len(rows)} epoch(s)"
        )

    print_table(rows)
    print_summary(d_rows, t_rows)

    if args.csv is not None:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
