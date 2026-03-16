from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ai_detection.config import TrainConfig
from ai_detection.train import predict_from_checkpoint, train


def parse_sequence_lengths(values: list[int] | None) -> list[int]:
    if values:
        return sorted(dict.fromkeys(values))
    return [64, 128, 256]


def build_config(args: argparse.Namespace, sequence_length: int) -> TrainConfig:
    run_artifacts_dir = Path(args.artifacts_dir) / f"seq_len_{sequence_length}"
    return TrainConfig(
        data_path=args.data_path,
        artifacts_dir=run_artifacts_dir,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_vocab_size=args.max_vocab_size,
        max_sequence_length=sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        random_seed=args.seed,
        lowercase=not args.preserve_case,
        num_workers=args.num_workers,
    )


def make_plot(study_frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for sequence_length, group in study_frame.groupby("sequence_length"):
        ordered = group.sort_values("epoch")
        plt.plot(ordered["epoch"], ordered[metric], marker="o", label=f"seq={sequence_length}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} by epoch and sequence length")
    plt.xticks(sorted(study_frame["epoch"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Human", "AI"])
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LSTM study over epochs and sequence lengths.")
    parser.add_argument("--data-path", default="data/data.csv")
    parser.add_argument("--artifacts-dir", default="artifacts/study")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=20_000)
    parser.add_argument("--sequence-lengths", nargs="+", type=int)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preserve-case", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_lengths = parse_sequence_lengths(args.sequence_lengths)
    study_root = Path(args.artifacts_dir)
    study_root.mkdir(parents=True, exist_ok=True)
    print(
        f"torch_version={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )

    run_summaries: list[dict] = []
    study_records: list[dict] = []
    best_run: dict | None = None

    for sequence_length in sequence_lengths:
        config = build_config(args, sequence_length)
        print(f"running sequence_length={sequence_length} epochs=1..{args.max_epochs}")
        result = train(config)
        run_summary = {
            "sequence_length": sequence_length,
            "config": result["config"],
            "device": result["device"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "vocab_size": result["vocab_size"],
            "checkpoint_path": result["checkpoint_path"],
            "best_test_metrics": result["best_test_metrics"],
        }
        run_summaries.append(run_summary)
        if best_run is None or run_summary["best_test_metrics"]["f1"] > best_run["best_test_metrics"]["f1"]:
            best_run = run_summary
        for epoch_metrics in result["history"]:
            if {"train_loss", "test_loss", "train_f1", "test_f1"}.issubset(epoch_metrics):
                study_records.append(
                    {
                        "sequence_length": sequence_length,
                        "epoch": int(epoch_metrics["epoch"]),
                        "train_loss": float(epoch_metrics["train_loss"]),
                        "test_loss": float(epoch_metrics["test_loss"]),
                        "train_f1": float(epoch_metrics["train_f1"]),
                        "test_f1": float(epoch_metrics["test_f1"]),
                    }
                )

    if not study_records:
        raise RuntimeError("No study metrics were collected from the training runs.")

    study_frame = pd.DataFrame(study_records).sort_values(["sequence_length", "epoch"])
    study_frame.to_csv(study_root / "study_metrics.csv", index=False)
    make_plot(study_frame, "train_loss", study_root / "plots" / "train_loss.png")
    make_plot(study_frame, "test_loss", study_root / "plots" / "test_loss.png")
    make_plot(study_frame, "train_f1", study_root / "plots" / "train_f1.png")
    make_plot(study_frame, "test_f1", study_root / "plots" / "test_f1.png")

    if best_run is None:
        raise RuntimeError("Unable to determine the best study run.")

    best_config = build_config(args, int(best_run["sequence_length"]))
    prediction_result = predict_from_checkpoint(best_config, best_run["checkpoint_path"])
    save_confusion_matrix(
        prediction_result["labels"],
        prediction_result["predictions"],
        study_root / "plots" / "test_confusion_matrix.png",
    )

    summary = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sequence_lengths": sequence_lengths,
        "max_epochs": args.max_epochs,
        "best_run": best_run,
        "best_run_test_metrics": prediction_result["metrics"],
        "plots": {
            "train_loss": str(study_root / "plots" / "train_loss.png"),
            "test_loss": str(study_root / "plots" / "test_loss.png"),
            "train_f1": str(study_root / "plots" / "train_f1.png"),
            "test_f1": str(study_root / "plots" / "test_f1.png"),
            "test_confusion_matrix": str(study_root / "plots" / "test_confusion_matrix.png"),
        },
        "study_metrics_csv": str(study_root / "study_metrics.csv"),
    }
    pd.Series(summary, dtype=object).to_json(study_root / "study_summary.json", indent=2)
    print(
        f"completed best_sequence_length={best_run['sequence_length']} "
        f"best_epoch={best_run['best_test_metrics']['epoch']} "
        f"best_test_accuracy={prediction_result['metrics']['accuracy']:.4f} "
        f"best_test_f1={prediction_result['metrics']['f1']:.4f} "
        f"study_metrics={study_root / 'study_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
