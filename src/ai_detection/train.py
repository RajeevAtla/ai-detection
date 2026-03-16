from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from ai_detection.config import TrainConfig
from ai_detection.data import DataBundle, PAD_TOKEN, build_dataloaders
from ai_detection.model import LSTMClassifier
from ai_detection.utils import ensure_parent, save_json, set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: LSTMClassifier,
    data_loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[float] = []
    all_predictions: list[float] = []

    with torch.no_grad():
        for inputs, lengths, labels in data_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(inputs, lengths)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "loss": total_loss / len(data_loader.dataset),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(config: TrainConfig) -> dict:
    set_seed(config.random_seed)
    bundle: DataBundle = build_dataloaders(config)
    device = get_device()

    model = LSTMClassifier(
        vocab_size=len(bundle.vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_id=bundle.vocab[PAD_TOKEN],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    training_started_at = perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, lengths, labels in bundle.train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / bundle.train_size
        test_metrics = evaluate(model, bundle.test_loader, loss_fn, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_accuracy={test_metrics['accuracy']:.4f} "
            f"test_f1={test_metrics['f1']:.4f}"
        )

        is_better = best_metrics is None or test_metrics["f1"] > best_metrics["f1"]
        if is_better:
            best_metrics = test_metrics
            ensure_parent(config.checkpoint_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": bundle.vocab,
                    "config": asdict(config),
                    "metrics": best_metrics,
                },
                config.checkpoint_path,
            )

    duration_seconds = perf_counter() - training_started_at
    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "device": str(device),
        "train_size": bundle.train_size,
        "test_size": bundle.test_size,
        "vocab_size": len(bundle.vocab),
        "best_test_metrics": best_metrics,
        "history": history,
        "duration_seconds": duration_seconds,
    }
    save_json(config.metrics_path, payload)
    return payload
