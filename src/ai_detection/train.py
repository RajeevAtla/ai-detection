from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from ai_detection.config import TrainConfig
from ai_detection.data import PAD_TOKEN, TextDataModule
from ai_detection.model import LSTMClassifier
from ai_detection.utils import save_json


def _binary_metrics(
    labels: list[float],
    predictions: list[float],
    loss_sum: float | None = None,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if loss_sum is not None:
        metrics["loss"] = float(loss_sum / len(labels))
    return metrics


def _load_history(metrics_csv: Path) -> list[dict[str, float]]:
    if not metrics_csv.exists():
        return []

    frame = pd.read_csv(metrics_csv)
    if "epoch" not in frame.columns:
        return []

    history: list[dict[str, float]] = []
    metrics = [
        "train_loss",
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    for epoch, group in frame.groupby("epoch", dropna=True):
        row: dict[str, float] = {"epoch": int(epoch) + 1}
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].dropna()
            if not values.empty:
                row[metric] = float(values.iloc[-1])
        history.append(row)
    history.sort(key=lambda item: item["epoch"])
    return history


def _trainer_hardware() -> tuple[str, int, str]:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "gpu", 1, f"cuda:{device_name}"
    return "cpu", 1, "cpu"


class LightningLSTMClassifier(pl.LightningModule):
    def __init__(self, config: TrainConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        self.config = config
        self.network = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pad_id=pad_id,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_labels: list[float] = []
        self.train_predictions: list[float] = []
        self.train_loss_sum = 0.0
        self.validation_labels: list[float] = []
        self.validation_predictions: list[float] = []
        self.validation_loss_sum = 0.0
        self.save_hyperparameters(
            {
                "learning_rate": config.learning_rate,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "max_vocab_size": config.max_vocab_size,
                "max_sequence_length": config.max_sequence_length,
            }
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.network(inputs, lengths)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def on_train_epoch_start(self) -> None:
        self.train_labels.clear()
        self.train_predictions.clear()
        self.train_loss_sum = 0.0

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, lengths, labels = batch
        logits = self(inputs, lengths)
        loss = self.loss_fn(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        self.train_loss_sum += loss.detach().cpu().item() * labels.size(0)
        self.train_labels.extend(labels.detach().cpu().tolist())
        self.train_predictions.extend(predictions.detach().cpu().tolist())
        return loss

    def on_train_epoch_end(self) -> None:
        if not self.train_labels:
            return
        metrics = _binary_metrics(self.train_labels, self.train_predictions, self.train_loss_sum)
        self.log("train_loss", metrics["loss"], prog_bar=True)
        self.log("train_accuracy", metrics["accuracy"])
        self.log("train_precision", metrics["precision"])
        self.log("train_recall", metrics["recall"])
        self.log("train_f1", metrics["f1"], prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.validation_labels.clear()
        self.validation_predictions.clear()
        self.validation_loss_sum = 0.0

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, lengths, labels = batch
        logits = self(inputs, lengths)
        loss = self.loss_fn(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        self.validation_loss_sum += loss.detach().cpu().item() * labels.size(0)
        self.validation_labels.extend(labels.detach().cpu().tolist())
        self.validation_predictions.extend(predictions.detach().cpu().tolist())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_labels:
            return
        metrics = _binary_metrics(self.validation_labels, self.validation_predictions, self.validation_loss_sum)
        self.log("test_loss", metrics["loss"], prog_bar=True)
        self.log("test_accuracy", metrics["accuracy"], prog_bar=True)
        self.log("test_precision", metrics["precision"])
        self.log("test_recall", metrics["recall"])
        self.log("test_f1", metrics["f1"], prog_bar=True)


def train(config: TrainConfig) -> dict:
    pl.seed_everything(config.random_seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    data_module = TextDataModule(config)
    data_module.setup("fit")
    save_json(config.vocab_path, data_module.vocab)

    model = LightningLSTMClassifier(
        config=config,
        vocab_size=len(data_module.vocab),
        pad_id=data_module.vocab[PAD_TOKEN],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_path.parent,
        filename=config.checkpoint_path.stem,
        monitor="test_f1",
        mode="max",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=str(config.artifacts_dir), name="lightning_logs")
    accelerator, devices, device_label = _trainer_hardware()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.epochs,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        default_root_dir=str(config.artifacts_dir),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data_module)

    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    history = _load_history(metrics_csv)
    best_metrics = None
    if history:
        best_epoch = max(history, key=lambda item: item.get("test_f1", float("-inf")))
        best_metrics = {
            "loss": best_epoch["test_loss"],
            "accuracy": best_epoch["test_accuracy"],
            "precision": best_epoch["test_precision"],
            "recall": best_epoch["test_recall"],
            "f1": best_epoch["test_f1"],
            "epoch": best_epoch["epoch"],
        }
    else:
        callback_metrics = trainer.callback_metrics
        best_metrics = {
            "loss": float(callback_metrics["test_loss"].item()),
            "accuracy": float(callback_metrics["test_accuracy"].item()),
            "precision": float(callback_metrics["test_precision"].item()),
            "recall": float(callback_metrics["test_recall"].item()),
            "f1": float(callback_metrics["test_f1"].item()),
            "epoch": config.epochs,
        }

    checkpoint_path = checkpoint_callback.best_model_path or str(config.checkpoint_path)
    best_metrics = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in best_metrics.items()
    }

    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "device": device_label,
        "train_size": data_module.train_size,
        "test_size": data_module.test_size,
        "vocab_size": len(data_module.vocab),
        "checkpoint_path": checkpoint_path,
        "best_test_metrics": best_metrics,
        "history": history,
    }
    save_json(config.metrics_path, payload)
    return payload


def predict_from_checkpoint(config: TrainConfig, checkpoint_path: str | Path) -> dict:
    data_module = TextDataModule(config)
    data_module.setup("fit")
    checkpoint = Path(checkpoint_path)
    model = LightningLSTMClassifier.load_from_checkpoint(
        str(checkpoint),
        config=config,
        vocab_size=len(data_module.vocab),
        pad_id=data_module.vocab[PAD_TOKEN],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = model.to(device)
    model.eval()

    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0

    with torch.no_grad():
        for inputs, lengths, batch_labels in data_module.val_dataloader():
            inputs = inputs.to(device, non_blocking=torch.cuda.is_available())
            lengths = lengths.to(device, non_blocking=torch.cuda.is_available())
            batch_labels = batch_labels.to(device, non_blocking=torch.cuda.is_available())
            logits = model(inputs, lengths)
            loss = loss_fn(logits, batch_labels)
            batch_probabilities = torch.sigmoid(logits)
            batch_predictions = (batch_probabilities >= 0.5).int()

            loss_sum += loss.detach().cpu().item() * batch_labels.size(0)
            labels.extend(batch_labels.detach().cpu().int().tolist())
            predictions.extend(batch_predictions.detach().cpu().tolist())
            probabilities.extend(batch_probabilities.detach().cpu().tolist())

    metrics = _binary_metrics([float(label) for label in labels], [float(pred) for pred in predictions], loss_sum)
    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }
