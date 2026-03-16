from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from ai_detection.config import TrainConfig
from ai_detection.data import PAD_TOKEN, TextDataModule
from ai_detection.model import LSTMClassifier
from ai_detection.utils import ensure_parent, save_json


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
        self.validation_labels: list[float] = []
        self.validation_predictions: list[float] = []
        self.validation_losses: list[float] = []
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

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, lengths, labels = batch
        logits = self(inputs, lengths)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_labels.clear()
        self.validation_predictions.clear()
        self.validation_losses.clear()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, lengths, labels = batch
        logits = self(inputs, lengths)
        loss = self.loss_fn(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        self.validation_losses.append(loss.detach().cpu().item() * labels.size(0))
        self.validation_labels.extend(labels.detach().cpu().tolist())
        self.validation_predictions.extend(predictions.detach().cpu().tolist())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_labels:
            return

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.validation_labels,
            self.validation_predictions,
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(self.validation_labels, self.validation_predictions)
        average_loss = sum(self.validation_losses) / len(self.validation_labels)
        self.log("test_loss", average_loss, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1, prog_bar=True)


def train(config: TrainConfig) -> dict:
    pl.seed_everything(config.random_seed, workers=True)
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
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.epochs,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        default_root_dir=str(config.artifacts_dir),
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)

    callback_metrics = trainer.callback_metrics
    best_metrics = {
        "loss": float(callback_metrics["test_loss"].item()),
        "accuracy": float(callback_metrics["test_accuracy"].item()),
        "precision": float(callback_metrics["test_precision"].item()),
        "recall": float(callback_metrics["test_recall"].item()),
        "f1": float(callback_metrics["test_f1"].item()),
    }
    history: list[dict[str, float]] = []
    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    if metrics_csv.exists():
        import pandas as pd

        history = pd.read_csv(metrics_csv).fillna("").to_dict(orient="records")

    ensure_parent(config.checkpoint_path)
    if checkpoint_callback.best_model_path:
        best_checkpoint = Path(checkpoint_callback.best_model_path)
        if best_checkpoint.resolve() != config.checkpoint_path.resolve():
            best_checkpoint.replace(config.checkpoint_path)

    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "device": str(trainer.strategy.root_device),
        "train_size": data_module.train_size,
        "test_size": data_module.test_size,
        "vocab_size": len(data_module.vocab),
        "checkpoint_path": str(config.checkpoint_path),
        "best_test_metrics": best_metrics,
        "history": history,
    }
    save_json(config.metrics_path, payload)
    return payload
