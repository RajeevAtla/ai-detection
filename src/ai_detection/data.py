from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import lightning.pytorch as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ai_detection.config import TrainConfig

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
LABEL_TO_ID = {"Human": 0, "AI": 1}
TOKEN_PATTERN = re.compile(r"\S+")


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    normalized = " ".join(text.split())
    if lowercase:
        normalized = normalized.lower()
    return TOKEN_PATTERN.findall(normalized)


def load_dataframe(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    unnamed_columns = [column for column in frame.columns if str(column).startswith("Unnamed") or column == ""]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)
    expected_columns = {"Text", "Author"}
    missing_columns = expected_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_columns)}")
    frame = frame.dropna(subset=["Text", "Author"]).copy()
    frame["Text"] = frame["Text"].astype(str)
    frame["Author"] = frame["Author"].astype(str)
    frame["label"] = frame["Author"].map(LABEL_TO_ID)
    if frame["label"].isna().any():
        unknown_labels = sorted(frame.loc[frame["label"].isna(), "Author"].unique())
        raise ValueError(f"Unknown labels found in Author column: {unknown_labels}")
    frame["label"] = frame["label"].astype(int)
    return frame


def stratified_split(frame: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, test_frame = train_test_split(
        frame,
        test_size=config.test_size,
        stratify=frame["label"],
        random_state=config.random_seed,
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def save_split_artifacts(train_frame: pd.DataFrame, test_frame: pd.DataFrame, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(split_dir / "train.csv", index=False)
    test_frame.to_csv(split_dir / "test.csv", index=False)


def build_vocab(texts: list[str], config: TrainConfig) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text, lowercase=config.lowercase))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max(config.max_vocab_size - len(vocab), 0)):
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], config: TrainConfig) -> list[int]:
    tokens = tokenize(text, lowercase=config.lowercase)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[: config.max_sequence_length]]
    return token_ids or [vocab[UNK_TOKEN]]


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], config: TrainConfig) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.config = config

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = encode_text(self.texts[index], self.vocab, self.config)
        sequence = torch.tensor(token_ids, dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return sequence, label


def make_collate_fn(pad_id: int):
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        label_tensor = torch.stack(labels)
        return padded, lengths, label_tensor

    return collate_fn


class TextDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab: dict[str, int] = {}
        self.train_size = 0
        self.test_size = 0
        self._train_dataset: TextDataset | None = None
        self._test_dataset: TextDataset | None = None
        self._collate_fn = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None and self._test_dataset is not None:
            return

        frame = load_dataframe(self.config.data_path)
        train_frame, test_frame = stratified_split(frame, self.config)
        save_split_artifacts(train_frame, test_frame, self.config.split_dir)

        self.vocab = build_vocab(train_frame["Text"].tolist(), self.config)
        self._collate_fn = make_collate_fn(self.vocab[PAD_TOKEN])
        self._train_dataset = TextDataset(
            train_frame["Text"].tolist(),
            train_frame["label"].tolist(),
            self.vocab,
            self.config,
        )
        self._test_dataset = TextDataset(
            test_frame["Text"].tolist(),
            test_frame["label"].tolist(),
            self.vocab,
            self.config,
        )
        self.train_size = len(self._train_dataset)
        self.test_size = len(self._test_dataset)

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None or self._collate_fn is None:
            raise RuntimeError("Data module must be set up before requesting dataloaders.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self._test_dataset is None or self._collate_fn is None:
            raise RuntimeError("Data module must be set up before requesting dataloaders.")
        return DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
