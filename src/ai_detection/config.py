from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    data_path: Path = Path("data/data.csv")
    artifacts_dir: Path = Path("artifacts")
    test_size: float = 0.2
    random_seed: int = 42
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    max_vocab_size: int = 20_000
    max_sequence_length: int = 256
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    lowercase: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.artifacts_dir = Path(self.artifacts_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.artifacts_dir / "checkpoints" / "best.ckpt"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def split_dir(self) -> Path:
        return self.artifacts_dir / "splits"

    @property
    def vocab_path(self) -> Path:
        return self.artifacts_dir / "vocab.json"
