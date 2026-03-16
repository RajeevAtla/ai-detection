from __future__ import annotations

import argparse

from ai_detection.config import TrainConfig
from ai_detection.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM classifier on the AI detection dataset.")
    parser.add_argument("--data-path", default="data/data.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=20_000)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preserve-case", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_vocab_size=args.max_vocab_size,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        random_seed=args.seed,
        lowercase=not args.preserve_case,
    )
    result = train(config)
    print(
        f"completed train_size={result['train_size']} "
        f"test_size={result['test_size']} "
        f"vocab_size={result['vocab_size']} "
        f"best_test_accuracy={result['best_test_metrics']['accuracy']:.4f} "
        f"best_test_f1={result['best_test_metrics']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
