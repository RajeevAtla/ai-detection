# ai-detection
differentiating between ai and human written text

## data source
https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset

## training scaffold
This repo includes a minimal PyTorch Lightning LSTM training scaffold for classifying `AI` vs `Human` text from [`data/data.csv`](data/data.csv).

## setup with uv
```powershell
uv sync
```

## run study
```powershell
uv run python scripts/train_lstm.py
```

Optional flags:
```powershell
uv run python scripts/train_lstm.py --batch-size 64 --sequence-lengths 64 128 256 --max-epochs 10
```

## outputs
The study writes:

- per-sequence-length Lightning runs under `artifacts/study/seq_len_*`
- `artifacts/study/study_metrics.csv`
- `artifacts/study/study_summary.json`
- `artifacts/study/plots/train_loss.png`
- `artifacts/study/plots/test_loss.png`
- `artifacts/study/plots/train_f1.png`
- `artifacts/study/plots/test_f1.png`
- `artifacts/study/plots/test_confusion_matrix.png`

