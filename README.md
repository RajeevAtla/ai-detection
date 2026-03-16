# ai-detection
differentiating between ai and human written text

## data source
https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset

## training scaffold
This repo includes a minimal PyTorch LSTM training scaffold for classifying `AI` vs `Human` text from [`data/data.csv`](data/data.csv).

## setup with uv
```powershell
uv sync
```

## train
```powershell
uv run python scripts/train_lstm.py
```

Optional flags:
```powershell
uv run python scripts/train_lstm.py --epochs 3 --batch-size 64 --max-sequence-length 128
```

## outputs
Training writes:

- `artifacts/checkpoints/best.pt`
- `artifacts/metrics.json`
- `artifacts/splits/train.csv`
- `artifacts/splits/test.csv`

