# Fine-tuning for Mac

BERT-based binary classification model training on Mac with MPS support.

## Setup

### 1. Install uv (Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Prepare data directories
```bash
mkdir -p data/train data/test
```

Place your data files manually:
- `data/train/pick_train_all.csv`
- `data/test/Hawks_正解データマスター - ver 5.0 csv出力用.csv`

## Training

```bash
python train_base.py
```

Model will be saved as `best_model.pt` and predictions will be output to `test_with_preds.csv`.
