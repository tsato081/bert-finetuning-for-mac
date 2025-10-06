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

���ա��K�g�n4@kMnWfO`UD
- `data/train/pick_train_all.csv`
- `data/test/Hawks_c����޹�� - ver 5.0 csv��(.csv`

## Training

```bash
python train_base.py
```

���o `best_model.pt` hWf�XU��,P�o `test_with_preds.csv` k��U�~Y
