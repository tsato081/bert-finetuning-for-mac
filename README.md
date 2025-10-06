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

Çü¿Õ¡¤ë’KÕgån4@kMnWfO`UD
- `data/train/pick_train_all.csv`
- `data/test/Hawks_cãÇü¿Þ¹¿ü - ver 5.0 csvú›(.csv`

## Training

```bash
python train_base.py
```

âÇëo `best_model.pt` hWfÝXUŒˆ,Pœo `test_with_preds.csv` kú›UŒ~Y
