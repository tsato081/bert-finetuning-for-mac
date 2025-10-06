import os
import random
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# —— 設定部分 —— #

# ファイルパス
TRAIN_CSV = os.path.join("data", "train", "pick_train_all.csv")
TEST_CSV = os.path.join("data", "test", "Hawks_正解データマスター - ver 5.0 csv出力用.csv")

# ハイパーパラメータ等
MAX_LEN = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
SEED = 42

MODEL_NAME = "tohoku-nlp/bert-base-japanese-v3"

LABEL_MAP = {"Decline": 0, "Pick": 1}

# 再現性確保
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(SEED)

# —— デバイス設定 —— #

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    # fallback CPU
    device = torch.device("cpu")
print("Using device:", device)

# —— データセット定義 —— #

class BertBinaryDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # encoding 各キーは形 (1, max_len) → squeeze して (max_len,)
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# —— ヘルパー関数 —— #

def make_texts(df: pd.DataFrame, title_col: str, body_col: str) -> List[str]:
    # タイトルと本文を結合する例。必要なら適宜変える
    texts = df[title_col].fillna("") + " " + df[body_col].fillna("")
    return texts.tolist()


def encode_labels(labels: pd.Series, dataset_name: str) -> List[int]:
    mapped = labels.map(LABEL_MAP)
    if mapped.isna().any():
        unknown = labels[mapped.isna()].unique()
        raise ValueError(f"Unexpected labels {unknown} found in {dataset_name} dataset")
    return mapped.astype(int).tolist()

# —— データ読み込みと分割 —— #

train_df = pd.read_csv(TRAIN_CSV)
# pick/body が欠損している行は学習対象から外す
train_df = train_df.dropna(subset=["pick", "body"]).reset_index(drop=True)
train_df = train_df[train_df["pick"].isin(LABEL_MAP)].reset_index(drop=True)
# 層化分割：ラベル列は 'pick'
train_texts = make_texts(train_df, "title", "body")
train_labels = encode_labels(train_df["pick"], "train")

# 層化でバリデーション分割
train_texts_, val_texts, train_labels_, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.15,
    stratify=train_labels,
    random_state=SEED,
)

# テストデータ
test_df = pd.read_csv(TEST_CSV)
test_df = test_df.dropna(subset=["pick"]).reset_index(drop=True)
test_df = test_df[test_df["pick"].isin(LABEL_MAP)].reset_index(drop=True)
test_texts = make_texts(test_df, "title_original", "body_original")
test_labels = encode_labels(test_df["pick"], "test")  # 推論時は使うか使わないかはお好み

# —— トークナイザとモデル準備 —— #

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 2 クラス分類なので num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# —— データローダー準備 —— #

train_dataset = BertBinaryDataset(train_texts_, train_labels_, tokenizer, MAX_LEN)
val_dataset = BertBinaryDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = BertBinaryDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# —— 重み付きクロスエントロピーの重み計算 —— #

# クラスごとのサンプル数
from collections import Counter
cnt = Counter(train_labels_)
# 例えばラベル 0, 1 の出現数
n0 = cnt.get(0, 0)
n1 = cnt.get(1, 0)
print("Class counts in train:", cnt)

# 重みを反転頻度に比例させる（例）
weight_for_0 = (n0 + n1) / (2 * n0) if n0 > 0 else 1.0
weight_for_1 = (n0 + n1) / (2 * n1) if n1 > 0 else 1.0
class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float).to(device)
print("Class weights:", class_weights)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# —— オプティマイザとスケジューラ —— #

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# —— 訓練ループ —— #

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids は日本語 BERT だと不要なモデルも多いので要確認
        # もし tokenizer に token_type_ids を返すなら使う:
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        labels = batch["labels"].to(device)
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=True)
        logits = outputs.logits  # (batch_size, 2)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# —— メイン訓練処理 —— #

best_val_loss = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
    print(f"  Train loss = {train_loss:.4f}, acc = {train_acc:.4f}")
    val_loss, val_acc = eval_one_epoch(model, val_loader, loss_fn, device)
    print(f"  Val   loss = {val_loss:.4f}, acc = {val_acc:.4f}")

    # モデル保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = "./best_model.pt"
        torch.save(model.state_dict(), save_path)
        print("  Saved best model.")

# —— テスト推論例（モデル読み込み可） —— #

# モデル復元
model.load_state_dict(torch.load("./best_model.pt", map_location=device))
model.to(device)

model.eval()
preds_all = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=True)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.cpu().tolist())

# 必要なら結果を保存する、または test_df に付与して出力
test_df["pred_pick"] = preds_all
test_df.to_csv("test_with_preds.csv", index=False)
