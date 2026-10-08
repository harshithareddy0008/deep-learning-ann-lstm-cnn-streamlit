from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import ANN_FEATURES, build_vocab, encode_text, save_generated_datasets
from src.models import ANNChurnModel, CNNDigitModel, LSTMSentimentModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
SAMPLE_DIR = ROOT / "assets" / "digit_samples"
SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def save_state_dict(model: nn.Module, path: Path):
    # Legacy tensor serialization keeps the file highly portable across PyTorch 2.x versions.
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)


def train_ann(churn_path: Path):
    print("\n[1/3] Training ANN churn model...")
    df = pd.read_csv(churn_path)
    X = df[ANN_FEATURES].to_numpy(dtype=np.float32)
    y = df["churn"].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = ANNChurnModel(input_size=len(ANN_FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=0.0015)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(25):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d} loss={total/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "test_samples": int(len(y_test)),
        "positive_rate": round(float(y.mean()), 4),
    }
    save_state_dict(model, ARTIFACT_DIR / "ann_model.pt")
    (ARTIFACT_DIR / "ann_scaler.json").write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, indent=2)
    )
    (ARTIFACT_DIR / "ann_features.json").write_text(json.dumps(ANN_FEATURES, indent=2))
    (ARTIFACT_DIR / "ann_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("  ANN metrics:", metrics)


def train_lstm(sentiment_path: Path):
    print("\n[2/3] Training LSTM sentiment model...")
    df = pd.read_csv(sentiment_path)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["label"]
    )
    vocab = build_vocab(train_df["text"], max_vocab=800)
    max_len = 24
    X_train = np.asarray([encode_text(t, vocab, max_len) for t in train_df["text"]], dtype=np.int64)
    X_test = np.asarray([encode_text(t, vocab, max_len) for t in test_df["text"]], dtype=np.int64)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = LSTMSentimentModel(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(10):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"  epoch {epoch+1:02d} loss={total/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "test_samples": int(len(y_test)),
        "vocab_size": len(vocab),
        "max_len": max_len,
    }
    save_state_dict(model, ARTIFACT_DIR / "lstm_model.pt")
    (ARTIFACT_DIR / "lstm_config.json").write_text(
        json.dumps({"vocab": vocab, "max_len": max_len, "embedding_dim": 32, "hidden_dim": 64}, indent=2)
    )
    (ARTIFACT_DIR / "lstm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("  LSTM metrics:", metrics)


def train_cnn():
    print("\n[3/3] Training CNN digit model...")
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None, :, :]
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = CNNDigitModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.0015)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(25):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:02d} loss={total/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        pred = logits.argmax(dim=1).numpy()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "test_samples": int(len(y_test)),
        "classes": 10,
    }
    save_state_dict(model, ARTIFACT_DIR / "cnn_model.pt")
    (ARTIFACT_DIR / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    # Save native 8x8 examples that this trained model classifies correctly.
    with torch.no_grad():
        all_pred = model(torch.tensor(X)).argmax(dim=1).numpy()
    for digit in range(10):
        candidates = np.where((y == digit) & (all_pred == digit))[0]
        idx = int(candidates[0] if len(candidates) else np.where(y == digit)[0][0])
        arr = (digits.images[idx] / 16.0 * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(SAMPLE_DIR / f"digit_{digit}.png")
    print("  CNN metrics:", metrics)


def main():
    seed_everything()
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    churn_path, sentiment_path = save_generated_datasets(DATA_DIR)
    train_ann(churn_path)
    train_lstm(sentiment_path)
    train_cnn()
    print("\nDone. Models, preprocessors, metrics and datasets are ready.")


if __name__ == "__main__":
    main()
