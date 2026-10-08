"""Small artifact smoke test. Run with: uv run python smoke_test.py"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import ANN_FEATURES, encode_text
from src.models import ANNChurnModel, CNNDigitModel, LSTMSentimentModel

ROOT = Path(__file__).resolve().parent
A = ROOT / "artifacts"

# ANN
scaler = json.loads((A / "ann_scaler.json").read_text())
ann = ANNChurnModel(len(ANN_FEATURES))
ann.load_state_dict(torch.load(A / "ann_model.pt", map_location="cpu", weights_only=True))
x = np.array([[18, 70, 2, 12, 3, 30, 1, 32]], dtype=np.float32)
x = ((x - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["scale"], dtype=np.float32)).astype(np.float32)
with torch.no_grad():
    ann_prob = torch.sigmoid(ann(torch.tensor(x))).item()
assert 0 <= ann_prob <= 1

# LSTM
cfg = json.loads((A / "lstm_config.json").read_text())
lstm = LSTMSentimentModel(len(cfg["vocab"]), cfg["embedding_dim"], cfg["hidden_dim"])
lstm.load_state_dict(torch.load(A / "lstm_model.pt", map_location="cpu", weights_only=True))
ids = encode_text("this movie was amazing and brilliant", cfg["vocab"], cfg["max_len"])
with torch.no_grad():
    sentiment_prob = torch.sigmoid(lstm(torch.tensor([ids], dtype=torch.long))).item()
assert 0 <= sentiment_prob <= 1

# CNN
cnn = CNNDigitModel()
cnn.load_state_dict(torch.load(A / "cnn_model.pt", map_location="cpu", weights_only=True))
img = Image.open(ROOT / "assets" / "digit_samples" / "digit_5.png").convert("L").resize((8, 8))
arr = np.asarray(img, dtype=np.float32) / 255.0
with torch.no_grad():
    probs = torch.softmax(cnn(torch.tensor(arr[None, None], dtype=torch.float32)), dim=1)[0]
assert probs.shape[0] == 10
assert abs(float(probs.sum()) - 1.0) < 1e-5

print("Smoke test passed")
print(f"ANN probability: {ann_prob:.3f}")
print(f"LSTM positive probability: {sentiment_prob:.3f}")
print(f"CNN prediction: {int(torch.argmax(probs))}")
