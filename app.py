from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.data import ANN_FEATURES, encode_text
from src.models import ANNChurnModel, CNNDigitModel, LSTMSentimentModel

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
SAMPLE_DIR = ROOT / "assets" / "digit_samples"

st.set_page_config(
    page_title="Deep Learning Mini Suite",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_models():
    scaler_cfg = json.loads((ARTIFACT_DIR / "ann_scaler.json").read_text())
    ann = ANNChurnModel(input_size=len(ANN_FEATURES))
    ann.load_state_dict(torch.load(ARTIFACT_DIR / "ann_model.pt", map_location="cpu", weights_only=True))
    ann.eval()

    lstm_cfg = json.loads((ARTIFACT_DIR / "lstm_config.json").read_text())
    lstm = LSTMSentimentModel(
        vocab_size=len(lstm_cfg["vocab"]),
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
    )
    lstm.load_state_dict(torch.load(ARTIFACT_DIR / "lstm_model.pt", map_location="cpu", weights_only=True))
    lstm.eval()

    cnn = CNNDigitModel()
    cnn.load_state_dict(torch.load(ARTIFACT_DIR / "cnn_model.pt", map_location="cpu", weights_only=True))
    cnn.eval()
    return scaler_cfg, ann, lstm_cfg, lstm, cnn


@st.cache_data
def load_metrics():
    return {
        "ANN": json.loads((ARTIFACT_DIR / "ann_metrics.json").read_text()),
        "LSTM": json.loads((ARTIFACT_DIR / "lstm_metrics.json").read_text()),
        "CNN": json.loads((ARTIFACT_DIR / "cnn_metrics.json").read_text()),
    }


def ann_predict(values, scaler_cfg, model):
    x = np.asarray([values], dtype=np.float32)
    mean = np.asarray(scaler_cfg["mean"], dtype=np.float32)
    scale = np.asarray(scaler_cfg["scale"], dtype=np.float32)
    x = ((x - mean) / scale).astype(np.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x))).item()
    return float(prob)


def sentiment_predict(text, cfg, model):
    encoded = encode_text(text, cfg["vocab"], cfg["max_len"])
    x = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return float(prob)


def preprocess_digit(image: Image.Image):
    img = ImageOps.grayscale(image)
    arr = np.asarray(img, dtype=np.float32)
    # sklearn digits uses a dark background and bright digit. Auto-invert common white-background uploads.
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr[None, None, :, :], dtype=torch.float32)
    return img, tensor


def cnn_predict(image, model):
    small, x = preprocess_digit(image)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return small, probs


scaler, ann_model, lstm_cfg, lstm_model, cnn_model = load_models()
metrics = load_metrics()

st.title("🧠 Deep Learning Mini Suite")
st.caption("One beginner project demonstrating ANN, RNN-LSTM, CNN, model inference, Streamlit UI and deployment.")

with st.sidebar:
    st.header("Project Guide")
    st.write("**ANN:** customer churn classification")
    st.write("**LSTM:** movie-review sentiment")
    st.write("**CNN:** handwritten digit recognition")
    st.divider()
    st.info("All models run locally inside this app. No API key is required.")

overview, ann_tab, lstm_tab, cnn_tab, model_tab = st.tabs(
    ["Overview", "ANN • Churn", "LSTM • Sentiment", "CNN • Digits", "Model Card"]
)

with overview:
    st.subheader("What this project teaches")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ANN")
        st.write("Structured numerical inputs → scaling → fully connected layers → binary probability.")
    with c2:
        st.markdown("### RNN-LSTM")
        st.write("Text → tokenization → word IDs → embeddings → LSTM → sentiment probability.")
    with c3:
        st.markdown("### CNN")
        st.write("Image → grayscale/resize → convolutions → pooling → 10-class digit prediction.")
    st.success("Tip: retrain everything with `uv run python train_all.py`, then start the UI with `uv run streamlit run app.py`.")

with ann_tab:
    st.subheader("ANN: Customer Churn Predictor")
    st.caption("Educational synthetic dataset — useful for learning, not for real business decisions.")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 18)
        monthly = st.slider("Monthly charges", 20.0, 120.0, 70.0, 1.0)
        support = st.slider("Support calls", 0, 8, 2)
        contract = st.select_slider("Contract length (months)", options=[1, 12, 24], value=12)
    with col2:
        satisfaction = st.slider("Satisfaction score", 1, 5, 3)
        usage = st.slider("Usage hours / week", 2.0, 80.0, 30.0, 1.0)
        late = st.slider("Late payments", 0, 6, 1)
        age = st.slider("Customer age", 18, 75, 32)

    values = [tenure, monthly, support, contract, satisfaction, usage, late, age]
    if st.button("Predict churn", type="primary"):
        prob = ann_predict(values, scaler, ann_model)
        st.metric("Churn probability", f"{prob:.1%}")
        if prob >= 0.5:
            st.warning("Model output: higher churn risk")
        else:
            st.success("Model output: lower churn risk")
        st.progress(min(max(prob, 0.0), 1.0))

with lstm_tab:
    st.subheader("RNN-LSTM: Movie Review Sentiment")
    example = st.selectbox(
        "Try an example",
        [
            "Custom review",
            "This movie was amazing and the acting was brilliant",
            "I really disliked this film, the story felt boring and weak",
            "Wonderful direction and a beautiful ending, I loved it",
        ],
    )
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""
    if st.button("Load selected example"):
        st.session_state["review_input"] = "" if example == "Custom review" else example
    review = st.text_area(
        "Review text",
        key="review_input",
        height=130,
        placeholder="Type a short movie review...",
    )
    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.error("Please enter a review first.")
        else:
            pos_prob = sentiment_predict(review, lstm_cfg, lstm_model)
            label = "Positive 😊" if pos_prob >= 0.5 else "Negative 🙁"
            st.metric("Prediction", label)
            st.write(f"Positive probability: **{pos_prob:.1%}**")
            st.progress(min(max(pos_prob, 0.0), 1.0))
            st.caption("This LSTM is trained on a small generated teaching dataset, so unusual real-world wording can be misclassified.")

with cnn_tab:
    st.subheader("CNN: Handwritten Digit Recognition")
    source = st.radio("Image source", ["Built-in sample", "Upload image"], horizontal=True)
    image = None
    if source == "Built-in sample":
        digit = st.selectbox("Choose a sample digit", list(range(10)), index=5)
        image = Image.open(SAMPLE_DIR / f"digit_{digit}.png")
    else:
        uploaded = st.file_uploader("Upload a PNG/JPG containing one digit", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image = Image.open(uploaded)

    if image is not None:
        left, right = st.columns([1, 2])
        with left:
            st.image(image, caption="Input image", width=220)
        with right:
            if st.button("Recognize digit", type="primary"):
                small, probs = cnn_predict(image, cnn_model)
                pred = int(np.argmax(probs))
                st.metric("Predicted digit", str(pred))
                st.write(f"Confidence: **{probs[pred]:.1%}**")
                prob_df = pd.DataFrame({"digit": [str(i) for i in range(10)], "probability": probs}).set_index("digit")
                st.bar_chart(prob_df)
                st.caption("For best results, use one centered dark digit on a light background, or a bright digit on a dark background.")

with model_tab:
    st.subheader("Training results and architecture")
    a, b, c = st.columns(3)
    a.metric("ANN test accuracy", f"{metrics['ANN']['accuracy']:.1%}")
    b.metric("LSTM test accuracy", f"{metrics['LSTM']['accuracy']:.1%}")
    c.metric("CNN test accuracy", f"{metrics['CNN']['accuracy']:.1%}")

    st.markdown(
        """
**ANN:** 8 inputs → Dense(32) → ReLU → Dropout → Dense(16) → ReLU → Dense(1)  
**LSTM:** Token IDs → Embedding(32) → LSTM(64) → Dropout → Dense(1)  
**CNN:** Conv(16) → Pool → Conv(32) → Pool → Dense(64) → Dense(10)
        """
    )
    st.json(metrics)
