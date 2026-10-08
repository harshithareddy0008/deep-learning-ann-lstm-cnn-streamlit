from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ANN_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_months",
    "satisfaction_score",
    "usage_hours_week",
    "late_payments",
    "age",
]


def generate_churn_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 73, n_samples),
            "monthly_charges": rng.uniform(20, 120, n_samples).round(2),
            "support_calls": rng.integers(0, 9, n_samples),
            "contract_months": rng.choice([1, 12, 24], n_samples, p=[0.48, 0.32, 0.20]),
            "satisfaction_score": rng.integers(1, 6, n_samples),
            "usage_hours_week": rng.uniform(2, 80, n_samples).round(1),
            "late_payments": rng.integers(0, 7, n_samples),
            "age": rng.integers(18, 76, n_samples),
        }
    )

    logit = (
        -1.30
        - 0.030 * df["tenure_months"]
        + 0.018 * (df["monthly_charges"] - 60)
        + 0.32 * df["support_calls"]
        - 0.055 * df["contract_months"]
        - 0.50 * (df["satisfaction_score"] - 3)
        - 0.009 * (df["usage_hours_week"] - 30)
        + 0.40 * df["late_payments"]
        + 0.006 * (df["age"] - 40)
        + rng.normal(0, 0.65, n_samples)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    df["churn"] = rng.binomial(1, probability)
    return df


POSITIVE_ADJ = [
    "amazing", "excellent", "wonderful", "fantastic", "brilliant", "beautiful",
    "engaging", "enjoyable", "impressive", "heartwarming", "funny", "powerful",
]
NEGATIVE_ADJ = [
    "awful", "terrible", "boring", "poor", "disappointing", "messy",
    "confusing", "forgettable", "weak", "annoying", "dull", "predictable",
]
POSITIVE_NOUN = ["story", "acting", "direction", "music", "ending", "cinematography", "dialogue"]
NEGATIVE_NOUN = ["story", "acting", "direction", "pacing", "ending", "dialogue", "script"]
POSITIVE_TEMPLATES = [
    "this movie was {a1} and the {n1} was {a2}",
    "i really loved this film the {n1} felt {a1}",
    "what a {a1} movie with {a2} {n1}",
    "the {n1} was {a1} and i enjoyed every minute",
    "a {a1} experience with {a2} performances",
    "i would happily watch this again because it was {a1}",
    "surprisingly {a1} film and the {n1} was {a2}",
]
NEGATIVE_TEMPLATES = [
    "this movie was {a1} and the {n1} was {a2}",
    "i really disliked this film the {n1} felt {a1}",
    "what a {a1} movie with {a2} {n1}",
    "the {n1} was {a1} and i wanted it to end",
    "a {a1} experience with {a2} performances",
    "i would not watch this again because it was {a1}",
    "sadly a {a1} film and the {n1} was {a2}",
]


def generate_sentiment_data(n_per_class: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in [1, 0]:
        adjs = POSITIVE_ADJ if label == 1 else NEGATIVE_ADJ
        nouns = POSITIVE_NOUN if label == 1 else NEGATIVE_NOUN
        templates = POSITIVE_TEMPLATES if label == 1 else NEGATIVE_TEMPLATES
        for _ in range(n_per_class):
            text = rng.choice(templates).format(
                a1=rng.choice(adjs),
                a2=rng.choice(adjs),
                n1=rng.choice(nouns),
            )
            if rng.random() < 0.25:
                text += rng.choice([
                    " overall", " for me", " from start to finish", " in my opinion", " honestly"
                ])
            rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def build_vocab(texts, max_vocab: int = 1000) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = 24) -> list[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)][:max_len]
    return ids + [vocab["<PAD>"]] * (max_len - len(ids))


def save_generated_datasets(data_dir: str | Path) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    churn_path = data_dir / "customer_churn.csv"
    sentiment_path = data_dir / "movie_reviews.csv"
    generate_churn_data().to_csv(churn_path, index=False)
    generate_sentiment_data().to_csv(sentiment_path, index=False)
    return churn_path, sentiment_path
