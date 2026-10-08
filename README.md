# Deep Learning Beginner Suite — ANN + LSTM + CNN + Streamlit + Render

A single beginner-friendly, portfolio-ready project that demonstrates three core deep-learning architectures and serves them in one Streamlit web application.

## What is inside?

| Model | Demo task | Input | Output |
|---|---|---|---|
| ANN | Customer churn prediction | 8 numeric customer features | Churn probability |
| RNN-LSTM | Movie-review sentiment | Short review text | Positive/negative probability |
| CNN | Handwritten digit recognition | Digit image | Digit 0–9 + confidence |

The ANN and LSTM datasets are generated locally by `train_all.py`. The CNN uses scikit-learn's built-in 8×8 digits dataset, so you do **not** need to download a separate dataset.

## Project structure

```text
deep_learning_beginner_suite/
├── app.py
├── train_all.py
├── train_ann.py
├── train_lstm.py
├── train_cnn.py
├── smoke_test.py
├── pyproject.toml
├── .python-version
├── Dockerfile
├── render.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── models.py
├── data/
│   ├── customer_churn.csv
│   └── movie_reviews.csv
├── artifacts/
│   ├── ann_model.pt
│   ├── ann_scaler.json
│   ├── ann_features.json
│   ├── ann_metrics.json
│   ├── lstm_model.pt
│   ├── lstm_config.json
│   ├── lstm_metrics.json
│   ├── cnn_model.pt
│   └── cnn_metrics.json
└── assets/digit_samples/
    └── digit_0.png ... digit_9.png
```

## 1. Prerequisite: install `uv`

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check:

```bash
uv --version
```

## 2. Use Python 3.10.6 exactly

The project already contains `.python-version` with `3.10.6`, and `pyproject.toml` pins versions tested for the Python 3.10 line.

```bash
uv python install 3.10.6
uv sync
```

`uv sync` creates `.venv`, resolves the dependencies in `pyproject.toml`, installs them, and creates a real `uv.lock` on your machine.

After the first successful `uv sync`, commit the generated `uv.lock` to GitHub for maximum reproducibility.

## 3. Run the ready-made app

Pretrained artifacts are included, so after `uv sync` you can start immediately:

```bash
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 4. Train all three models yourself

```bash
uv run python train_all.py
```

This command:

1. creates `data/customer_churn.csv`;
2. creates `data/movie_reviews.csv`;
3. trains the ANN and saves its scaler statistics/model/metrics;
4. trains the LSTM and saves its vocabulary/model/metrics;
5. trains the CNN on `sklearn.datasets.load_digits()`;
6. saves 10 example digit images;
7. overwrites the files in `artifacts/` with your newly trained models.

Then restart Streamlit:

```bash
uv run streamlit run app.py
```

## 5. ANN workflow you should understand

**Problem:** binary classification — will a synthetic customer churn?

**Features:** tenure, monthly charges, support calls, contract length, satisfaction, weekly usage, late payments and age.

Pipeline:

```text
CSV → train/test split → StandardScaler → ANN → sigmoid → churn probability
```

Architecture:

```text
8 inputs → Linear(32) → ReLU → Dropout → Linear(16) → ReLU → Linear(1)
```

Loss: `BCEWithLogitsLoss`  
Optimizer: Adam

## 6. RNN-LSTM workflow you should understand

**Problem:** binary text classification — positive or negative movie review.

Pipeline:

```text
Review → lowercase/tokenize → vocabulary IDs → pad to fixed length
       → Embedding → LSTM → Linear(1) → sigmoid
```

The included teaching dataset is generated from positive/negative review templates. This makes the project fully reproducible, but it is intentionally small and is not a production NLP model.

## 7. CNN workflow you should understand

**Problem:** classify handwritten digits 0–9.

Pipeline:

```text
8×8 grayscale image → Conv2D → ReLU → MaxPool
                    → Conv2D → ReLU → MaxPool
                    → Flatten → Dense → 10 logits → softmax
```

The app can use included examples or an uploaded PNG/JPG.

## 8. Docker test before Render

```bash
docker build -t deep-learning-mini-suite .
docker run --rm -p 10000:10000 -e PORT=10000 deep-learning-mini-suite
```

Then open `http://localhost:10000`.

## 9. Deploy on Render

This project uses Docker so the production environment is pinned to **Python 3.10.6**.

1. Create a GitHub repository.
2. Push this entire project, including `artifacts/`, to the repository.
3. Sign in to Render and create a **Web Service** from the repository.
4. Select the **Docker** runtime. Render will build the root `Dockerfile`.
5. Deploy. The Docker command starts Streamlit on Render's `$PORT`.

You can also use the included `render.yaml` as a Render Blueprint.

### Git commands

```bash
git init
git add .
git commit -m "Initial deep learning mini suite"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## 10. Interview explanation — 60 seconds

> I built an end-to-end deep learning mini suite to demonstrate three neural-network architectures. For structured customer data I used a feed-forward ANN with standardized numeric features. For text sentiment I built a tokenizer and vocabulary, converted reviews to padded integer sequences, and passed embeddings through an LSTM. For image classification I trained a CNN on handwritten digits. I saved the trained PyTorch model weights and preprocessing objects, loaded them in a Streamlit interface, containerized the application with Python 3.10.6 and `uv`, and prepared it for Render deployment. The project helped me understand the difference between dense networks, sequence models, and convolutional networks as well as the complete training-to-deployment lifecycle.

## 11. Resume bullet

**Deep Learning Mini Suite | PyTorch, ANN, LSTM, CNN, Streamlit, Docker, Render**  
Built and deployed an end-to-end deep learning application combining an ANN for tabular churn prediction, an LSTM for text sentiment classification, and a CNN for handwritten-digit recognition; implemented preprocessing, model training, saved-model inference, interactive Streamlit UI, reproducible `uv` environment management, and Docker-based Render deployment.

## Important limitations

- The churn and sentiment datasets are synthetic and intended for learning.
- Accuracy on the generated sentiment dataset can look unusually high because its language patterns are deliberately simple.
- Uploaded digit images that look very different from the 8×8 scikit-learn digits may be misclassified.
- Do not use any of these outputs for medical, financial, legal or other high-stakes decisions.


## 12. Train only one model (optional)

```bash
uv run python train_ann.py
uv run python train_lstm.py
uv run python train_cnn.py
```

Use these when you want to study one architecture at a time.
