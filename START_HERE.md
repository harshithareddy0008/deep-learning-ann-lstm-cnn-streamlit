# START HERE

This ZIP is already trained. You only need to install the environment and start Streamlit.

## Windows / VS Code terminal

```powershell
cd deep_learning_beginner_suite
uv python install 3.10.6
uv sync
uv run python smoke_test.py
uv run streamlit run app.py
```

If `uv` is not installed yet:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## What to click in the Streamlit app

- **ANN • Churn** → change customer values → `Predict churn`
- **LSTM • Sentiment** → choose/load a sample or type text → `Analyze sentiment`
- **CNN • Digits** → choose a sample digit or upload a digit image → `Recognize digit`
- **Model Card** → show test metrics and architectures

## Retrain everything

```powershell
uv run python train_all.py
```

Or train separately:

```powershell
uv run python train_ann.py
uv run python train_lstm.py
uv run python train_cnn.py
```

## Dataset location

- `data/customer_churn.csv` — ANN
- `data/movie_reviews.csv` — LSTM
- CNN uses the built-in `sklearn.datasets.load_digits()` dataset

## Deploy

Read `DEPLOY_RENDER.md`, push the project to GitHub, then create a Docker Web Service on Render.
