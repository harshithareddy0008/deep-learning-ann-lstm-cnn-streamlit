# Render Deployment Checklist

## Recommended route: Docker

Docker is used here because it guarantees Python **3.10.6** and keeps your local and production environments aligned.

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Deep learning ANN LSTM CNN project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

### Step 2 — Create Render Web Service

- Render Dashboard → **New** → **Web Service**
- Connect the GitHub repository
- Runtime/Language: **Docker**
- Dockerfile path: `./Dockerfile`
- Health check path: `/_stcore/health`
- Create the service

No custom start command is required because `Dockerfile` already starts Streamlit using Render's `$PORT` environment variable.

### Step 3 — Blueprint alternative

The included `render.yaml` declares the service as a Docker web service. You can create a Blueprint from the repository instead of filling the service form manually.

### Step 4 — Common troubleshooting

**Build fails because a model file is missing**  
Run `uv run python train_all.py`, confirm files exist in `artifacts/`, commit them, and push again.

**App starts locally but Render says port problem**  
Do not hard-code port 8501 in production. This project uses `${PORT:-10000}` automatically.

**Memory issue**  
This app keeps three small PyTorch models in memory. If a selected Render instance cannot hold the Python/PyTorch runtime, use a larger instance type.

**After retraining**  
Commit the updated files inside `artifacts/` so the deployed app uses the new weights.
