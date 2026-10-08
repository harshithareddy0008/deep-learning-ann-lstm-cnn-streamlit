from pathlib import Path
from src.data import save_generated_datasets
from train_all import DATA_DIR, seed_everything, train_lstm

if __name__ == "__main__":
    seed_everything()
    _, sentiment_path = save_generated_datasets(DATA_DIR)
    train_lstm(Path(sentiment_path))
