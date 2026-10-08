from pathlib import Path
from src.data import save_generated_datasets
from train_all import DATA_DIR, seed_everything, train_ann

if __name__ == "__main__":
    seed_everything()
    churn_path, _ = save_generated_datasets(DATA_DIR)
    train_ann(Path(churn_path))
