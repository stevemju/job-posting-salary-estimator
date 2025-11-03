from catboost import CatBoostRegressor
from datetime import datetime


def save_model(model: CatBoostRegressor, name: str, folder: str = 'data/models/'):
    date = datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_name = f"{name}_{date}.cbm"
    model_path = f"{folder}{model_name}"
    model.save_model(model_path, format="cbm")
    print(f"Model saved as {model_path}")

def load_model(path: str) -> CatBoostRegressor:
    loaded_model = CatBoostRegressor()
    return loaded_model.load_model(path)
