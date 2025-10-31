import pandas as pd

from typing import Dict, List
from pprint import pp
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from model.eval import eval_model
from model.save import save_model
from src.predictions.features import categorical_features, all_features, target_column

best_params = {
        'iterations': 2000,
        'depth': 7,
        'learning_rate': 0.04467250853587068,
        'l2_leaf_reg': 2.968200742527676,
        'bagging_temperature': 0.4146706543683366,
    }


def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,  
        y_test: pd.DataFrame, 
        categorical_features: List[str], 
        params: Dict, 
        gpu: bool = True, 
        seed: int = 42
        ) -> CatBoostRegressor:
    model = CatBoostRegressor(
        **params,
        task_type= 'GPU' if gpu else 'CPU',
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=seed,
        verbose=100,
        cat_features=categorical_features
    )

    print(f"Training CatBoost model on {'GPU' if gpu else 'CPU'}...")

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100
    )

    return model

def load_final_dataset(path: str, all_features: List[str], target_column: str, categorical_features: List[str]) -> pd.DataFrame:
    print(f"Reading csv {path}...")
    df = pd.read_csv(path, usecols=all_features + [target_column])

    for col in categorical_features:
        df[col] = df[col].astype('category')

    return df

def load_initial_dataset(path: str) -> pd.DataFrame:
    print(f"Reading csv {path}...")
    df = pd.read_csv(path, usecols=["company_name", "title", "description", "location", "normalized_salary"])
    df = df.dropna(subset=['normalized_salary', 'description'])
    return df

def split_dataset(df: pd.DataFrame, all_features: List[str], target_column: str, test_size: int = 0.2, seed: int = 42):
    print("Splitting dataset...")
    X = df[all_features]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=seed)



if __name__ == '__main__':
    path = 'data/datasets/postings_final.csv'
    df = load_final_dataset(path, all_features, target_column, categorical_features)
    X_train, X_test, y_train, y_test = split_dataset(df, all_features, target_column)

    model = train_model(X_train, X_test, y_train, y_test, categorical_features, best_params, gpu=False)
    metrics = eval_model(model, X_test, y_test)

    pp(metrics)
    save_model(model)
