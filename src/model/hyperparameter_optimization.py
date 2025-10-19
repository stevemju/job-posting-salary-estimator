import pandas as pd

from typing import Dict, List
from catboost import CatBoostRegressor
from optuna.trial import Trial
from optuna import create_study
from pprint import pp

from sklearn.model_selection import train_test_split
from model.eval import eval_model
from model.save import save_model
from model.train import load_and_prepare_dataset, split_dataset
from src.predictions.features import categorical_features, all_features, target_column



def objective(
        trial: Trial,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        categorical_features: List[str],
        gpu: bool = False,
        seed: int = 42
        ):
    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    # define the hyperparameter search space
    params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'task_type': 'GPU' if gpu else 'CPU',
        'verbose': 0,
        'random_seed': seed,
        'cat_features': categorical_features,
        'iterations': 200,
        # suggest values for the hyperparameters
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 30.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.5),
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train_main, y_train_main,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=0
    )

    metrics = eval_model(model, X_val, y_val)
    return metrics['RMSE']


def run_optuna_study(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        categorical_features: List[str],
        n_trials: int = 50,
        gpu: bool = False,
        seed: int = 42
    ) -> Dict:
    study = create_study(direction='minimize')

    print("Starting hyperparameter tuning with Optuna...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, categorical_features, gpu, seed), 
        n_trials=n_trials
    )

    print("\nTuning complete.")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"    Value (RMSE): ${best_trial.value:,.2f}")
    print("    Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial.params



if __name__ == '__main__':
    path = 'data/datasets/postings_final.csv'
    df = load_and_prepare_dataset(path, all_features, target_column, categorical_features)
    X_train, X_test, y_train, y_test = split_dataset(df, all_features, target_column)

    best_params = run_optuna_study(X_train, y_train, categorical_features, n_trials=2)
    pp(best_params)

    # train a final model with the best parameters
    final_model = CatBoostRegressor(**best_params, task_type='CPU', cat_features=categorical_features)
    final_model.fit(X_train, y_train)

    metrics = eval_model(final_model, X_test, y_test)
    pp(metrics)

    save_model(final_model)
