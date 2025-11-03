import pandas as pd

from typing import Dict, List
from catboost import CatBoostRegressor
from optuna.trial import Trial
from optuna import create_study
from pprint import pp

from sklearn.model_selection import train_test_split
from model.eval import eval_model
from model.save import save_model
from model.train import load_final_dataset, split_dataset
from src.predictions.features import categorical_features, all_features, target_column



def objective(
        trial: Trial,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        categorical_features: List[str],
        alpha: float,
        gpu: bool = False,
        seed: int = 42
        ):
    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    loss_function_name = f'Quantile:alpha={alpha}'
    eval_metric_name = f'Quantile:alpha={alpha}'

    # define the hyperparameter search space
    params = {
        'loss_function': loss_function_name,
        'eval_metric': eval_metric_name,
        'task_type': 'GPU' if gpu else 'CPU',
        'verbose': 0,
        'random_seed': seed,
        'cat_features': categorical_features,
        'iterations': 3000,
        # suggest values for the hyperparameters
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 30.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.5),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train_main, y_train_main,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=0
    )

    trial.set_user_attr('best_iteration', model.get_best_iteration() + 1)

    best_score = model.get_best_score()['validation'][eval_metric_name]
    return best_score


def run_optuna_study(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        categorical_features: List[str],
        alpha: float,
        n_trials: int = 50,
        gpu: bool = False,
        seed: int = 42
    ) -> Dict:
    study = create_study(direction='minimize')

    print(f"Starting hyperparameter tuning for Quantile alpha={alpha}...")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, categorical_features, alpha, gpu, seed), 
        n_trials=n_trials
    )

    print("\nTuning complete.")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value (Quantile Loss): {best_trial.value:,.4f}")

    best_params = best_trial.params
    best_iteration = best_trial.user_attrs.get('best_iteration', 3000)
    best_params['iterations'] = best_iteration

    print("    Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_params



if __name__ == '__main__':
    path = 'data/datasets/postings_final.csv'
    df = load_final_dataset(path, all_features, target_column, categorical_features)
    X_train, X_test, y_train, y_test = split_dataset(df, all_features, target_column)

    lower_bound_alpha = 0.25
    upper_bound_alpha = 0.75

    # Optimize lower bound model
    lower_bound_best_params = run_optuna_study(X_train, y_train, categorical_features, alpha=lower_bound_alpha, n_trials=1)
    print(f"\n\nBest params for model with alpha={lower_bound_alpha}:")
    pp(lower_bound_best_params)

    # Train and export lower bound model
    print("\nTraining final lower model with best parameters...")
    lower_model = CatBoostRegressor(
        **lower_bound_best_params, 
        loss_function=f'Quantile:alpha={lower_bound_alpha}',
        task_type='CPU', 
        cat_features=categorical_features)
    lower_model.fit(X_train, y_train)
    print(f"\n\nFinal metrics for model with alpha={lower_bound_alpha}:")
    pp(eval_model(lower_model, X_test, y_test))
    save_model(lower_model, name="lower_catboost")

    # Optimize upper bound model
    upper_bound_best_params = run_optuna_study(X_train, y_train, categorical_features, alpha=upper_bound_alpha, n_trials=1)
    print(f"\n\nBest params for model with alpha={upper_bound_alpha}:")
    pp(upper_bound_best_params)

    # Train and export upper bound model
    print("\nTraining final upper model with best parameters...")
    upper_model = CatBoostRegressor(
        **upper_bound_best_params, 
        loss_function=f'Quantile:alpha={upper_bound_alpha}',
        task_type='CPU', 
        cat_features=categorical_features)
    upper_model.fit(X_train, y_train)
    print(f"\n\nFinal metrics for model with alpha={upper_bound_alpha}:")
    pp(eval_model(upper_model, X_test, y_test))
    save_model(upper_model, name="upper_catboost")
    