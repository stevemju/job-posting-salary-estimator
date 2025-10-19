import numpy as np
import pandas as pd

from typing import Dict
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def eval_model(model: CatBoostRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    predictions_log = model.predict(X_test)

    # convert log predictions back to dollar amounts
    predictions_actual = np.expm1(predictions_log)
    y_test_actual = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    return {'RMSE': rmse}
