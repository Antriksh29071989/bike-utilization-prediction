import logging
import os

import joblib
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def _is_dir_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def train_model(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, model_save_path: str):
    """ Train and test the model with X and Y datasets
    :param x_train: training data (75% of original data)
    :param y_train: labels (75% of original data)
    :param x_test:  test data (25% of original data)
    :param y_test:  labels (25% of original data)
    :param model_save_path: Save model on local path.
    :return:
    """
    logger.info("Initiating model..")
    try:
        model = RandomForestRegressor()
        # Hyper-tuning parameters
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='neg_mean_absolute_error',
                                   verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        logger.info("model training completed.")
        best_model = grid_search.best_estimator_
        _is_dir_exists(model_save_path)
        # Saving the model, so we don't have to train the model for the same dataset and model can be used for
        # prediction/scoring.
        joblib.dump(best_model, model_save_path)
        logger.info("Started with prediction..")
        y_pred = best_model.predict(x_test)
        logger.info("Prediction completed.")
        return mean_absolute_error(y_test, y_pred)
    except Exception as e:
        logger.error(f"Issue encountered in training the model.Manual intervention is required {e}")
        raise
