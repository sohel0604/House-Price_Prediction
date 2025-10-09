

import os
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import evaluate_model


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.utils import evaluate_model
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(" Splitting dependent and independent variables for training and testing.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f" Model Report: {model_report}")

            # Find best model
            best_model_name = max(model_report, key=lambda k: model_report[k])
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f" Best Model Found: {best_model_name} with R2 Score: {best_model_score}")

            # Save best model
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            logging.info(f" Model saved to {self.model_trainer_config.trained_model_file_path}")

            # Evaluate best model on test data
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(f"🏁 Final Model Performance: R2={r2}, RMSE={rmse}, MAE={mae}")

            return r2

        except Exception as e:
            logging.error(f" Error in ModelTrainer: {e}")
            raise CustomException(e, sys)
