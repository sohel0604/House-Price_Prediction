
# utils.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    """Save any Python object (like model or preprocessor) using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f" Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load any saved object (model or preprocessor)."""
    try:
        obj = joblib.load(file_path)
        logging.info(f" Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Train and evaluate multiple ML models and return R² scores."""
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)

            report[name] = round(test_r2, 4)
            logging.info(f" {name} → R² Score: {test_r2:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
