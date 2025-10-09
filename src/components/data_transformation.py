# data_transformation.py

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        """
        Creates preprocessing pipelines for both numeric and categorical features.
        """
        try:
            logging.info(" Creating preprocessing pipelines...")

            numeric_features = [
                'longitude',
                'latitude',
                'housing_median_age',
                'total_rooms',
                'total_bedrooms',
                'population',
                'households',
                'median_income'
            ]
            categorical_features = ['ocean_proximity']

            # Numeric pipeline: handle missing + scale
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline: encode + scale
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            logging.info(" Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(" Error while creating preprocessor")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test data, applies preprocessing,
        and returns transformed arrays along with preprocessor path.
        """
        try:
            logging.info(" Starting data transformation...")

            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f" Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Define target and features
            target_column = 'median_house_value'
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info(" Splitting target and features completed.")

            # Get preprocessor
            preprocessor = self.get_preprocessor()

            # Fit-transform train and transform test
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info(" Data transformed successfully.")

            # Combine X and y for model input
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f" Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            logging.info(" Data transformation completed successfully.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(" Data transformation failed")
            raise CustomException(e, sys)
