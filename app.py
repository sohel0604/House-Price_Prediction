#  app.py — Main Pipeline Runner

import sys
import os

#  Add root folder to sys.path so Python can find "src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException


if __name__ == "__main__":
    try:
        logging.info(" Pipeline Execution Started")

        # ==========================================
        # 1️⃣ DATA INGESTION
        # ==========================================
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(f" Data Ingestion Completed: \nTrain → {train_data_path}\nTest → {test_data_path}")

        # ==========================================
        # 2️⃣ DATA TRANSFORMATION
        # ==========================================
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f" Data Transformation Completed. Preprocessor saved at: {preprocessor_path}")

        # ==========================================
        # 3️⃣ MODEL TRAINING
        # ==========================================
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f" Model Training Completed. Best Model R² Score: {r2:.4f}")

        print("\n Pipeline executed successfully!")
        print(f" Final Model R² Score: {r2:.4f}")
        print(" Model saved in: artifacts/model.pkl")
        print(" Preprocessor saved in: artifacts/preprocessor.pkl")

        logging.info(" Pipeline Execution Completed Successfully ")

    except Exception as e:
        logging.error(f"Pipeline Execution Failed: {e}")
        raise CustomException(e, sys)
