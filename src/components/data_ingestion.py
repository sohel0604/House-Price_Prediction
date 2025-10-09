# src/data_ingestion.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

class DataIngestion:
    """
    Data ingestion: read housing.csv => save artifacts/raw.csv, artifacts/train.csv, artifacts/test.csv
    """
    def __init__(self, data_path="housing.csv", artifacts_dir="artifacts"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.raw_path = os.path.join(self.artifacts_dir, "raw.csv")
        self.train_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_path = os.path.join(self.artifacts_dir, "test.csv")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion...")
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path} with shape: {df.shape}")

            # Save raw copy
            df.to_csv(self.raw_path, index=False)
            logger.info(f"Saved raw data to {self.raw_path}")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.train_path, index=False)
            test_set.to_csv(self.test_path, index=False)
            logger.info(f"Saved train to {self.train_path} and test to {self.test_path}")

            return self.train_path, self.test_path
        except Exception as e:
            logger.exception("Data ingestion failed")
            raise CustomException(e,sys)
