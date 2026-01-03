import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    try:
        # Define the path to your dataset
        data_path = r"C:\Users\panch\Desktop\folders\PROJECTS\telecom_churn_prediction\data\Telecom_customer churn.csv"
        
        # 1. Data Ingestion
        logging.info("Starting Data Ingestion")
        ingestion_obj = DataIngestion()
        # PASS THE PATH HERE
        train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion(data_path)

        # 2. Data Transformation
        logging.info("Starting Data Transformation")
        data_transformation = DataTransformation()
        train_arr, train_target, test_arr, test_target = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # 3. Model Training
        logging.info("Starting Model Training")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(
            train_arr, train_target, test_arr, test_target
        )
        
        print(f"\nModel Training Successful!")
        print(f"Model Evaluation Metric (ROC-AUC/Accuracy): {accuracy:.4f}")

    except Exception as e:
        logging.error(f"Error in main.py execution: {str(e)}")
        raise CustomException(e, sys)