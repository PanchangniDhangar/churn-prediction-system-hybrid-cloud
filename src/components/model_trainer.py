import os
import sys
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, train_target, test_array, test_target):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array,
                train_target,
                test_array,
                test_target
            )

            logging.info("Initializing XGBoost with Early Stopping (XGBoost 2.0+ API)")
            
            # IMPROVEMENT: Define early_stopping_rounds in the constructor
            model = XGBClassifier(
                n_estimators=1000, 
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=50, # FIXED: Moved from .fit() to here
                random_state=42
            )

            # Train the model
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)], # Validation set for early stopping
                verbose=False
            )

            logging.info("Model training completed. Evaluating metrics...")

            # Predictions
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)

            # Metrics
            auc_score = roc_auc_score(y_test, probs)
            recall = recall_score(y_test, preds)
            acc = accuracy_score(y_test, preds)

            logging.info(f"Best Model found! ROC-AUC: {auc_score:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return auc_score

        except Exception as e:
            raise CustomException(e, sys)