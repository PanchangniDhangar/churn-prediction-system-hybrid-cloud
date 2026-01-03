import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self):
        # The 27 base features for prediction
        self.selected_features = [
            'mou_Mean', 'avgmou', 'peak_vce_Mean', 'opk_vce_Mean', 'mou_peav_Mean', 
            'mou_opkv_Mean', 'roam_Mean', 'change_mou', 'change_rev', 'rev_Mean', 
            'totmrc_Mean', 'ovrmou_Mean', 'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 
            'drop_blk_Mean', 'attempt_Mean', 'complete_Mean', 'months', 'uniqsubs', 
            'actvsubs', 'eqpdays', 'phones', 'models', 'hnd_price', 'refurb_new', 'creditcd'
        ]

    def get_data_transformer_object(self):
        try:
            # 25 Numerical features
            numerical_columns = [
                'mou_Mean', 'avgmou', 'peak_vce_Mean', 'opk_vce_Mean', 'mou_peav_Mean', 
                'mou_opkv_Mean', 'roam_Mean', 'change_mou', 'change_rev', 'rev_Mean', 
                'totmrc_Mean', 'ovrmou_Mean', 'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 
                'drop_blk_Mean', 'attempt_Mean', 'complete_Mean', 'months', 'uniqsubs', 
                'actvsubs', 'eqpdays', 'phones', 'models', 'hnd_price'
            ]
            categorical_columns = ['refurb_new', 'creditcd']

            # QuantileTransformer for better accuracy with skewed data
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("quantile", QuantileTransformer(output_distribution='normal', random_state=42)),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            return ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Filter for 28 columns (27 features + 1 target)
            train_df = train_df[self.selected_features + ['churn']]
            test_df = test_df[self.selected_features + ['churn']]

            preprocessing_obj = self.get_data_transformer_object()
            
            X_train = train_df.drop(columns=["churn"], axis=1)
            y_train = train_df["churn"]
            X_test = test_df.drop(columns=["churn"], axis=1)
            y_test = test_df["churn"]

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            save_object(os.path.join('artifacts', 'preprocessor.pkl'), preprocessing_obj)

            return X_train_arr, y_train, X_test_arr, y_test
        except Exception as e:
            raise CustomException(e, sys)