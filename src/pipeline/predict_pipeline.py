import sys
import os
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        # The 27 base features the preprocessor expects
        self.feature_list = [
            'mou_Mean', 'avgmou', 'peak_vce_Mean', 'opk_vce_Mean', 'mou_peav_Mean', 
            'mou_opkv_Mean', 'roam_Mean', 'change_mou', 'change_rev', 'rev_Mean', 
            'totmrc_Mean', 'ovrmou_Mean', 'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 
            'drop_blk_Mean', 'attempt_Mean', 'complete_Mean', 'months', 'uniqsubs', 
            'actvsubs', 'eqpdays', 'phones', 'models', 'hnd_price', 'refurb_new', 'creditcd'
        ]

    def predict(self, user_data_dict):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            test_data_path = os.path.join("artifacts", "test.csv")

            with open(model_path, "rb") as f:
                model = dill.load(f)
            with open(preprocessor_path, "rb") as f:
                preprocessor = dill.load(f)

            # 1. LOAD REFERENCE PROFILE (Medians from dataset)
            test_df = pd.read_csv(test_data_path)
            reference_profile = test_df[self.feature_list].median(numeric_only=True).to_dict()
            reference_profile['refurb_new'] = test_df['refurb_new'].mode()[0]
            reference_profile['creditcd'] = test_df['creditcd'].mode()[0]

            # 2. MERGE USER INPUTS
            full_data = reference_profile.copy()
            for key, value in user_data_dict.items():
                full_data[key] = value

            # 3. INTERACTION FEATURE ENGINEERING (Accuracy Boosters)
            # These derived features help the model understand 'value for money'
            full_data['rev_per_mou'] = full_data['rev_Mean'] / (full_data['mou_Mean'] + 1)
            full_data['overage_ratio'] = full_data['ovrmou_Mean'] / (full_data['mou_Mean'] + 1)
            full_data['equipment_age_ratio'] = full_data['eqpdays'] / (full_data['months'] + 1)

            # 4. PREDICT
            final_df = pd.DataFrame([full_data])
            # Ensure order matches training exactly
            final_df = final_df[self.feature_list] 
            
            data_transformed = preprocessor.transform(final_df)
            preds = model.predict(data_transformed)
            
            return preds

        except Exception as e:
            raise CustomException(e, sys)