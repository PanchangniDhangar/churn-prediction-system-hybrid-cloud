from fastapi import FastAPI
from app.schemas import CustomerData, PredictionResponse
from src.pipeline.predict_pipeline import PredictPipeline
import joblib
import os

# This works perfectly with the Dockerfile WORKDIR /app
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load your model using MODEL_PATH
app = FastAPI(title="Telecom Churn Prediction API")

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    # The 'customer' object is now automatically validated
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(customer.data)
    
    status_label = "Churn" if results[0] == 1 else "Not Churn"
    
    return {
        "prediction": status_label,
        "status": "success"
    }