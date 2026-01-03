from fastapi import FastAPI
from app.schemas import CustomerData, PredictionResponse
from src.pipeline.predict_pipeline import PredictPipeline

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