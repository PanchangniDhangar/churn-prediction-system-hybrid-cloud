from pydantic import BaseModel, Field
from typing import Dict, Any

class CustomerData(BaseModel):
    """
    Schema for the input data received from Streamlit.
    We use a dictionary to allow flexibility, but we can also
    explicitly define the 8-10 key fields for stricter validation.
    """
    data: Dict[str, Any] = Field(
        ..., 
        example={
            "rev_Mean": 50.0,
            "mou_Mean": 300.0,
            "totmrc_Mean": 45.0,
            "months": 12,
            "change_mou": -10.5,
            "change_rev": 2.0,
            "ovrmou_Mean": 0.0,
            "hnd_price": 199.99,
            "phones": 1,
            "models": 1
        }
    )

class PredictionResponse(BaseModel):
    """
    Schema for the response sent back to the frontend.
    """
    prediction: str = Field(..., example="Churn")
    status: str = Field(..., example="success")