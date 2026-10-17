
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.predict import predict_fraud


# Create the FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud detection using Random Forest",
    version="1.0.0"
)


# Define the expected request structure
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="V1-V28 and Amount"
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }


# Fraud prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    try:
        result = predict_fraud(transaction.features)
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )
