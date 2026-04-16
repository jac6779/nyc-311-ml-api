from fastapi import FastAPI, Body
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="NYC 311 Prediction API",
    version="1.0.0",
    description="Predict NYC 311 complaint outcome from raw complaint features."
)

pipeline = joblib.load("models/deploy_pipeline.joblib")


class PredictionRequest(BaseModel):
    agency: str
    complaint_type: str
    location_type: str
    borough: str
    latitude: float
    longitude: float
    complaint_hr: int
    complaint_day: int
    complaint_month: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "agency": "NYPD",
                "complaint_type": "Noise - Residential",
                "location_type": "Residential Building/House",
                "borough": "BROOKLYN",
                "latitude": 40.6943,
                "longitude": -73.9928,
                "complaint_hr": 22,
                "complaint_day": 5,
                "complaint_month": 4
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.model_dump()])
    prediction = pipeline.predict(df)[0]

    response = {"prediction": float(prediction)}

    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(df)[0][1]
        response["probability"] = float(probability)

    return response