import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_assistant import AssistantChatRequest, generate_assistant_reply
from pipeline import MODEL_COLS, add_engagement, prepare, shap_positive

BASE_DIR = Path(__file__).resolve().parent
art = joblib.load(BASE_DIR / "model.joblib")
model, norms, explainer = art["model"], art["norms"], art["explainer"]

app = FastAPI(
    title="Student Success API",
    description="Student risk prediction and Gemini-powered study support.",
    version="2.0.0",
)

# Set CORS_ORIGINS on the host as a comma-separated list for custom deployments.
_default_origins = (
    "https://divyanshu-sup.github.io,"
    "http://127.0.0.1:5500,http://localhost:5500"
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Student(BaseModel):
    G1: int = Field(ge=0, le=20)
    G2: int = Field(ge=0, le=20)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    absences: int = Field(ge=0, le=100)
    schoolsup: str = Field(default="no", pattern="^(yes|no)$")
    famsup: str = Field(default="no", pattern="^(yes|no)$")


@app.get("/health")
def health():
    return {"status": "ok", "model": art["name"], "assistant": "gemini"}


@app.post("/predict")
def predict(s: Student):
    X = add_engagement(prepare(pd.DataFrame([s.model_dump()])), norms)
    probability = float(model.predict_proba(X)[0, 1])
    shap_values = np.asarray(shap_positive(explainer.shap_values(X))).reshape(-1)
    factors = sorted(
        (
            {"feature": feature, "impact": float(value), "value": float(X[feature].iloc[0])}
            for feature, value in zip(MODEL_COLS, shap_values)
        ),
        key=lambda item: -abs(item["impact"]),
    )[:6]
    return {
        "risk_percent": round(probability * 100, 1),
        "at_risk": probability >= 0.5,
        "model": art["name"],
        "factors": factors,
    }


@app.post("/assistant/chat")
def assistant_chat(request: AssistantChatRequest):
    """Generate study guidance; role selection is not authentication."""
    return {
        "reply": generate_assistant_reply(request),
        "role": request.role,
    }
