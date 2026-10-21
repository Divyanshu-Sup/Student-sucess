import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pipeline import prepare, add_engagement, shap_positive, MODEL_COLS
from pathlib import Path
from ai_assistant import AssistantChatRequest, generate_assistant_reply

BASE_DIR = Path(__file__).resolve().parent
art = joblib.load(BASE_DIR / "model.joblib")
model, norms, explainer = art["model"], art["norms"], art["explainer"]

app = FastAPI(title="Student Success API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Student(BaseModel):
    G1: int = Field(ge=0, le=20)          # period 1 grade
    G2: int = Field(ge=0, le=20)          # period 2 grade
    studytime: int = Field(ge=1, le=4)    # weekly study time band
    failures: int = Field(ge=0, le=4)     # past class failures
    absences: int = Field(ge=0, le=100)   # number of school absences
    schoolsup: str = "no"                 # extra school support yes/no
    famsup: str = "no"                    # family support yes/no


@app.get("/health")
def health():
    return {"status": "ok", "model": art["name"]}


@app.post("/predict")
def predict(s: Student):
    X = add_engagement(prepare(pd.DataFrame([s.model_dump()])), norms)
    p = float(model.predict_proba(X)[0, 1])
    sv = np.asarray(shap_positive(explainer.shap_values(X))).reshape(-1)
    factors = sorted(({"feature": f, "impact": float(v), "value": float(X[f].iloc[0])}
    for f, v in zip(MODEL_COLS, sv)), key=lambda d: -abs(d["impact"]))[:6]
    return {"risk_percent": round(p * 100, 1), "at_risk": p >= 0.5, "model": art["name"], "factors": factors}


@app.post("/assistant/chat")
def assistant_chat(request: AssistantChatRequest):
    """Return AI study guidance using only the context supplied with this request.

    NOTE: role is a demo selector, not authentication. Add verified identities and
    server-side authorization before enabling real student/teacher accounts.
    """
    reply = generate_assistant_reply(request)
    return {"reply": reply, "role": request.role}
