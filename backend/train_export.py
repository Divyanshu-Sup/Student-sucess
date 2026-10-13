import json, joblib, numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from pipeline import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR.parent / "docs" / "data"

raw = load_students_csv(BASE_DIR / "data" / "student-mat.csv")
base = prepare(raw)
base["at_risk"] = (raw["G3"].astype(float) < 10).astype(int)
base = base.replace([np.inf, -np.inf], np.nan).dropna()
raw = raw.loc[base.index]
y = base.pop("at_risk")
norms = get_norms(base)
X = add_engagement(base, norms)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced").fit(Xtr, ytr)
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42,
                    scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1)).fit(Xtr, ytr)
models = {"RandomForest": rf, "XGBoost": xgb}

metrics = {}
for n, m in models.items():
    p, pr = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    metrics[n] = {"accuracy": accuracy_score(yte, p), "roc_auc": roc_auc_score(yte, pr),
                "precision": precision_score(yte, p, zero_division=0), "recall": recall_score(yte, p, zero_division=0),
                "f1": f1_score(yte, p, zero_division=0), "confusion_matrix": confusion_matrix(yte, p).tolist()}
best = max(metrics, key=lambda n: metrics[n]["roc_auc"])
model = models[best]
metrics["best_model"] = best
metrics["test_size"] = int(len(yte))

expl = shap.TreeExplainer(model)
sv = np.asarray(shap_positive(expl.shap_values(X)))
imp = pd.Series(model.feature_importances_, index=MODEL_COLS).sort_values(ascending=False)
importance = {"model_importance": [{"feature": k, "value": float(v)} for k, v in imp.items()],
                    "shap_mean_abs": [{"feature": k, "value": float(v)} for k, v in
                                pd.Series(np.abs(sv).mean(0), index=MODEL_COLS).sort_values(ascending=False).items()]}

prob = model.predict_proba(X)[:, 1]
students = []
for i, idx in enumerate(X.index):
    top = int(np.argmax(sv[i]))
    students.append({"id": f"STU{idx+1:04d}", "risk": round(float(prob[i]) * 100, 1), "at_risk": int(prob[i] >= .5),
                    "G1": int(raw.loc[idx, "G1"]), "G2": int(raw.loc[idx, "G2"]), "G3": int(raw.loc[idx, "G3"]),
                    "failures": int(raw.loc[idx, "failures"]), "absences": int(raw.loc[idx, "absences"]),
                    "studytime": int(raw.loc[idx, "studytime"]), "top_factor": MODEL_COLS[top]})
students.sort(key=lambda s: -s["risk"])

hist = np.histogram(prob * 100, bins=[0, 20, 40, 60, 80, 100.01])[0].tolist()
stats = {"total": len(X), "at_risk_actual": int(y.sum()), "at_risk_rate": float(y.mean()),
        "predicted_at_risk": int((prob >= .5).sum()), "avg_g3": float(raw["G3"].mean()),
        "avg_absences": float(raw["absences"].mean()),
        "risk_distribution": {"labels": ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], "counts": hist},
        "risk_by_failures": {str(k): float(v) for k, v in y.groupby(raw["failures"]).mean().items()}}

joblib.dump({"model": model, "norms": norms, "name": best, "explainer": expl}, "model.joblib")
for name, obj in [("metrics", metrics), ("importance", importance), ("students", students), ("stats", stats)]:
    json.dump(obj, open(f"{OUT}{name}.json", "w"), indent=1)
print("Best:", best, {k: round(v, 3) for k, v in metrics[best].items() if isinstance(v, float)})
print("Rows:", len(X), "at-risk rate:", round(float(y.mean()), 3))