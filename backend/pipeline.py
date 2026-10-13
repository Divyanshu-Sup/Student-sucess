import numpy as np
import pandas as pd

FEATURES = ["attendance_rate", "avg_quiz_score", "assignment_attempts", "time_spent_minutes",
            "prev_term_gpa", "topic_algebra_score", "topic_calculus_score", "topic_stats_score",
            "late_submissions", "forum_participation"]
MODEL_COLS = FEATURES + ["engagement_index"]


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df_raw.index)
    ab = df_raw["absences"].fillna(0).clip(lower=0)
    d["attendance_rate"] = 100 * (1 - ab / (ab + 100))
    g1, g2 = df_raw["G1"].astype(float), df_raw["G2"].astype(float)
    d["avg_quiz_score"] = (g1 / 20 * 100).clip(0, 100)
    d["assignment_attempts"] = df_raw["studytime"].fillna(1).astype(int).clip(1, 4)
    d["time_spent_minutes"] = df_raw["studytime"].fillna(1).astype(float) * 60
    d["prev_term_gpa"] = (g1 / 20 * 4).clip(0, 4)
    d["topic_algebra_score"] = (g1 / 20 * 100).clip(0, 100)
    d["topic_calculus_score"] = (g2 / 20 * 100).clip(0, 100)
    d["topic_stats_score"] = ((g1 + g2) / 40 * 100).clip(0, 100)
    d["late_submissions"] = df_raw["failures"].fillna(0).astype(int).clip(lower=0)
    d["forum_participation"] = (df_raw["schoolsup"].astype(str).str.lower().eq("yes").astype(int)
                                + df_raw["famsup"].astype(str).str.lower().eq("yes").astype(int))
    return d


def get_norms(d):
    return {"forum_participation": max(float(d["forum_participation"].max()), 1.0),
            "time_spent_minutes": max(float(d["time_spent_minutes"].max()), 1.0)}


def add_engagement(d, norms):
    d = d.copy()
    d["engagement_index"] = (0.5 * d["attendance_rate"] / 100
                             + 0.3 * d["forum_participation"] / norms["forum_participation"]
                             + 0.2 * d["time_spent_minutes"] / norms["time_spent_minutes"]) * 100
    return d[MODEL_COLS].astype(float)


def shap_positive(sv):
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]
    if getattr(sv, "ndim", 0) == 3:
        return sv[:, :, 1]
    return sv


def load_students_csv(path):
    import io
    lines = []
    for ln in open(path, encoding="utf-8-sig").read().splitlines():
        ln = ln.strip()
        if len(ln) > 1 and ln[0] == '"' and ln[-1] == '"':
            ln = ln[1:-1].replace('""', '"')
        if ln:
            lines.append(ln)
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", quotechar='"')
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df