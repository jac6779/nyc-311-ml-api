import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "latitude",
    "longitude",
    "agency",
    "complaint_type",
    "borough",
    "location_type",
    "complaint_hr",
    "complaint_day",
    "complaint_month"
]

def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    for col in ["agency", "complaint_type", "borough", "location_type"]:
        if col not in X.columns:
            X[col] = "unknown"
        X[col] = X[col].fillna("unknown").astype(str).str.strip()
        X.loc[X[col].eq("") | X[col].eq("nan"), col] = "unknown"

    for col in ["latitude", "longitude"]:
        if col not in X.columns:
            X[col] = np.nan
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in ["complaint_hr", "complaint_day", "complaint_month"]:
        if col not in X.columns:
            X[col] = -1
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype(int)

    return X[MODEL_FEATURES].copy()