import io
import json
import os
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from feature_prep import prepare_model_input


class PredictionException(Exception):
    pass


JSON_CONTENT_TYPES = {"application/json"}
CSV_CONTENT_TYPES = {"text/csv", "text/plain"}


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load all artifacts needed for inference from /opt/ml/model."""
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    model_path = os.path.join(model_dir, "nyc_info_random_forest_model.joblib")
    zip_lookup_path = os.path.join(model_dir, "zip_lat_lon_lookup.csv")

    for path in [preprocessor_path, model_path, zip_lookup_path]:
        if not os.path.exists(path):
            raise PredictionException(f"Missing artifact: {path}")

    try:
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        zip_lookup = pd.read_csv(zip_lookup_path, dtype={"incident_zip": str})
        zip_lookup["incident_zip"] = zip_lookup["incident_zip"].astype(str).str.strip()
    except Exception as e:
        raise PredictionException(f"Failed to load model artifacts: {e}") from e

    return {
        "preprocessor": preprocessor,
        "model": model,
        "zip_lookup": zip_lookup,
    }


def _payload_to_dataframe(
    payload: Union[Dict[str, Any], List[Dict[str, Any]], List[Any]]
) -> pd.DataFrame:
    """Normalize supported JSON payload shapes into a DataFrame."""
    if isinstance(payload, dict):
        if "instances" in payload:
            instances = payload["instances"]
            if not isinstance(instances, list):
                raise PredictionException("'instances' must be a list.")
            return pd.DataFrame(instances)

        if "data" in payload:
            data = payload["data"]
            if not isinstance(data, list):
                raise PredictionException("'data' must be a list of records.")
            return pd.DataFrame(data)

        return pd.DataFrame([payload])

    if isinstance(payload, list):
        if len(payload) == 0:
            return pd.DataFrame()
        return pd.DataFrame(payload)

    raise PredictionException("Unsupported JSON payload shape.")


def input_fn(request_body: Union[str, bytes], request_content_type: str) -> pd.DataFrame:
    """Deserialize the incoming request into a pandas DataFrame."""
    content_type = (request_content_type or "").split(";")[0].strip().lower()

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    try:
        if content_type in JSON_CONTENT_TYPES:
            payload = json.loads(request_body)
            df = _payload_to_dataframe(payload)
        elif content_type in CSV_CONTENT_TYPES:
            df = pd.read_csv(io.StringIO(request_body))
        else:
            raise PredictionException(
                f"Unsupported content type '{request_content_type}'. "
                f"Use application/json or text/csv."
            )
    except json.JSONDecodeError as e:
        raise PredictionException(f"Invalid JSON payload: {e.msg}") from e
    except pd.errors.EmptyDataError as e:
        raise PredictionException("CSV payload is empty.") from e
    except PredictionException:
        raise
    except Exception as e:
        raise PredictionException(f"Failed to parse request body: {e}") from e

    if df.empty:
        raise PredictionException("Request payload produced an empty DataFrame.")

    return df


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare features, transform them, and score with the RF model."""
    zip_lookup = model_artifacts["zip_lookup"]
    preprocessor = model_artifacts["preprocessor"]
    model = model_artifacts["model"]

    try:
        prepared = prepare_model_input(input_data, zip_lookup)
        transformed = preprocessor.transform(prepared)

        pred_class = model.predict(transformed)

        if not hasattr(model, "predict_proba"):
            raise PredictionException("Loaded model does not support predict_proba().")

        pred_probs = model.predict_proba(transformed)
        if pred_probs.shape[1] < 2:
            raise PredictionException("Expected binary classification probabilities.")

        pred_prob_gt_7d = pred_probs[:, 1]
        pred_prob_within_7d = 1.0 - pred_prob_gt_7d

    except PredictionException:
        raise
    except Exception as e:
        raise PredictionException(f"Prediction failed: {e}") from e

    results = input_data.copy()
    results["pred_gt_7d"] = pred_class.astype(int)
    results["pred_prob_gt_7d"] = pred_prob_gt_7d.astype(float)
    results["pred_prob_within_7d"] = pred_prob_within_7d.astype(float)
    results["predicted_close_within_7_days"] = (
        results["pred_prob_within_7d"] >= 0.5
    ).astype(int)

    return {
        "n_rows": int(len(results)),
        "predictions": results.to_dict(orient="records"),
    }


def output_fn(prediction: Dict[str, Any], accept: str) -> Tuple[str, str]:
    """Serialize the prediction response."""
    accept_type = (accept or "application/json").split(";")[0].strip().lower()

    if accept_type == "application/json":
        return json.dumps(prediction, default=_json_default), "application/json"

    if accept_type == "text/csv":
        df = pd.DataFrame(prediction["predictions"])
        return df.to_csv(index=False), "text/csv"

    raise PredictionException(
        f"Unsupported accept type '{accept}'. Use application/json or text/csv."
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")