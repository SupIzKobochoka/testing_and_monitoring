import time
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info
)

from ml_service.features import FEATURE_COLUMNS

HTTP_REQUESTS_TOTAL = Counter(
    "ml_service_http_requests_total_KLEVCOV",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    "ml_service_http_request_exceptions_total_KLEVCOV",
    "Total number of unhandled exceptions in HTTP handlers",
    ["method", "path", "exception_type"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ml_service_http_request_duration_seconds_KLEVCOV",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05,
        0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ),
)

PREPROCESS_DURATION_SECONDS = Histogram(
    "ml_service_preprocess_duration_seconds_KLEVCOV",
    "Preprocessing latency in seconds",
    buckets=(
        0.0005, 0.001, 0.0025, 0.005, 0.01,
        0.025, 0.05, 0.1, 0.25, 0.5, 1.0
    ),
)

PREPROCESS_ERRORS_TOTAL = Counter(
    "ml_service_preprocess_errors_total_KLEVCOV",
    "Total number of preprocessing errors",
    ["error_type"],
)

MISSING_FEATURES_TOTAL = Counter(
    "ml_service_missing_features_total_KLEVCOV",
    "How many times a required feature was missing in requests",
    ["feature"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "ml_service_inference_duration_seconds_KLEVCOV",
    "Model inference latency in seconds",
    buckets=(
        0.0005, 0.001, 0.0025, 0.005, 0.01,
        0.025, 0.05, 0.1, 0.25, 0.5, 1.0
    ),
)

INFERENCE_ERRORS_TOTAL = Counter(
    "ml_service_inference_errors_total_KLEVCOV",
    "Total number of inference errors",
    ["error_type"],
)

PREDICTION_PROBABILITY = Histogram(
    "ml_service_prediction_probability_KLEVCOV",
    "Distribution of model probabilities for positive class",
    buckets=(
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    ),
)

PREDICTIONS_TOTAL = Counter(
    "ml_service_predictions_total_KLEVCOV",
    "Total number of predictions by predicted class",
    ["prediction"],
)

FEATURE_NUMERIC_VALUE = Histogram(
    "ml_service_feature_numeric_value_KLEVCOV",
    "Distribution of numeric feature values",
    ["feature"],
    buckets=(
        0, 1, 5, 10, 18, 25, 35, 45, 55, 65, 75, 90,
        100, 1000, 10000, 100000, 1000000
    ),
)

FEATURE_CATEGORICAL_VALUE_TOTAL = Counter(
    "ml_service_feature_categorical_value_total_KLEVCOV",
    "Count of categorical feature values",
    ["feature", "value"],
)

MODEL_UPDATES_TOTAL = Counter(
    "ml_service_model_updates_total_KLEVCOV",
    "Total number of successful model updates",
    ["run_id", "model_type"],
)

MODEL_UPDATE_ERRORS_TOTAL = Counter(
    "ml_service_model_update_errors_total_KLEVCOV",
    "Total number of model update errors",
    ["error_type"],
)

CURRENT_MODEL_INFO = Info(
    "ml_service_current_model_KLEVCOV",
    "Current production model info",
)

CURRENT_MODEL_LAST_UPDATE_UNIX = Gauge(
    "ml_service_current_model_last_update_unix_KLEVCOV",
    "Unix timestamp of the last successful model update",
)

CURRENT_MODEL_REQUIRED_FEATURE = Gauge(
    "ml_service_current_model_required_feature_KLEVCOV",
    "Required features of the current production model (1 - required, 0 - not required)",
    ["feature"],
)


def _extract_model_type(model: Any) -> str:
    if model is None:
        return "unknown"
    if hasattr(model, "steps") and model.steps:
        try:
            return model.steps[-1][1].__class__.__name__
        except Exception:
            pass
    return model.__class__.__name__


def observe_request_feature_stats(payload: dict[str, Any]) -> None:
    for feature, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            FEATURE_NUMERIC_VALUE.labels(feature=feature).observe(float(value))
        else:
            # Для этого датасета кардинальность значений ограничена и приемлема.
            # Если категорий станет много, лучше нормализовать/ограничить value.
            FEATURE_CATEGORICAL_VALUE_TOTAL.labels(
                feature=feature,
                value=str(value),
            ).inc()


def observe_missing_features(missing_features: list[str]) -> None:
    for feature in missing_features:
        MISSING_FEATURES_TOTAL.labels(feature=feature).inc()


def update_current_model_metrics(model_state) -> None:
    data = model_state.get()
    model = data.model
    run_id = data.run_id or "unknown"
    model_type = _extract_model_type(model)

    features = []
    try:
        features = list(model_state.features)
    except Exception:
        features = []

    CURRENT_MODEL_INFO.info({
        "run_id": run_id,
        "model_type": model_type,
        "features": ",".join(features),
    })

    CURRENT_MODEL_LAST_UPDATE_UNIX.set(time.time())

    # Сначала всё обнуляем, потом только нужные фичи ставим в 1
    for feature in FEATURE_COLUMNS:
        CURRENT_MODEL_REQUIRED_FEATURE.labels(feature=feature).set(0)

    for feature in features:
        if feature in FEATURE_COLUMNS:
            CURRENT_MODEL_REQUIRED_FEATURE.labels(feature=feature).set(1)

    MODEL_UPDATES_TOTAL.labels(run_id=run_id, model_type=model_type).inc()