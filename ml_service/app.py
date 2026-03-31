from typing import Any
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ml_service import config
from ml_service.features import to_dataframe, MissingColumnsError, FEATURE_COLUMNS
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from ml_service.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_EXCEPTIONS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    PREPROCESS_DURATION_SECONDS,
    PREPROCESS_ERRORS_TOTAL,
    INFERENCE_DURATION_SECONDS,
    INFERENCE_ERRORS_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    MODEL_UPDATE_ERRORS_TOTAL,
    observe_request_feature_stats,
    observe_missing_features,
    update_current_model_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = Model()
    configure_mlflow()

    run_id = config.default_run_id()
    model.set(run_id=run_id)

    app.state.model = model
    update_current_model_metrics(model)

    logger.info("Model loaded on startup: run_id=%s", run_id)
    yield


def get_model(request: Request):
    return request.app.state.model


def _get_missing_features(request_data: dict[str, Any], needed_columns: list[str]) -> list[str]:
    required_fields = [
        column.replace(".", "_")
        for column in needed_columns
        if column in FEATURE_COLUMNS
    ]
    return [field for field in required_fields if field not in request_data]


def create_app() -> FastAPI:
    app = FastAPI(title="MLflow FastAPI service", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def prometheus_http_middleware(request: Request, call_next):
        method = request.method
        path = request.url.path
        start_time = time.perf_counter()

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(
                method=method,
                path=path,
                exception_type=exc.__class__.__name__,
            ).inc()
            raise
        finally:
            duration = time.perf_counter() - start_time

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                path=path,
            ).observe(duration)

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status_code=str(status_code),
            ).inc()

    @app.get("/health")
    def health(model_state=Depends(get_model)) -> dict[str, Any]:
        current_model = model_state.get()
        return {"status": "ok", "run_id": current_model.run_id}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, model_state=Depends(get_model)) -> PredictResponse:
        current_model = model_state.get().model
        if current_model is None:
            logger.error("Prediction requested but model is not loaded")
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        request_data = request.model_dump(exclude_none=True, by_alias=False)
        observe_request_feature_stats(request_data)

        preprocess_start = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=model_state.features)
        except MissingColumnsError as exc:
            PREPROCESS_DURATION_SECONDS.observe(time.perf_counter() - preprocess_start)
            PREPROCESS_ERRORS_TOTAL.labels(error_type=exc.__class__.__name__).inc()

            missing_features = _get_missing_features(request_data, model_state.features)
            observe_missing_features(missing_features)

            logger.warning("Preprocessing failed: %s", exc)
            raise HTTPException(status_code=404, detail=str(exc))
        PREPROCESS_DURATION_SECONDS.observe(time.perf_counter() - preprocess_start)

        inference_start = time.perf_counter()
        try:
            probability = current_model.predict_proba(df)[0][1]
        except Exception as exc:
            INFERENCE_DURATION_SECONDS.observe(time.perf_counter() - inference_start)
            INFERENCE_ERRORS_TOTAL.labels(error_type=exc.__class__.__name__).inc()

            logger.error("Inference failed: %s", exc)
            raise HTTPException(status_code=505, detail="Error while predicting")
        INFERENCE_DURATION_SECONDS.observe(time.perf_counter() - inference_start)

        prediction = int(probability >= 0.5)

        PREDICTION_PROBABILITY.observe(probability)
        PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post("/updateModel", response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest, model_state=Depends(get_model)) -> UpdateModelResponse:
        run_id = req.run_id
        available_runs = config.get_all_runs()
        available_run_ids = [run["run_id"] for run in available_runs]

        if run_id not in available_run_ids:
            MODEL_UPDATE_ERRORS_TOTAL.labels(error_type="URINotFoundError").inc()
            logger.warning("Model update failed: run_id=%s not found", run_id)
            raise config.URINotFoundError(
                f"cant find run_id={run_id}, avaliables models: {available_runs}"
            )

        try:
            model_state.set(run_id=run_id)
            update_current_model_metrics(model_state)
        except Exception as exc:
            MODEL_UPDATE_ERRORS_TOTAL.labels(error_type=exc.__class__.__name__).inc()
            logger.error("Model update failed: %s", exc)
            raise

        logger.info("Model updated: run_id=%s", run_id)
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()