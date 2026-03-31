import pytest
import pandas as pd
from fastapi.testclient import TestClient
from ml_service.features import to_dataframe, MissingColumnsError
from ml_service.schemas import PredictRequest

from ml_service.app import create_app, get_model


class FakeModel:
    def predict_proba(self, df):
        return [[0.1, 0.9]]


class FakeState:
    def __init__(self):
        self.model = FakeModel()
        self.run_id = "fake_run"


class FakeModelWrapper:
    def __init__(self):
        self.features = [
            "race", "sex", "native_country",
            "occupation", "education", "capital_gain"
        ]
        self._state = FakeState()

    def get(self):
        return self._state

    def set(self, run_id):
        self._state.run_id = run_id


@pytest.fixture
def client():
    app = create_app()
    fake_model = FakeModelWrapper()
    app.dependency_overrides[get_model] = lambda: fake_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def client_no_model():
    app = create_app()

    class EmptyWrapper:
        features = ["race"]

        def get(self):
            class State:
                model = None
                run_id = "none"
            return State()

        def set(self, run_id):
            pass

    app.dependency_overrides[get_model] = lambda: EmptyWrapper()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def valid_payload():
    return {
        "race": "A",
        "sex": "M",
        "native_country": "US",
        "occupation": "IT",
        "education": "Bachelors",
        "capital_gain": 0
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["run_id"] == "fake_run"


def test_predict_success(client):
    r = client.post("/predict", json=valid_payload())
    assert r.status_code == 200
    assert r.json()["prediction"] == 1


def test_predict_missing_columns(client):
    r = client.post("/predict", json={"race": "A"})
    assert r.status_code in (404, 422)


def test_predict_model_not_loaded(client_no_model):
    r = client_no_model.post("/predict", json=valid_payload())
    assert r.status_code == 503


def test_update_model(client, monkeypatch):
    monkeypatch.setattr(
        "ml_service.config.get_all_runs",
        lambda: [{"run_id": "new_run"}]
    )

    r = client.post("/updateModel", json={"run_id": "new_run"})
    assert r.status_code == 200
    assert r.json()["run_id"] == "new_run"


def test_full_flow(client):
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=valid_payload()).status_code == 200

def test_to_dataframe_success():
    req = PredictRequest(
        age=30,
        race="White",
        sex="Male",
        native_country="US",
        occupation="IT",
        education="Bachelors",
        capital_gain=0,
    )

    df = to_dataframe(
        req,
        needed_columns=[
            "age", "race", "sex",
            "native.country", "occupation",
            "education", "capital.gain"
        ]
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_to_dataframe_missing_columns():
    req = PredictRequest(age=30)

    with pytest.raises(MissingColumnsError):
        to_dataframe(req, needed_columns=["age", "race"])


# Если с . вместо _
def test_to_dataframe_alias_columns():
    req = PredictRequest(
        age=30,
        capital_gain=100
    )

    df = to_dataframe(
        req,
        needed_columns=["age", "capital.gain"]
    )

    assert "capital.gain" in df.columns
    assert df.shape[0] == 1