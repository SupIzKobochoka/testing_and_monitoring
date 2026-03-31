import os
import requests

BASE_URL = "http://158.160.2.37:5000/api/2.0/mlflow"
EXPERIMENT_ID = "37"
MODEL_ARTIFACT_PATH = 'model'

class URINotFoundError(Exception):
    pass

def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id

def get_all_runs() -> list[dict]:
    '''
    Returns [{run_id: ..., run_name: ...}, ...]
    '''
    url = f"{BASE_URL}/runs/search"
    runs, token = [], None

    while True:
        r = requests.post(url, json={
            "experiment_ids": [EXPERIMENT_ID],
            "max_results": 1000,
            **({"page_token": token} if token else {})
        }).json()

        runs += [{"run_id": x["info"]["run_id"], "name": x["info"]["run_name"]} for x in r.get("runs", [])]
        token = r.get("next_page_token")
        if not token:
            return runs