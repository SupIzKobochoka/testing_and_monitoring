import pandas as pd

from ml_service.schemas import PredictRequest

class MissingColumnsError(Exception):
    pass

FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = [
        column for column in needed_columns if column in FEATURE_COLUMNS
    ] if needed_columns is not None else FEATURE_COLUMNS
    columns_without_dot = [col.replace('.', '_') for col in columns]
    
    missing_collumns = [col for col in columns_without_dot
                        if col not in req.model_dump(exclude_none=True).keys()]
    if missing_collumns:
        raise MissingColumnsError(f'missing {missing_collumns} values')

    row = [getattr(req, column) for column in columns_without_dot]
    return pd.DataFrame([row], columns=columns)
