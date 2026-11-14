import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from screening.predictor import Screener  # noqa: E402

VALID = {
    "Age": 27, "Systolic BP": 150, "Diastolic": 110, "BS": 11, "Body Temp": 98, "BMI": 25,
    "Previous Complications": 0, "Preexisting Diabetes": 1, "Gestational Diabetes": 0,
    "Mental Health": 1, "Heart Rate": 85,
}


@pytest.fixture(scope="session")
def screener():
    return Screener()


@pytest.fixture()
def client(screener, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "predictions.csv")
    return create_app(screener=screener, enable_log=True).test_client()


@pytest.fixture()
def valid():
    return dict(VALID)
