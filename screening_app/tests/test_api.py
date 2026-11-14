import csv

import app as app_module


def test_form_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Check risk" in r.data


def test_form_submit_shows_result(client, valid):
    r = client.post("/", data={k: str(v) for k, v in valid.items()})
    assert r.status_code == 200 and b"High risk" in r.data


def test_form_errors_are_shown(client, valid):
    r = client.post("/", data={**{k: str(v) for k, v in valid.items()}, "Age": "325"})
    assert r.status_code == 422 and b"must be between 10 and 60" in r.data


def test_api_success_shape(client, valid):
    body = client.post("/api/predict", json=valid).get_json()
    assert set(body) == {"probability", "risk", "threshold", "factors", "warnings"}
    assert 0 <= body["probability"] <= 1 and body["risk"] in {"High", "Low"}


def test_api_rejects_non_json(client):
    assert client.post("/api/predict", data="hello", content_type="text/plain").status_code == 400


def test_api_rejects_json_list(client):
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 400


def test_api_reports_field_errors(client, valid):
    r = client.post("/api/predict", json=dict(valid, BS="high"))
    assert r.status_code == 422 and "BS" in r.get_json()["fields"]


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["training_records"] == 1166


def test_log_is_anonymous(client, valid):
    client.post("/api/predict", json=dict(valid, name="Test Patient", phone="0170000000"))
    with open(app_module.LOG_PATH) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert "name" not in header and "phone" not in header
    assert "Test Patient" not in "".join(rows[1])
