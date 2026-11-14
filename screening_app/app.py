"""Flask web app for maternal health risk screening.

Routes
  GET  /             form for one antenatal visit
  POST /             validate, predict and show the result
  POST /api/predict  same prediction as JSON (for other systems)
  GET  /health       model status for monitoring
"""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from screening.pipeline import FEATURES, ROOT
from screening.predictor import BINARY, LABELS, NUMERIC_RANGES, OPTIONAL, Screener, ValidationError

LOG_PATH = Path(os.environ.get("SCREENING_LOG", ROOT / "logs" / "predictions.csv"))
_log_lock = threading.Lock()


def log_prediction(values: dict, result: dict) -> None:
    """Append an anonymous record (no names or IDs) for later audit."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with _log_lock, LOG_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc", *FEATURES, "probability", "risk"])
        w.writerow(
            [datetime.now(timezone.utc).isoformat(timespec="seconds")]
            + [values.get(k, "") for k in FEATURES]
            + [result["probability"], result["risk"]]
        )


def create_app(screener: Screener | None = None, enable_log: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["SCREENER"] = screener or Screener()
    app.config["ENABLE_LOG"] = enable_log

    def fields():
        out = []
        for name in FEATURES:
            if name in BINARY:
                out.append({"name": name, "label": LABELS[name], "binary": True})
            else:
                lo, hi, unit = NUMERIC_RANGES[name]
                out.append(
                    {"name": name, "label": LABELS[name], "binary": False, "min": lo, "max": hi,
                     "unit": unit, "optional": name in OPTIONAL}
                )
        return out

    @app.get("/")
    def index():
        return render_template("index.html", fields=fields(), values={}, result=None, errors={})

    @app.post("/")
    def submit():
        values = {k: request.form.get(k, "") for k in FEATURES}
        try:
            result = app.config["SCREENER"].predict(values).as_dict()
        except ValidationError as e:
            return render_template("index.html", fields=fields(), values=values, result=None, errors=e.errors), 422
        if app.config["ENABLE_LOG"]:
            log_prediction(values, result)
        return render_template("index.html", fields=fields(), values=values, result=result, errors={})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        try:
            result = app.config["SCREENER"].predict(payload).as_dict()
        except ValidationError as e:
            return jsonify({"error": "invalid input", "fields": e.errors}), 422
        if app.config["ENABLE_LOG"]:
            log_prediction(payload, result)
        return jsonify(result)

    @app.get("/health")
    def health():
        meta = app.config["SCREENER"].meta
        return jsonify({"status": "ok", "threshold": app.config["SCREENER"].threshold,
                        "training_records": meta.get("training_records"),
                        "trained_at": meta.get("trained_at")})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
