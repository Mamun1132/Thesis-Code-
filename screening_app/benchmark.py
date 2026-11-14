"""Performance benchmark of the screening system. Writes results to benchmark_results.json."""
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from app import create_app
from screening.pipeline import FEATURES, MODEL_PATH, load_clean
from screening.predictor import Screener

ROOT = Path(__file__).resolve().parent
SAMPLE = {
    "Age": 27, "Systolic BP": 150, "Diastolic": 110, "BS": 11, "Body Temp": 98, "BMI": 25,
    "Previous Complications": 0, "Preexisting Diabetes": 1, "Gestational Diabetes": 0,
    "Mental Health": 1, "Heart Rate": 85,
}


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def summary(ms):
    return {"n": len(ms), "mean_ms": round(statistics.mean(ms), 1), "median_ms": round(pct(ms, 50), 1),
            "p95_ms": round(pct(ms, 95), 1), "max_ms": round(max(ms), 1)}


def main():
    out = {"cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    out["model_file_mb"] = round(MODEL_PATH.stat().st_size / 1e6, 2)

    t0 = time.perf_counter()
    screener = Screener()
    out["model_load_s"] = round(time.perf_counter() - t0, 2)

    X, _ = load_clean()
    t0 = time.perf_counter()
    screener.predict_proba(X)
    out["batch_1166_records_s"] = round(time.perf_counter() - t0, 3)

    client = create_app(screener=screener, enable_log=False).test_client()
    for _ in range(20):
        client.post("/api/predict", json=SAMPLE)
    ms = []
    for _ in range(1000):
        t0 = time.perf_counter()
        r = client.post("/api/predict", json=SAMPLE)
        ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    out["in_process_api"] = summary(ms)

    # Real HTTP requests to a running server on localhost
    env = dict(os.environ, PORT="5057", SCREENING_LOG=str(ROOT / "logs" / "benchmark_log.csv"))
    server = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = "http://127.0.0.1:5057/api/predict"
        data = json.dumps(SAMPLE).encode()
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:5057/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)

        def call():
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
                ok = resp.status == 200
            return (time.perf_counter() - t) * 1000, ok

        for _ in range(10):
            call()
        seq = [call() for _ in range(300)]
        out["http_sequential"] = summary([m for m, _ in seq])
        out["http_sequential"]["errors"] = sum(not ok for _, ok in seq)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            res = list(pool.map(lambda _: call(), range(500)))
        wall = time.perf_counter() - t0
        out["http_10_concurrent_users"] = summary([m for m, _ in res])
        out["http_10_concurrent_users"]["errors"] = sum(not ok for _, ok in res)
        out["http_10_concurrent_users"]["throughput_req_per_s"] = round(len(res) / wall, 1)
    finally:
        server.terminate()
        server.wait()
        (ROOT / "logs" / "benchmark_log.csv").unlink(missing_ok=True)

    out["peak_memory_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (ROOT / "benchmark_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
