# Maternal Health Risk Screening App

This folder is the working system described in Section 4.4 of the FYDP report. It reads the dataset from `../data/maternal_health_risk_kurigram.csv` in this repository.

A small web application that turns the Random Forest model from the FYDP report
"Machine Learning Model for Accurate Prediction of Maternal Health Risk" into a
working screening tool. A health worker enters the 11 routine antenatal values;
the app returns a risk score (the model output on a 0 to 100 scale, not a calibrated probability), a High/Low risk class at the screening
threshold of 0.3 and the inputs that raised the predicted risk most.

The tool supports clinical judgement and does not replace it. It was trained on
1,166 records from a single hospital (Kurigram General Hospital, Bangladesh) and
has not been validated on other populations.

## Project layout

```
app.py                 Flask app: web form, JSON API, health check, anonymous log
train_model.py         trains the model and writes models/rf_model.joblib
benchmark.py           latency, throughput, model size and memory benchmark
screening/pipeline.py  data cleaning and model definition (same as the report)
screening/predictor.py input validation, prediction, explanation
templates/, static/    web page
tests/                 pytest suite (46 tests)
docs/                  architecture diagram and screenshot used in Section 4.4 of the report
models/                trained model and metadata (created by train_model.py, not stored in git)
logs/                  predictions.csv (created on first use, no identifiers)
```

## Run

**Windows:** install Python 3.10 or newer from python.org (tick "Add python.exe to PATH"),
then double-click `run_app.bat`. The first run installs the packages (needs internet),
trains the model and opens http://127.0.0.1:5000 in your browser. Keep the black
window open while using the app; close it to stop.

**macOS / Linux:** run `./run_app.sh` in a terminal.

**Manual:**

```bash
pip install -r requirements.txt
python train_model.py      # about 2 seconds; prints 1166 records, 471 high risk
python app.py              # open http://127.0.0.1:5000
```

The run scripts retrain the model at every start, so the saved model always matches
the installed scikit-learn version.

## API

`POST /api/predict` with a JSON object using the dataset column names:

```json
{"Age": 27, "Systolic BP": 150, "Diastolic": 110, "BS": 11, "Body Temp": 98,
 "BMI": 25, "Previous Complications": 0, "Preexisting Diabetes": 1,
 "Gestational Diabetes": 0, "Mental Health": 1, "Heart Rate": 85}
```

Response (200):

```json
{"probability": 0.9975, "risk": "High", "threshold": 0.3,
 "factors": [{"feature": "Pre-existing diabetes", "value": 1.0, "effect": 0.502}, "..."],
 "warnings": []}
```

Invalid input returns 422 with a message per field; a body that is not a JSON
object returns 400. `GET /health` reports the model status.

## Input rules

| Field | Accepted values |
|---|---|
| Age | 10 to 60 years |
| Systolic BP | 70 to 200 mmHg |
| Diastolic | 40 to 140 mmHg, lower than systolic |
| BS (blood sugar) | 3.0 to 19.0 mmol/L |
| Body Temp | 97 to 103 °F |
| BMI | 15 to 37 kg/m², optional (filled with the training median) |
| Heart Rate | 58 to 92 bpm |
| Previous Complications, Preexisting Diabetes, Gestational Diabetes, Mental Health | 0 or 1 |

The ranges are those of the cleaned training data, so the model is never asked
about values it has not seen.

## Test and benchmark

```bash
python -m pytest -q        # 46 tests
python benchmark.py        # writes benchmark_results.json
```
