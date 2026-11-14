#!/usr/bin/env bash
# One-click start for macOS and Linux: sets up packages, trains the model and opens the app.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10 or newer from https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating a virtual environment..."
  python3 -m venv .venv
fi
PY=".venv/bin/python"

if [ ! -f ".venv/installed.txt" ]; then
  echo "Installing packages (needs internet the first time)..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
  echo ok > .venv/installed.txt
fi

echo "Training the model on the Kurigram data..."
"$PY" train_model.py

echo "Starting the app at http://127.0.0.1:5000 (press Ctrl+C to stop)"
( sleep 3; (command -v open >/dev/null && open http://127.0.0.1:5000) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:5000) || true ) &
"$PY" app.py
