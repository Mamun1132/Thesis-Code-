@echo off
REM One-click start for Windows: sets up Python packages, trains the model and opens the app.
cd /d "%~dp0"
title Maternal Health Risk Screening

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation. Then run this file again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment, please wait...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

set "PY=.venv\Scripts\python.exe"

if not exist ".venv\installed.txt" (
    echo Installing packages. The first time needs internet and can take a few minutes...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Package installation failed. Check your internet connection and run this file again.
        pause
        exit /b 1
    )
    echo ok> ".venv\installed.txt"
)

echo Training the model on the Kurigram data...
"%PY%" train_model.py
if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
)

echo.
echo The app is starting. Your browser will open http://127.0.0.1:5000
echo Keep this window open while you use the app. Close it to stop the app.
echo.
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5000"
"%PY%" app.py
pause
