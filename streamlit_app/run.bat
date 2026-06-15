@echo off
REM Streamlit App Launcher for Windows

echo.
echo ======================================
echo  Breast Cancer Detection Streamlit App
echo ======================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
pip list | findstr /i streamlit >nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run Streamlit app
echo.
echo Starting Streamlit app...
echo Opening browser at http://localhost:8501
echo.
streamlit run app.py

pause
