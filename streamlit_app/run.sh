#!/bin/bash

echo ""
echo "======================================"
echo "  Breast Cancer Detection Streamlit App"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! pip list | grep -i streamlit > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run Streamlit app
echo ""
echo "Starting Streamlit app..."
echo "Opening browser at http://localhost:8501"
echo ""
streamlit run app.py
