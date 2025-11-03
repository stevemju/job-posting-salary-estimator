#!/bin/bash

echo "--- This is the DEBUG run.sh script ---"
echo "--- We are skipping the Ollama installation to test the Streamlit app ---"

echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Starting Streamlit app..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```