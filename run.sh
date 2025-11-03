#!/bin/bash

echo "--- This is DEBUG script v2 ---"
echo "--- We are now testing the streamlit command directly ---"

echo "Listing files in the current directory:"
ls -l

echo "Installing ONLY streamlit..."
pip install streamlit

echo "Attempting to start Streamlit app..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0

echo "--- Script finished ---"
