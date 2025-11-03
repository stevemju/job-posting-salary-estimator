#!/bin/bash

echo "--- Docker container has started ---"

# 1. Start the Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# 2. Wait a few seconds for the server to be ready
sleep 5

# 3. Pull the LLM model
# This runs INSIDE the container's disk space.
echo "Pulling phi3:mini model..."
ollama pull phi3:mini

# 4. Launch the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
