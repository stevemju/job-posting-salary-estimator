#!/bin/bash

# 1. Install and start the Ollama server
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Starting Ollama server in the background..."
ollama serve &
sleep 5

# 2. Pull the LLM model
echo "Pulling the phi3:mini model..."
ollama pull phi3:mini

# 3. Launch the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
