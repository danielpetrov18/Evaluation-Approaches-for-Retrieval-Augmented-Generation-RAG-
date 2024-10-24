#!/bin/bash

# Make sure docker is available.
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

# Load environment variables from the .env file if present. If not stop execution.
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found. Please create one and try again."
  exit 1
fi

# Start pgvector and unstructured in detached mode.
docker compose up -d --build

# Check if Ollama is already installed
if [ -f "/usr/local/bin/ollama" ]; then
  echo "Ollama is already installed, skipping download"
else
  # Download Ollama
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Download required models for LLM and embeddings.
ollama pull llama3.1
ollama pull mxbai-embed-large

# Run R2R locally.
# The RESTful API is accessible at:  http://localhost:7272.
# However, one can use the /backend/r2r_backend.py for easier work.
r2r serve &

sleep 2

# Wait for R2R to be ready.
while ! curl -s http://localhost:7272/health > /dev/null; do
  sleep 1 
done

# Wait for r2r to boot up.
sleep 8 

# Make create_index.py executable and run it
chmod u+x ./script/create_index.py
python3 ./script/create_index.py

# Bring back the r2r service to the foreground. One can observe the logs in the terminal.
fg