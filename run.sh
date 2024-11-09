#!/bin/bash

if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found. Please create one and try again."
  exit 1
fi

echo "[+] ENVIRONMENT VARIABLES EXPORTED [+]"

if which ollama >/dev/null 2>&1; then
  echo "Ollama is already installed, skipping download."
else
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling OLLAMA models ..."
ollama pull "$OLLAMA_CHAT_MODEL"
ollama pull "$OLLAMA_EMBEDDING_MODEL"

echo "[+] OLLAMA IS READY [+]"

docker compose up -d

python3 ./backend/config_processor.py
if [ $? -ne 0 ]; then
  echo "Error processing config template. Please check your environment variables."
  exit 1
fi

r2r serve