#!/bin/bash

if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

if which ollama >/dev/null 2>&1; then
  echo "Ollama is already installed, skipping download."
else
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling OLLAMA models ..."

ollama pull llama3.1
ollama pull mxbai-embed-large

echo "[+] OLLAMA IS READY [+]"

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found. Please create one and try again."
  exit 1
fi

echo "[+] ENVIRONMENT VARIABLES EXPORTED [+]"

docker compose up -d

echo "[+] DOCKER CONTAINERS STARTED [+]"

r2r serve