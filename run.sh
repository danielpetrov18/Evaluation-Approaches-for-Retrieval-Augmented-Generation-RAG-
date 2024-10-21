#!/bin/bash

# Make sure docker is available.
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

# Load environment variables from the .env file
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found. Please create one and try again."
  exit 1
fi

# Start pgvector, ollama, and unstructured in detached mode
docker compose up -d --build

# Run R2R locally
# The RESTful API is accessible at:  http://localhost:7272 
r2r serve &

R2R_PID=$!  # Capture the PID of the R2R process

# Wait for R2R to be ready
while ! curl -s http://localhost:7272/health > /dev/null; do
  sleep 1 
done

sleep 8 # Wait for r2r to boot up

# Make create_index.py executable and run it
chmod u+x create_index.py
python3 create_index.py

fg