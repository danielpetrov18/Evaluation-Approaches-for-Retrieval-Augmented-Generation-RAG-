#!/bin/bash

if [ ! -f .env ]; then
  echo ".env file not found! Exiting."
  exit 1
fi

if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker and try again."
  exit 1
fi

# Load environment variables from the .env file
set -o allexport
source .env
set -o allexport

# Postgres R2R settings for backend to connect
export R2R_POSTGRES_USER="r00t"
export R2R_POSTGRES_PASSWORD="t00r"
export R2R_POSTGRES_HOST="localhost"
export R2R_POSTGRES_PORT=5432
export R2R_POSTGRES_DBNAME="r2r"
export R2R_PROJECT_NAME="vector_store"

# Additional R2R settings
export R2R_CONFIG_PATH="./r2r/config.toml"
export R2R_HOSTNAME="http://localhost:7272"
export TELEMETRY_ENABLED=false

export OLLAMA_API_BASE="http://localhost:11434"

export SCARF_NO_ANALYTICS="true"
export UNSTRUCTURED_LOCAL_URL="http://localhost:7275"
export UNSTRUCTURED_API_URL="https://api.unstructured.io/general/v0/general"

# Start pgvector, ollama and unstructured in detached mode
docker compose up -d --build

# Run R2R locally
# The RESTful API is accessible at:  http://localhost:7272 
r2r serve