#!/bin/bash

# Start pgvector in detached mode
docker compose up -d

# Postgres R2R settings for backend to connect
export R2R_POSTGRES_USER="r00t"
export R2R_POSTGRES_PASSWORD="t00r"
export R2R_POSTGRES_HOST="localhost"
export R2R_POSTGRES_PORT=6666
export R2R_POSTGRES_DBNAME="r2r"
export R2R_PROJECT_NAME="vector_store"

export R2R_HOSTNAME="http://localhost:7272"

# Run R2R locally
# The RESTful API is accessible at:  http://localhost:7272 
r2r serve --config-path=r2r/config.toml