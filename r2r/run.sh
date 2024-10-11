#!/bin/bash

# Postgres R2R
export POSTGRES_USER="admin"
export POSTGRES_PASSWORD="admin007"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT=5432
export POSTGRES_DBNAME="ragdb"
export R2R_PROJECT_NAME="vector_store"

export R2R_HOSTNAME="http://localhost:7272"

# Run R2R locally
# The RESTful API is accessible at:  http://localhost:7272 
r2r serve --config-path=config.toml