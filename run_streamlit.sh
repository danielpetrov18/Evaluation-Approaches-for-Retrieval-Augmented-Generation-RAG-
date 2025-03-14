#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting streamlit ..."

# Check for Python virtual environment
VENV_DIR="venv"

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Setting environment variables..."
for env_file in env/*.env
do
    if [ -f "$env_file" ]
    then
        echo "Loading variables from $env_file"
        set -a # automatically export all variables
        source "$env_file"
        set +a
    fi
done

cd frontend || { echo "Error: Directory not found!"; exit 1; }

streamlit run st_app.py
