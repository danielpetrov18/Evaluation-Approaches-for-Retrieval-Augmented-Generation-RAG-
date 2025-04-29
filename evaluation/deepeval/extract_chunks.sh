#!/usr/bin/bash

# Check if the data directory already exists
if [ ! -d "data" ]; then
    echo "Downloading data..."
    git clone https://huggingface.co/datasets/explodinggradients/ragas-airline-dataset data
    echo "Downloaded data from GitHub"
else
    echo "Data directory already exists. Skipping download."
fi

# Check if the virtual environment already exists
if [ ! -d "r2r_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv r2r_venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate
source r2r_venv/bin/activate

# Check if dependencies need to be installed
if [ ! -f "r2r_venv/.dependencies_installed" ]; then
    echo "Installing dependencies..."
    # ========= THIS VERSION NEEDS TO BE EQUIVALENT TO THE ONE IN THE /project FOLDER =========
    pip3 install "r2r[core]==3.5.11"
    # Create a marker file to indicate dependencies are installed
    touch r2r_venv/.dependencies_installed
    echo "Dependencies installed."
else
    echo "Dependencies already installed. Skipping installation."
fi

echo "Environment is set and ready to be used"

# Run the python script to extract and save the data
python3 extract_chunks.py chunks
echo "Data extracted and saved to chunks.json"