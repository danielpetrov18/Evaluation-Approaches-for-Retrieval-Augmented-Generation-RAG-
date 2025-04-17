#!/usr/bin/bash

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
    pip3 install "r2r[core]==3.5.11"
    # Create a marker file to indicate dependencies are installed
    touch r2r_venv/.dependencies_installed
    echo "Dependencies installed."
else
    echo "Dependencies already installed. Skipping installation."
fi

echo "Environment is set and ready to be used"

# Run the python script to extract and save the data
python3 fill_dataset.py "$@" # The "$@" is used to pass the arguments to the script
echo "Full dataset saved..."