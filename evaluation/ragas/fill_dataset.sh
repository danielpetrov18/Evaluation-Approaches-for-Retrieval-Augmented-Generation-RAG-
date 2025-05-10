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
    pip3 install "r2r[core]==3.5.11" langchain==0.3.25 langchain-community==0.3.23 "unstructured[md]"==0.17.2
    # Create a marker file to indicate dependencies are installed
    touch r2r_venv/.dependencies_installed
    echo "Dependencies installed."
else
    echo "Dependencies already installed. Skipping installation."
fi

echo "Environment is set and ready to be used"

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <goldens-filename> <test-id>"
    exit 1
fi

# Run the python script to extract and save the data
# First argument is where the goldens are
# Second argument represents the test-id configuration
python3 fill_dataset.py "$1" "$2"
echo "Full dataset saved..."