#!/usr/bin/bash

# Check if virtual environment `evaluation` already exists
if [ ! -d "eval" ]; then
    echo "Creating virtual environment..."
    python3 -m venv eval
    echo "Virtual environment created."
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate the virtual environment
source eval/bin/activate

# Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Environment is set and ready to be used. In your notebook select the kernel 'eval'."