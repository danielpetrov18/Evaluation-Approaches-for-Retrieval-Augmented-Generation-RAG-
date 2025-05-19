#!/usr/bin/bash

# Check if the virtual environment already exists
if [ ! -d "r2r_venv" ]; then
    echo "[+] CREATING VIRTUAL ENVIRONMENT... [+]"
    python3 -m venv r2r_venv
    echo "[+] VIRTUAL ENVIRONMENT CREATED. [+]"
else
    echo "[+] VIRTUAL ENVIRONMENT ALREADY EXISTS. SKIPPING CREATION. [+]"
fi

# Activate
source r2r_venv/bin/activate

echo "[+] INSTALLING DEPENDENCIES... [+]"
pip3 install "r2r[core]==3.5.11" langchain==0.3.25 langchain-community==0.3.23 "unstructured[md]"==0.17.2

echo "[+] ENVIRONMENT IS SET AND READY TO BE USED. [+]"

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "[-] USAGE: $0 <goldens-filename> <test-id> [-]"
    exit 1
fi

# Run the python script to extract and save the data
# First argument is where the goldens are
# Second argument represents the test-id configuration
python3 fill_dataset.py "$1" "$2"
echo "[+] FULL DATASET SAVED... [+]"