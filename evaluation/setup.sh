#!/usr/bin/bash

# Check if virtual environment `eval` already exists
# This virtual environment will hold all the dependecies required by all the evaluation frameworks
if [ ! -d "eval" ]; then
    echo "[+] CREATING VIRTUAL ENVIRONMENT... [+]"
    python3 -m venv eval
    echo "[+] VIRTUAL ENVIRONMENT CREATED. [+]"
else
    echo "[+] VIRTUAL ENVIRONMENT ALREADY EXISTS. SKIPPING CREATION. [+]"
fi

# Activate the virtual environment
source eval/bin/activate

echo "[+] INSTALLING DEPENDENCIES... [+]"

# Install dependencies
pip3 install --upgrade pip
pip3 install --upgrade -r requirements.txt

echo "[+] DEPENDENCIES INSTALLED. SELECT `eval` AS KERNEL. [+]"