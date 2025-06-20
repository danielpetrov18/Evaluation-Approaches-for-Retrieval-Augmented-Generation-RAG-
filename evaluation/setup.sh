#!/usr/bin/bash

if [ ! -d "eval" ]; then
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

cd ..

python3 -m jupyterlab