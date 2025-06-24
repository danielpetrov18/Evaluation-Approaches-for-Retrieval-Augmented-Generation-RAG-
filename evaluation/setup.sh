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

# https://www.comet.com/docs/opik/self-host/local_deployment
if [ ! -d "opik" ]; then
    echo "[+] DOWNLOADING OPIK REPOSITORY FOR LOCAL HOSTING. [+]"
    git clone https://github.com/comet-ml/opik.git
fi

# Run OPIK local instance
cd opik
./opik.sh --debug

./opik.sh --verify

# Root of the project
cd ../..

python3 -m jupyterlab