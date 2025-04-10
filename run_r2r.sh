#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting initialization script..."

# Export environment variables
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

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null
then
    echo "Ollama is not running. Exiting."
    exit 1
else
    echo "Ollama is running. Checking models..."

    # Check if llama3.1 is available
    if ! ollama list | grep -q "llama3.1"
    then
        echo "Downloading llama3.1 model..."
        ollama pull llama3.1
    fi

    # Check if embedding model is available
    if ! ollama list | grep -q "mxbai-embed-large"
    then
        echo "Downloading embedding model..."
        ollama pull mxbai-embed-large
    fi

    # Set num_ctx to 16000/24000 for llama3.1
    # https://r2r-docs.sciphi.ai/self-hosting/local-rag
    echo "Setting context window for llama3.1 to 24000 tokens..."
    echo -e 'FROM llama3.1\nPARAMETER num_ctx 24000' > Modelfile
    ollama create llama3.1 -f Modelfile
    echo "Model configuration complete."
fi

# Add nvidia container toolkit
# https://huggingface.co/docs/text-embeddings-inference/quick_tour
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Check if Docker is running
if ! docker info > /dev/null 2>&1
then
    echo "Docker is not running. Exiting."
    exit 1
else
    echo "Docker is running. Starting containers with docker-compose..."
    docker compose up -d
fi

# Check for Python virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]
then
    echo "Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade tools
python3 -m pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing/updating required packages..."
if [ -f "requirements.txt" ]
then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping package installation."
    exit 1
fi

# Start the application
echo "Starting application..."
python3 -m r2r.serve
