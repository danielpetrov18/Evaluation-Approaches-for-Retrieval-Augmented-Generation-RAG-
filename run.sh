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

    # Check if chat model is available
    if ! ollama list | grep -q "$CHAT_MODEL"
    then
        echo "Downloading chat model \"$CHAT_MODEL\"..."
        ollama pull "$CHAT_MODEL"
    fi

    # Check if embedding model is available
    if ! ollama list | grep -q "$EMBEDDING_MODEL"
    then
        echo "Downloading embedding model \"$EMBEDDING_MODEL\"..."
        ollama pull "$EMBEDDING_MODEL"
    fi

    # Set num_ctx to 16000-32000 for chat model
    # https://r2r-docs.sciphi.ai/self-hosting/local-rag
    echo "Setting context window for \"$CHAT_MODEL\" to \"$LLM_CONTEXT_WINDOW_TOKENS\" tokens..."
    echo -e "FROM $CHAT_MODEL\nPARAMETER num_ctx $LLM_CONTEXT_WINDOW_TOKENS" > Modelfile
    ollama create "$CHAT_MODEL" -f Modelfile
    echo "Model configuration complete."
fi

# Add NVIDIA container toolkit only if not already installed
# https://huggingface.co/docs/text-embeddings-inference/quick_tour
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# This section would be relevant for the re-ranker component in the RAG pipeline
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "NVIDIA Container Toolkit not found. Installing..."

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
else
  echo "NVIDIA Container Toolkit is already installed. Skipping installation."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1
then
    echo "Docker is not running. Exiting."
    exit 1
else
    echo "Docker is running. Starting containers with docker-compose..."
    docker compose up
fi