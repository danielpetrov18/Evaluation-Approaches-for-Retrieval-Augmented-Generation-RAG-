#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "[+] STARTING INITIALIZATION SCRIPT [+]"

# Export environment variables
echo "[+] SETTING ENVIRONMENT VARIABLES [+]"
for env_file in env/*.env
do
    if [ -f "$env_file" ]
    then
        echo "    [+] LOADING ENVIRONMENT VARIABLES FROM $env_file [+]"
        set -a # automatically export all variables
        source "$env_file"
        set +a
    fi
done

# You can modify this on your device as needed
OLLAMA_MODELS="${HOME}/.ollama/models" OLLAMA_HOST="0.0.0.0:11434" OLLAMA_KEEP_ALIVE="1h" OLLAMA_NUM_PARALLEL=4 ollama serve &
echo "[+] OLLAMA SERVER STARTED AT $OLLAMA_HOST. [+]"

# Check if chat model is available
if ! ollama list | grep -q "$CHAT_MODEL"
then
    echo "    [+] DOWNLOADING CHAT MODEL \"$CHAT_MODEL\" [+]"
    ollama pull "$CHAT_MODEL"
fi

# Check if embedding model is available
if ! ollama list | grep -q "$EMBEDDING_MODEL"
then
    echo "    [+] DOWNLOADING EMBEDDING MODEL \"$EMBEDDING_MODEL\" [+]"
    ollama pull "$EMBEDDING_MODEL"
fi

# https://r2r-docs.sciphi.ai/self-hosting/local-rag
echo "Creating Modelfile with context window \"$LLM_CONTEXT_WINDOW_TOKENS\" tokens..."
echo -e "FROM $CHAT_MODEL\nPARAMETER num_ctx $LLM_CONTEXT_WINDOW_TOKENS" > Modelfile
ollama create "$CHAT_MODEL" -f Modelfile
echo "    [+] CHAT MODEL \"$CHAT_MODEL\" CREATED WITH CONTEXT WINDOW \"$LLM_CONTEXT_WINDOW_TOKENS\" TOKENS. [+]"

# Add NVIDIA container toolkit only if not already installed
# https://huggingface.co/docs/text-embeddings-inference/quick_tour
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# This section would be relevant for the re-ranker component in the RAG pipeline
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[+] NVIDIA CONTAINER TOOLKIT NOT INSTALLED. INSTALLING... [+]"

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
else
  echo "[+] NVIDIA CONTAINER TOOLKIT ALREADY INSTALLED. SKIPPING... [+]"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1
then
    echo "[-] DOCKER IS NOT RUNNING. EXITING... [-]"
    exit 1
else
    echo "[+] DOCKER IS RUNNING. STARTING DOCKER COMPOSE... [+]"
    docker compose up --build --detach
fi

echo "[+] INITIALIZATION SCRIPT COMPLETED. [+]"
echo "[+] TO ACCESS THE DOCKER LOGS ENTER: docker compose logs -f [+]"