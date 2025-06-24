#!/usr/bin/bash

# Exit immediately on error, undefined variable, or failed pipe segment
set -e

echo "[+] STARTING INITIALIZATION SCRIPT [+]"

# Load all .env files from env directory
echo "[+] SETTING ENVIRONMENT VARIABLES [+]"
for env_file in env/*.env; do
    if [ -f "$env_file" ]; then
        echo "    [+] LOADING ENV FILE: $env_file"
        set -a
        source "$env_file"
        set +a
    fi
done

# You can modify the environment variables on your device as needed.
# For example if you have a powerful GPU you can use `OLLAMA_NUM_PARALLEL` to increase parallelism.
# https://github.com/ollama/ollama/blob/05a01fdecbf9077613c57874b3f8eb7919f76527/envconfig/config.go#L258
# 
# Do note that the service will be started in the background.
# To stop it run: `ps aux | grep ollama`
# Identify the process ID (PID) and run `kill <PID>` to stop it.
OLLAMA_MODELS="${HOME}/.ollama/models" \
OLLAMA_HOST="0.0.0.0:11434" \
OLLAMA_KEEP_ALIVE="1h" \
OLLAMA_CONTEXT_LENGTH="${LLM_CONTEXT_WINDOW_TOKENS:-16000}" \
ollama serve &

# Wait for Ollama server to be ready
echo "[*] Waiting for Ollama server to become available..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
    sleep 1
done
echo "[✓] Ollama server is ready."

# Define required models
declare -A MODELS=(
    ["llama3.1:8b"]="llama3.1:8b"
    ["deepseek-r1:7b"]="deepseek-r1:7b"
    ["llama3.1:8b-instruct-q4_1"]="llama3.1:8b-instruct-q4_1"
)

# Add dynamic models if specified
MODELS["${CHAT_MODEL:-}"]="${CHAT_MODEL:-}"
MODELS["${EMBEDDING_MODEL:-}"]="${EMBEDDING_MODEL:-}"

# Check and pull models if missing
echo "[*] Checking required models..."
for MODEL in "${!MODELS[@]}"; do
    if [[ -n "$MODEL" ]] && ! ollama list | awk '{print $1}' | grep -q "^$MODEL$"; then
        echo "    [+] Model '$MODEL' not found. Pulling..."
        ollama pull "${MODELS[$MODEL]}"
    else
        echo "    [✓] Model '$MODEL' is available."
    fi
done

# Uncomment the following lines if you want to use the re-ranker with a GPU.
# Add NVIDIA container toolkit only if not already installed.
# https://huggingface.co/docs/text-embeddings-inference/quick_tour
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# This section would be relevant for the re-ranker component in the RAG pipeline
# if ! command -v nvidia-ctk >/dev/null 2>&1; then
#   echo "[+] NVIDIA CONTAINER TOOLKIT NOT INSTALLED. INSTALLING... [+]"

#   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

#   sudo apt-get update
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker
# else
#   echo "[+] NVIDIA CONTAINER TOOLKIT ALREADY INSTALLED. SKIPPING... [+]"
# fi

# Since docker takes the bindmount literally as a folder make sure you create a file if it doesn't exist
if [ ! -f project/.langsearch_key ]; then
    touch project/.langsearch_key
fi

# Check if Docker is running
# This will start the containers in the background
if ! docker info > /dev/null 2>&1
then
    echo "[-] DOCKER IS NOT RUNNING. EXITING... [-]"
    exit 1
else
    echo "[+] DOCKER IS RUNNING. STARTING DOCKER COMPOSE... [+]"
    docker compose up --detach --build
fi

echo "[+] INITIALIZATION SCRIPT COMPLETED. [+]"
echo "[+] TO ACCESS THE DOCKER LOGS ENTER: docker compose logs -f [+]"