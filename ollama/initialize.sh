#!/bin/bash

# Start the Ollama app
ollama serve &

# Give the Ollama service time to start
sleep 5

# Pull the models after the service has started
ollama pull llama3.1
ollama pull mxbai-embed-large

# Keep the service running
wait