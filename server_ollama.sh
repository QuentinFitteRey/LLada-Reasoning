#!/bin/bash

# Load the module
module load ollama

# Set environment variables
export OLLAMA_HOST="127.0.0.1:11435"
export CUDA_VISIBLE_DEVICES=4

# Start the Ollama server
ollama serve
