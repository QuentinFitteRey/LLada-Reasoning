#!/bin/bash

# Load the module
module load ollama

# Set environment variables
export OLLAMA_CONTEXT_LENGTH=40960
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_MODELS="/home/hice1/yluo432/scratch/hf_cache/ollama_cache"
export CUDA_VISIBLE_DEVICES=4

# Start the Ollama server
ollama serve
