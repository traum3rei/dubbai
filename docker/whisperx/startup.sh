#!/bin/bash

# Activate virtual environment
source whisperx_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize CUDA memory

# Create directories
mkdir -p uploads output

# Start the server
python whisperx_api.py
