#!/bin/bash
set -euo pipefail
set -x

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "Error: HUGGINGFACE_TOKEN environment variable is not set."
  exit 1
fi

echo "Logging into Hugging Face..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

echo "Downloading model..."
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

echo "Verifying download..."
if [ ! -f "checkpoints/openaudio-s1-mini/codec.pth" ]; then
  echo "Error: codec.pth not found after download"
  exit 1
fi

echo "Download verification complete"
ls -la checkpoints/openaudio-s1-mini
