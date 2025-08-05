# Use the base image
FROM fishaudio/fish-speech:latest

# Accept the HUGGINGFACE_TOKEN as a build argument
ARG HUGGINGFACE_TOKEN

# Install huggingface_hub and other dependencies
RUN pip install huggingface_hub supervisor

# Create necessary directories
RUN mkdir -p checkpoints/openaudio-s1-mini /etc/supervisor/conf.d /var/log

# Create a script to download the model with better error handling
RUN echo '#!/bin/bash\n\
set -euo pipefail  # Exit on error, undefined vars, pipe failures\n\
set -x  # Enable debugging\n\
\n\
if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then\n\
  echo "Error: HUGGINGFACE_TOKEN environment variable is not set."\n\
  exit 1\n\
fi\n\
\n\
echo "Logging into Hugging Face..."\n\
huggingface-cli login --token "$HUGGINGFACE_TOKEN"\n\
\n\
echo "Downloading model..."\n\
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini\n\
\n\
# Verify download completed\n\
echo "Verifying download..."\n\
if [ ! -f "checkpoints/openaudio-s1-mini/codec.pth" ]; then\n\
  echo "Error: codec.pth not found after download"\n\
  exit 1\n\
fi\n\
\n\
echo "Download verification complete"\n\
ls -la checkpoints/openaudio-s1-mini\n\
' > /download_model.sh && chmod +x /download_model.sh

# Download the model during build (only if token is provided)
RUN if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then /download_model.sh; fi

# Create startup script that handles both download and service startup
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
set -x\n\
\n\
# Function to check if model is downloaded\n\
check_model() {\n\
    if [ -f "checkpoints/openaudio-s1-mini/codec.pth" ]; then\n\
        echo "Model found, proceeding..."\n\
        return 0\n\
    else\n\
        echo "Model not found, need to download..."\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Download model if not present\n\
if ! check_model; then\n\
    echo "Downloading model at runtime..."\n\
    /download_model.sh\n\
    \n\
    # Verify download again\n\
    if ! check_model; then\n\
        echo "Model download failed!"\n\
        exit 1\n\
    fi\n\
fi\n\
\n\
echo "Starting services..."\n\
\n\
# Start the API server in background\n\
python -m tools.api_server \\\n\
    --listen 0.0.0.0:7859 \\\n\
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \\\n\
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \\\n\
    --decoder-config-name modded_dac_vq &\n\
\n\
API_PID=$!\n\
\n\
# Give API server time to start\n\
sleep 5\n\
\n\
# Start the WebUI\n\
python -m tools.run_webui \\\n\
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \\\n\
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \\\n\
    --decoder-config-name modded_dac_vq &\n\
\n\
WEBUI_PID=$!\n\
\n\
# Function to cleanup on exit\n\
cleanup() {\n\
    echo "Shutting down services..."\n\
    kill $API_PID $WEBUI_PID 2>/dev/null || true\n\
    wait $API_PID $WEBUI_PID 2>/dev/null || true\n\
}\n\
\n\
# Set trap for cleanup\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Wait for both processes\n\
wait $API_PID $WEBUI_PID\n\
' > /start_services.sh && chmod +x /start_services.sh

# Create supervisord main config
RUN echo '[supervisord]\n\
nodaemon=true\n\
user=root\n\
logfile=/var/log/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[unix_http_server]\n\
file=/var/run/supervisor.sock\n\
\n\
[rpcinterface:supervisor]\n\
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface\n\
\n\
[supervisorctl]\n\
serverurl=unix:///var/run/supervisor.sock\n\
\n\
[include]\n\
files = /etc/supervisor/conf.d/*.conf\n\
' > /etc/supervisor/supervisord.conf

# Create supervisor configuration for fish-speech
RUN echo '[program:api_server]\n\
command=python -m tools.api_server --listen 0.0.0.0:7859 --llama-checkpoint-path checkpoints/openaudio-s1-mini --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth --decoder-config-name modded_dac_vq\n\
directory=/\n\
autorestart=true\n\
stderr_logfile=/var/log/api_server.err.log\n\
stdout_logfile=/var/log/api_server.out.log\n\
user=root\n\
\n\
[program:webui]\n\
command=python -m tools.run_webui --llama-checkpoint-path checkpoints/openaudio-s1-mini --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth --decoder-config-name modded_dac_vq\n\
directory=/\n\
autorestart=true\n\
stderr_logfile=/var/log/webui.err.log\n\
stdout_logfile=/var/log/webui.out.log\n\
user=root\n\
' > /etc/supervisor/conf.d/fish-speech.conf

# Create a health check script
RUN echo '#!/bin/bash\n\
# Check if API server is responding\n\
curl -f -s http://localhost:7859/ >/dev/null 2>&1 || exit 1\n\
' > /health_check.sh && chmod +x /health_check.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD /health_check.sh

# Expose ports
EXPOSE 7859

# Environment variable to control startup method
ENV STARTUP_METHOD=simple

# Startup script that chooses the method
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Ensure model is downloaded\n\
if [ ! -f "checkpoints/openaudio-s1-mini/codec.pth" ] && [ -n "${HUGGINGFACE_TOKEN:-}" ]; then\n\
    echo "Downloading model..."\n\
    /download_model.sh\n\
fi\n\
\n\
# Check if model exists\n\
if [ ! -f "checkpoints/openaudio-s1-mini/codec.pth" ]; then\n\
    echo "Error: Model not found and no HUGGINGFACE_TOKEN provided"\n\
    echo "Please provide HUGGINGFACE_TOKEN environment variable"\n\
    exit 1\n\
fi\n\
\n\
case "${STARTUP_METHOD:-simple}" in\n\
    "supervisor")\n\
        echo "Starting with supervisor..."\n\
        exec supervisord -c /etc/supervisor/supervisord.conf\n\
        ;;\n\
    *)\n\
        echo "Starting with simple method..."\n\
        exec /start_services.sh\n\
        ;;\n\
esac\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
