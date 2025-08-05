# Use a newer base image with PyTorch 2.x
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    tzdata && \
    ln -sf /usr/share/zoneinfo/UTC /etc/localtime && \
    echo "UTC" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./docker/whisperx/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ./docker/whisperx/whisperx_api.py .

# Create directories
RUN mkdir -p uploads output

EXPOSE 5001

CMD ["python", "whisperx_api.py"]
