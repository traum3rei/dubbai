FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    ffmpeg \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Upgrade pip and install required libs
RUN pip install --upgrade pip setuptools wheel

# Remove Whisper installation - now using standalone WhisperX service
# RUN pip3 install -U openai-whisper

RUN pip3 install jupyter
RUN pip install gptqmodel
RUN pip install gradio_client
RUN pip install requests  # For communicating with WhisperX API
RUN pip install moviepy
RUN pip install pydub
RUN pip install openai

# === DeepFilterNet installation (pretrained model usage) ===
RUN pip install deepfilternet

# === Demucs installation (Facebook AI) ===
RUN pip install demucs

RUN pip install transformers 
RUN pip install optimum
RUN pip install accelerate 

RUN pip install librosa matplotlib numpy scipy

# === SenseVoice installation ===
RUN git clone https://github.com/FunAudioLLM/SenseVoice.git /SenseVoice
WORKDIR /SenseVoice
RUN pip install -r requirements.txt

WORKDIR /DubbAI
RUN mkdir -p input output models

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
