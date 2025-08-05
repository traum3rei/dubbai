# web-frontend.Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install flask
RUN pip install flask-wtf
RUN pip install python-dotenv
RUN pip install gradio_client
RUN pip install openai-whisper
RUN pip install demucs
RUN pip install deepfilternet
RUN pip install torch
RUN pip install torchaudio
RUN pip install requests
RUN pip install openai
RUN pip install pydub
RUN pip install moviepy
RUN pip install funasr

# Copy the frontend application
COPY frontend/ .

EXPOSE 5000

CMD ["python", "app.py"]
