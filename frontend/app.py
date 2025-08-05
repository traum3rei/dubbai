from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
import uuid
from gradio_client import Client, handle_file
import torch
import whisper as whisper # Keep alias for clarity if preferred, but direct whisper is fine
import demucs.separate
from df.enhance import enhance, init_df, load_audio, save_audio
import shlex
import shutil
import json
from openai import OpenAI
import time
from openai import RateLimitError
from threading import Lock
from datetime import datetime
import requests
import numpy as np
from pydub import AudioSegment
from collections import defaultdict
from typing import Optional, Dict
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import re
from difflib import SequenceMatcher

# --- SenseVoice (emotion detection) support ---
try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

sensevoice_model = None

def load_sensevoice_model(model_dir="iic/SenseVoiceSmall"):
    global sensevoice_model
    if AutoModel is None:
        app.logger.warning("SenseVoice not available - funasr not installed")
        return None
    if sensevoice_model is not None:
        app.logger.info("SenseVoice Model loaded")
        return sensevoice_model
    try:
        sensevoice_model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            disable_update=True,
        )
        app.logger.info("SenseVoice model loaded successfully")
        return sensevoice_model
    except Exception as e:
        app.logger.error(f"Failed to load SenseVoice model: {e}")
        return None

def extract_emotions_from_text(text):
    emotion_pattern = r'<\|([^|]+)\|>'
    valid_emotions = [
        'happy', 'sad', 'angry', 'neutral', 'excited', 'surprised', 'disgusted', 'fearful', 'emo_unknown',
        'laughter', 'music', 'applause', 'cough', 'sneeze', 'cry'
    ]
    emotions = []
    for match in re.finditer(emotion_pattern, text):
        tag = match.group(1).lower()
        if tag in valid_emotions:
            emotions.append({'emotion': tag, 'start_pos': match.start(), 'end_pos': match.end()})
    clean_text = re.sub(r'<\|[^|]+\|>', '', text).strip()
    clean_text = ' '.join(clean_text.split())
    return emotions, clean_text

def extract_emotions_with_sensevoice(audio_file_path):
    global sensevoice_model
    if sensevoice_model is None:
        return None
    try:
        res = sensevoice_model.generate(
            input=audio_file_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if res and len(res) > 0:
            raw_text = res[0]["text"]
            processed_text = ' '.join(raw_text.split())
            return processed_text
        else:
            return None
    except Exception as e:
        app.logger.error(f"SenseVoice error: {e}")
        return None

def fuzzy_match_segments_with_emotions(whisper_segments, sensevoice_text):
    if not sensevoice_text:
        for segment in whisper_segments:
            segment['emotion'] = 'none'
        return whisper_segments
    emotions, clean_sensevoice_text = extract_emotions_from_text(sensevoice_text)
    if not emotions:
        for segment in whisper_segments:
            segment['emotion'] = 'none'
        return whisper_segments
    total_whisper_text = ' '.join([seg['text'].strip() for seg in whisper_segments])
    similarity = SequenceMatcher(None, total_whisper_text.lower(), clean_sensevoice_text.lower()).ratio()
    if emotions and similarity > 0.3:
        segments_per_emotion = max(1, len(whisper_segments) // len(emotions))
        emotion_index = 0
        for i, segment in enumerate(whisper_segments):
            if emotion_index < len(emotions):
                segment['emotion'] = emotions[emotion_index]['emotion']
            else:
                segment['emotion'] = 'none'
            if (i + 1) % segments_per_emotion == 0 and emotion_index < len(emotions) - 1:
                emotion_index += 1
    else:
        for segment in whisper_segments:
            segment['emotion'] = 'none'
    return whisper_segments

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/uploads'
app.config['OUTPUT_FOLDER'] = '/app/output'
app.config['STATUS_FOLDER'] = '/app/status'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATUS_FOLDER'], exist_ok=True)

# Thread lock for status updates
status_lock = Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav', 'ogg', 'flac', 'mp4', 'mkv', 'avi'}

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'mkv', 'avi'}

def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def extract_segments(data):
    if "result" in data and "segments" in data["result"]:
        return data["result"]["segments"]
    elif "segments" in data:
        return data["segments"]
    else:
        return []


def separate_audio_from_video(video_path, audio_out_path, muted_video_out_path):
    """Extract audio from video and save both audio and muted video separately."""

    # Ensure output directories exist
    audio_dir = os.path.dirname(audio_out_path)
    muted_video_dir = os.path.dirname(muted_video_out_path)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
    if not os.path.exists(muted_video_dir):
        os.makedirs(muted_video_dir, exist_ok=True)

    # Extract audio (48 kHz mono WAV)
    audio_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        audio_out_path
    ]

    # Remove audio from video
    muted_video_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-an", muted_video_out_path
    ]

    app.logger.info("🔊 Extracting audio...")
    subprocess.run(audio_cmd, check=True)
    app.logger.info(f"✅ Audio saved to: {audio_out_path}")

    app.logger.info("🎥 Saving muted video...")
    subprocess.run(muted_video_cmd, check=True)
    app.logger.info(f"✅ Muted video saved to: {muted_video_out_path}")


def separate_vocals(input_audio_path, output_dir, model='mdx_extra'):
    """Separate vocals and no-vocals with Demucs"""
    app.logger.info(f"Separating vocals from {input_audio_path} using model {model}...")
    # Demucs expects the output to be in a 'separated' subdirectory relative to CWD.
    # We'll run it and then move the files.
    
    # Ensure the current working directory for demucs allows it to create 'separated'
    # For simplicity, change CWD temporarily if needed, or ensure input_audio_path is absolute
    # and demucs handles paths well. If demucs bases its output on CWD:
    original_cwd = os.getcwd()
    # It's safer if demucs can take an output directory argument, but if not:
    # We assume demucs will create ./separated/{model}/{base_name_of_input}
    # Let's ensure the input_audio_path is absolute or demucs is run from a predictive CWD.
    # The shlex.split implies demucs.separate.main directly uses these args.
    # Demucs default output is ./separated/<model_name>/<basename(input_file)>
    
    cmd = f'--two-stems vocals -n {model} "{input_audio_path}"' # removed --out "{output_dir}" as it might conflict
    
    # Check if output_dir for demucs internal working needs to be set
    # The library call demucs.separate.main() will use its default behavior for output,
    # which is to create a 'separated' folder in the current working directory.
    
    demucs.separate.main(shlex.split(cmd))

    base_name_input = os.path.splitext(os.path.basename(input_audio_path))[0]
    # Default Demucs output structure
    demucs_output_base = os.path.join(original_cwd, "separated", model, base_name_input)

    vocals_src_path = os.path.join(demucs_output_base, "vocals.wav")
    no_vocals_src_path = os.path.join(demucs_output_base, "no_vocals.wav")

    os.makedirs(output_dir, exist_ok=True) # This is the final destination for app

    vocals_dst_path = os.path.join(output_dir, f"{base_name_input}_vocals.wav")
    no_vocals_dst_path = os.path.join(output_dir, f"{base_name_input}_no_vocals.wav")
    
    if not os.path.exists(vocals_src_path):
        raise FileNotFoundError(f"Demucs output vocals not found at {vocals_src_path}. CWD: {original_cwd}, Demucs base: {demucs_output_base}")
    if not os.path.exists(no_vocals_src_path):
        raise FileNotFoundError(f"Demucs output no_vocals not found at {no_vocals_src_path}")

    shutil.move(vocals_src_path, vocals_dst_path) # Move instead of copy
    shutil.move(no_vocals_src_path, no_vocals_dst_path) # Move instead of copy
    
    # Try to clean up the demucs 'separated' directory structure if it's empty
    try:
        os.rmdir(demucs_output_base) # remove base_name_input folder
        os.rmdir(os.path.join(original_cwd, "separated", model)) # remove model folder
        os.rmdir(os.path.join(original_cwd, "separated")) # remove separated root
        app.logger.info(f"Cleaned up demucs temporary directory: {demucs_output_base} and its parents if empty.")
    except OSError as e:
        app.logger.info(f"Could not cleanup all demucs temp dirs (this is often fine): {e}")


    app.logger.info(f"Vocals saved to: {vocals_dst_path}")
    app.logger.info(f"No vocals saved to: {no_vocals_dst_path}")
    return vocals_dst_path, no_vocals_dst_path


def convert_to_48k_wav(input_audio_path, output_wav_path):
    """Convert any audio input to 48kHz stereo WAV"""
    app.logger.info(f"Converting {input_audio_path} to 48kHz WAV at {output_wav_path}...")
    subprocess.run([
        'ffmpeg', '-y', '-i', input_audio_path,
        '-ar', '48000',
        '-ac', '2',
        output_wav_path
    ], check=True, capture_output=True) # Added capture_output for cleaner logs
    app.logger.info("Conversion done.")


def denoise_audio(input_wav_path, output_wav_path):
    """Denoise/enhance the WAV audio using DeepFilterNet"""
    app.logger.info(f"Denoising audio {input_wav_path} to {output_wav_path}...")
    model, df_state, _ = init_df()
    audio, _ = load_audio(input_wav_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(output_wav_path, enhanced, df_state.sr())
    app.logger.info("Denoising done.")

def update_status(job_id, status, current_file=None, message=None):
    """Update the processing status for a job"""
    status_file = os.path.join(app.config['STATUS_FOLDER'], f"{job_id}.json")
    with status_lock:
        try:
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            
            data.update({
                'status': status,
                'timestamp': str(datetime.now())
            })
            if current_file:
                data['current_file'] = current_file
            if message:
                data['message'] = message
            
            with open(status_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            app.logger.error(f"Error updating status for {job_id}: {str(e)}")


def get_status(job_id):
    """Get the current status of a job"""
    status_file = os.path.join(app.config['STATUS_FOLDER'], f"{job_id}.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {'status': 'error', 'message': 'Status file corrupted'}
    return {'status': 'not_found'}


def load_whisper_model(model_size="base"):
    """
    Load the Whisper ASR model with specified size.
    Args:
        model_size (str): Size of model to load (tiny, base, small, medium, large, large-v2, large-v3)
    Returns:
        whisper.model.Whisper: Loaded Whisper model
    """
    size_map = { # Mapping for user-friendly names to actual model names if different
        'tiny': 'tiny', 'base': 'base', 'small': 'small',
        'medium': 'medium', 'large': 'large',
        'large-v2': 'large-v2', 'large-v3': 'large-v3'
    }
    actual_size = size_map.get(model_size.lower(), 'base') # Default to 'base'
    
    try:
        app.logger.info(f"Loading Whisper {actual_size} model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(actual_size, device=device)
        app.logger.info(f"Successfully loaded Whisper {actual_size} model on {device}")
        return model
    except Exception as e:
        error_msg = f"Failed to load Whisper model ({actual_size}): {str(e)}"
        app.logger.error(error_msg)
        raise RuntimeError(error_msg)


def transcribe_audio(model, audio_path, language=None):
    """
    Transcribe an audio file using Whisper with explicit processing stages.
    This version strictly uses the input audio_path passed from Flask.
    """
    try:
        app.logger.info(f"Starting transcription for {audio_path}...")
        start_time = datetime.now()
        
        # Ensure the input file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
        
        # Load and process audio
        raw_audio = whisper.load_audio(audio_path)
        app.logger.debug(f"Audio loaded with duration: {len(raw_audio)/16000:.2f}s")  # 16000 is Whisper's sample rate
        
        # Pad/trim to 30 seconds if needed
        processed_audio = whisper.pad_or_trim(raw_audio)
        app.logger.debug("Audio padded/trimmed")
        
        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(processed_audio).to(model.device)
        app.logger.debug("Mel spectrogram created")
        
        # Language detection if not specified
        detected_lang = language
        if not language or language.lower() == 'auto':
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            app.logger.info(f"Detected language: {detected_lang} (confidence: {probs[detected_lang]:.2f})")
        
        # Decoding options
        options = whisper.DecodingOptions(
            language=detected_lang if detected_lang and detected_lang.lower() != 'auto' else None,
            fp16=torch.cuda.is_available(),
            task="transcribe"
        )
        
        # Decode audio
        result = whisper.decode(model, mel, options)
        app.logger.debug("Audio decoded")
        
        duration = datetime.now() - start_time
        app.logger.info(f"Transcription completed in {duration.total_seconds():.2f}s")
        
        if not result.text:
            raise ValueError("Whisper returned empty transcription")
            
        return result.text.strip()
    
    except Exception as e:
        error_msg = f"Transcription failed for {audio_path}: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def transcribe_with_whisperx(audio_file_path, language=None, enable_diarization=True, min_speakers=1, max_speakers=10, whisperx_url="http://whisperx:5001", enable_emotion_detection=False, sensevoice_model_dir="iic/SenseVoiceSmall"):
    """Transcribe audio using WhisperX API service with diarization support."""
    app.logger.info(f"Transcribing audio file with WhisperX: {audio_file_path}")
    
    # Check if WhisperX service is available
    try:
        health_response = requests.get(f"{whisperx_url}/health", timeout=10)
        if health_response.status_code != 200:
            raise Exception("WhisperX service not healthy")
        app.logger.info("✅ WhisperX service is healthy")
    except Exception as e:
        app.logger.info(f"❌ WhisperX service unavailable: {e}")
        return None
    
    # Prepare the request
    with open(audio_file_path, 'rb') as audio_file:
        files = {'audio': audio_file}
        data = {
            'language': language or 'en',
            'diarize': str(enable_diarization).lower(),
            'min_speakers': str(min_speakers),
            'max_speakers': str(max_speakers),
            'format': 'json'
        }
        
        app.logger.info(f"📤 Sending audio to WhisperX API...")
        app.logger.info(f"   Language: {data['language']}")
        app.logger.info(f"   Diarization: {data['diarize']}")
        app.logger.info(f"   Speaker range: {min_speakers}-{max_speakers}")
        
        try:
            response = requests.post(
                f"{whisperx_url}/transcribe",
                files=files,
                data=data,
                timeout=900  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                app.logger.info(f"✅ Transcription complete. Found {len(result.get('segments', []))} segments.")
                
                # Print preview with speaker information if available
                for i, segment in enumerate(result.get('segments', [])[:5]):  # First 5 segments
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    speaker = segment.get('speaker', None)
                    
                    if speaker:
                        app.logger.info(f"[{start:06.1f} --> {end:06.1f}] [{speaker}] {text}")
                    else:
                        app.logger.info(f"[{start:06.1f} --> {end:06.1f}] {text}")
                
                if len(result.get('segments', [])) > 5:
                    app.logger.info("...")
                
                # --- EMOTION DETECTION ---
                if enable_emotion_detection:
                    app.logger.info("🔍 Detecting emotions in audio...")
                    sensevoice_model = load_sensevoice_model(sensevoice_model_dir)
                    if sensevoice_model:
                        sensevoice_text = extract_emotions_with_sensevoice(audio_file_path)
                        if sensevoice_text:
                            app.logger.info(f"😊 Detected emotions: {sensevoice_text}")
                            # Optionally, add emotion info to segments
                            result = fuzzy_match_segments_with_emotions(result.get('segments', []), sensevoice_text)
                        else:
                            app.logger.warning("No emotions detected, proceeding without emotions.")
                    else:
                        app.logger.warning("SenseVoice model not loaded, skipping emotion detection.")
                
                return result
            else:
                app.logger.error(f"❌ WhisperX API error: {response.status_code}")
                app.logger.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            app.logger.error("❌ WhisperX API timeout - file might be too large")
            return None
        except Exception as e:
            app.logger.error(f"❌ Error calling WhisperX API: {e}")
            return None

def rich_transcription_postprocess(text):
    """Basic post-processing for SenseVoice output."""
    text = ' '.join(text.split())
    return text.strip()

def replace_emojis_with_tags(text):
    """Replace emojis with emotion tags and normalize SenseVoice emotion tags."""
    emoji_to_tag = {
         "😊": "happy",
         "😡": "angry", 
         "😔": "sad",
         "😀": "laughter",
         "🎼": "music",
         "👏": "applause",
         "🤧": "cough & sneeze",
         "😭": "cry",
    }
    for emoji, tag in emoji_to_tag.items():
        text = text.replace(emoji, f"<|{tag.upper()}|>")
    emotion_mapping = {
        'HAPPY': 'happy',
        'SAD': 'sad', 
        'ANGRY': 'angry',
        'NEUTRAL': 'neutral',
        'EXCITED': 'excited',
        'SURPRISED': 'surprised',
        'DISGUSTED': 'disgusted',
        'FEARFUL': 'fearful',
        'EMO_UNKNOWN': 'unknown'
    }
    for sensevoice_emotion, normalized_emotion in emotion_mapping.items():
        text = text.replace(f'<|{sensevoice_emotion}|>', f'<|{normalized_emotion}|>')
    return text.strip()

def batch_translate_transcript_json(
    input_json_path,
    output_dir="../output",
    model_name="deepseek/deepseek-chat-v3-0324:free",
    source_lang="English",
    target_lang="German",
    translated_field="translated_text",
    batch_size=20,
    separator="[SEGMENT_BREAK]"
):
    """
    Batch translates segment 'text' fields in a transcript JSON.
    Each batch is translated as a block and split back up by a unique separator.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_KEY")
    )

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find segments list
    if "result" in data and "segments" in data["result"]:
        segments = data["result"]["segments"]
    elif "segments" in data:
        segments = data["segments"]
    else:
        raise ValueError("No segments found in the transcript JSON.")

    prompt = (
        f"You are a precise and helpful {source_lang}-to-{target_lang} translator. "
        f"Translate the following segments, keeping them in order. "
        f"Separate each translation with the tag {separator}. "
        f"Only translate, do not change the text, and do not add extra notes."
    )

    n = len(segments)
    i = 0

    app.logger.info(f"🔤 Starting batch translation: {n} segments, batch size {batch_size}")
    while i < n:
        # Prepare batch
        batch = segments[i:i+batch_size]
        batch_texts = [seg.get("text", "") for seg in batch]
        joined_text = f"\n{separator}\n".join(batch_texts)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": joined_text}
        ]

        app.logger.info(f"🔤 Translating batch {i}-{i+len(batch)-1}...")
        # Rate-limit-safe translation
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=2048  # Can tune higher/lower as needed
                )
                break
            except RateLimitError:
                app.logger.warning("⚠️ Hit rate limit. Sleeping for 65 seconds...")
                time.sleep(65)
            except Exception as e:
                app.logger.warning(f"⚠️ Other error during translation: {e}. Sleeping for 10 seconds...")
                time.sleep(10)

        translated_block = response.choices[0].message.content.strip()
        translated_segments = [t.strip() for t in translated_block.split(separator)]

        # If the number doesn't match, try to fallback gracefully
        if len(translated_segments) != len(batch):
            app.logger.warning(f"⚠️ Mismatch in batch {i}-{i+batch_size}: expected {len(batch)}, got {len(translated_segments)}. Will align what we can.")
        for j, seg in enumerate(batch):
            if j < len(translated_segments):
                seg["text"] = translated_segments[j]
            else:
                seg["text"] = ""  # Fallback

        app.logger.info(f"✅ Translated segments {i} to {i+len(batch)-1}")
        i += batch_size

    # Save to new JSON file
    base_filename = os.path.splitext(os.path.basename(input_json_path))[0]
    translated_filename = f"{base_filename}_batched.json"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, translated_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    app.logger.info(f"✅ Translated transcript saved to {output_path}")
    return output_path


def write_transcription_to_json(result, output_path):
    """Write the WhisperX transcription result to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create output dir
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    app.logger.info(f"✅ JSON transcript saved to: {output_path}")


def write_transcription_to_srt(result, output_path):
    """Convert WhisperX result to SRT format and save."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    srt_content = ""
    for i, segment in enumerate(result.get('segments', []), 1):
        start_time = format_timestamp_srt(segment.get('start', 0))
        end_time = format_timestamp_srt(segment.get('end', 0))
        text = segment.get('text', '').strip()
        speaker = segment.get('speaker', '')
        
        if speaker:
            text = f"[{speaker}] {text}"
        
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✅ SRT transcript saved to: {output_path}")


def format_timestamp_srt(seconds):
    """Format timestamp for SRT format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')


def save_transcription(text, output_path):
    """
    Save transcribed text to a file.
    Args:
        text (str): Transcription text
        output_path (str): Path to save the text file (e.g., /path/to/transcript.txt)
    Returns:
        str: Path where text was saved
    """
    try:
        app.logger.info(f"Saving transcription to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        app.logger.info(f"Successfully saved transcription to {output_path}")
        return output_path
    except Exception as e:
        error_msg = f"Failed to save transcription to {output_path}: {str(e)}"
        app.logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def convert_to_16k_wav(input_audio_path, output_wav_path):
    """
    Convert any audio input to 16kHz mono WAV using ffmpeg.
    """
    app.logger.info(f"Converting {input_audio_path} to 16kHz mono WAV at {output_wav_path}...")
    subprocess.run([
        'ffmpeg', '-y', '-i', input_audio_path,
        '-ar', '16000',
        '-ac', '1',
        output_wav_path
    ], check=True, capture_output=True)
    app.logger.info("16kHz conversion done.")


def synthesize_speech(text, reference_audio_path, output_dir, segment_id):
    """Synthesize speech using Fish Speech via Gradio client."""
    app.logger.info(f"Synthesizing speech. Text: '{text[:50]}...', Reference: {reference_audio_path}, Output Dir: {output_dir}")
    try:
        # --- Ensure reference audio is 16kHz mono WAV ---
        ref_16k_path = os.path.join(output_dir, "reference_16k.wav")
        convert_to_16k_wav(reference_audio_path, ref_16k_path)

        client = Client("http://fish-speech:7860/")
        reference_text = "" # As per notebook's default
        
        app.logger.info(f"JOB Calling Fish Speech API predict...")
        result = client.predict(
            text=text,
            reference_id="",
            reference_audio=handle_file(ref_16k_path),
            reference_text=reference_text,
            max_new_tokens=0, chunk_length=200, top_p=0.5,
            repetition_penalty=1.7, temperature=0.6, seed=0,
            use_memory_cache="on",
            api_name="/partial"
        )
        
        app.logger.info(f"JOB: Fish Speech raw response: {result}")
        
        """wav_path_from_fish_speech = None
        if isinstance(result, str) and result.endswith('.wav'):
            wav_path_from_fish_speech = result
        elif isinstance(result, (tuple, list)) and len(result) > 0:
            if isinstance(result[0], str) and result[0].endswith('.wav'):
                wav_path_from_fish_speech = result[0]
            else:
                for item in result:
                    if isinstance(item, str) and item.endswith('.wav'):
                        wav_path_from_fish_speech = item
                        break
        
        if not wav_path_from_fish_speech:
            raise ValueError(f"JOB: No .wav file path found in Fish Speech response: {result}")
        
        app.logger.info(f"JOB: Fish Speech returned WAV path: {wav_path_from_fish_speech}")

        if not os.path.exists(wav_path_from_fish_speech):
            # This can happen if path is relative to fish-speech container and not accessible here.
            # Or if fish-speech actually returns a URL. handle_file usually downloads it to a temp location.
            # Check if the file downloaded by handle_file is accessible by app.py.
            # If handle_file returns /tmp/gradio/xxxx.wav, that /tmp is inside *this (app.py)* container.
            app.logger.error(f"JOB: Synthesized WAV file NOT FOUND at {wav_path_from_fish_speech}. CWD: {os.getcwd()}")
            # List /tmp/gradio content for debugging if this error occurs often
            try:
                gradio_tmp_dir = os.path.dirname(wav_path_from_fish_speech)
                if "gradio" in gradio_tmp_dir and os.path.exists(gradio_tmp_dir):
                    app.logger.info(f"JOB: Contents of {gradio_tmp_dir}: {os.listdir(gradio_tmp_dir)}")
            except Exception as list_err:
                 app.logger.warning(f"JOB: Could not list Gradio temp dir: {list_err}")
            raise FileNotFoundError(f"Synthesized WAV file not found: {wav_path_from_fish_speech}")
            
        os.makedirs(output_dir, exist_ok=True)
        final_output_wav = os.path.join(output_dir, "output.wav") # Standardized name
        
        app.logger.info(f"JOB: Moving synthesized file from {wav_path_from_fish_speech} to {final_output_wav}")
        shutil.move(wav_path_from_fish_speech, final_output_wav)
        
        app.logger.info(f"JOB: Successfully synthesized speech saved to {final_output_wav}")
        return final_output_wav"""
        if isinstance(result, (tuple, list)) and len(result) > 0:
            if isinstance(result[0], str) and result[0].endswith(".wav"):
                wav_file_path = result[0]
                
                # Ensure the file exists
                if not os.path.exists(wav_file_path):
                    print(f"❌ Generated .wav file not found: {wav_file_path}")
                    return None
                
                # Save with a unique filename using the segment ID
                output_file_path = os.path.join(output_dir, f"{segment_id}.wav")
                shutil.move(wav_file_path, output_file_path)
                print(f"✅ Generated .wav file saved to {output_file_path}")
                return output_file_path
            else:
                print(f"❌ Unexpected result format: {result}")
                return None
        else:
            print(f"❌ Unexpected result format: {result}")
            return None
        
    except Exception as e:
        error_msg = f"JOB: Speech synthesis failed: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e



def process_file_in_background(input_path, output_dir_job, job_id, language, target_lang, model_size, reference_choice, min_speakers=1, max_speakers=10, enable_emotion_detection=False):
    transcribed_text = None
    processing_files = {
        'original': None, 'separated_vocals': None, 'converted_48k': None,
        'denoised': None, 'transcription': None, 'final_output': None,
        'extracted_audio': None, 'muted_video': None
    }

    try:
        app.logger.info(f"JOB {job_id}: Starting background processing for {input_path}")
        update_status(job_id, 'starting', message='Job accepted, starting processing.')

        # Store original path relative to UPLOAD_FOLDER
        try:
            processing_files['original'] = os.path.relpath(input_path, app.config['UPLOAD_FOLDER'])
        except ValueError as e:
            app.logger.warning(f"JOB {job_id}: Could not make original path relative to UPLOAD_FOLDER: {e}. Storing absolute path.")
            processing_files['original'] = input_path

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        working_dir = os.path.join(output_dir_job, "working")
        os.makedirs(working_dir, exist_ok=True)
        app.logger.info(f"JOB {job_id}: Working directory is {working_dir}")

        # --- Video/audio handling ---
        audio_input_for_pipeline = input_path
        if is_video_file(input_path):
            update_status(job_id, 'extracting_audio', message='Extracting audio from video...')
            extracted_audio_path = os.path.join(working_dir, f"{base_name}_audio.wav")
            muted_video_path = os.path.join(working_dir, f"{base_name}_muted.mp4")
            app.logger.info(f"JOB {job_id}: Detected video file. Extracting audio to {extracted_audio_path} and muted video to {muted_video_path}")
            separate_audio_from_video(input_path, extracted_audio_path, muted_video_path)
            processing_files['extracted_audio'] = os.path.relpath(extracted_audio_path, app.config['OUTPUT_FOLDER'])
            processing_files['muted_video'] = os.path.relpath(muted_video_path, app.config['OUTPUT_FOLDER'])
            audio_input_for_pipeline = extracted_audio_path
            update_status(job_id, 'audio_extracted', current_file=processing_files['extracted_audio'], message='Audio extracted from video.')
        else:
            processing_files['extracted_audio'] = None
            processing_files['muted_video'] = None

        # 1. Vocal separation
        update_status(job_id, 'separating_vocals', message='Separating vocals...')
        app.logger.info(f"JOB {job_id}: Starting vocal separation for {audio_input_for_pipeline}")
        vocals_path, _ = separate_vocals(audio_input_for_pipeline, working_dir, model='mdx_extra')
        processing_files['separated_vocals'] = os.path.relpath(vocals_path, app.config['OUTPUT_FOLDER'])
        app.logger.info(f"JOB {job_id}: Vocal separation complete. Vocals at {vocals_path}")
        update_status(job_id, 'vocals_separated', current_file=processing_files['separated_vocals'], message='Vocal separation complete.')

        # 2. Convert separated vocals to 48kHz WAV
        update_status(job_id, 'converting_audio', message='Converting vocals to 48kHz WAV...')
        converted_wav_path = os.path.join(working_dir, f"{base_name}_vocals_48k.wav")
        app.logger.info(f"JOB {job_id}: Starting conversion to 48kHz for {vocals_path} to {converted_wav_path}")
        convert_to_48k_wav(vocals_path, converted_wav_path)
        processing_files['converted_48k'] = os.path.relpath(converted_wav_path, app.config['OUTPUT_FOLDER'])
        app.logger.info(f"JOB {job_id}: Audio conversion complete. Output: {converted_wav_path}")
        update_status(job_id, 'audio_converted', current_file=processing_files['converted_48k'], message='Audio conversion to 48kHz WAV complete.')

        # 3. Denoise the 48kHz vocals
        update_status(job_id, 'denoising', message='Denoising audio...')
        denoised_wav_path = os.path.join(working_dir, f"{base_name}_vocals_denoised.wav")
        app.logger.info(f"JOB {job_id}: Starting denoising for {converted_wav_path} to {denoised_wav_path}")
        denoise_audio(converted_wav_path, denoised_wav_path)
        processing_files['denoised'] = os.path.relpath(denoised_wav_path, app.config['OUTPUT_FOLDER'])
        app.logger.info(f"JOB {job_id}: Denoising complete. Output: {denoised_wav_path}")
        update_status(job_id, 'denoising_complete', current_file=processing_files['denoised'], message='Audio denoising complete.')

        # 4. Transcription (WhisperX, JSON output)
        update_status(job_id, 'transcribing', message=f'Transcribing original audio ({os.path.basename(input_path)})...')
        app.logger.info(f"JOB {job_id}: Starting transcription with WhisperX for {audio_input_for_pipeline} (Language: {language})")
        transcribed_json = transcribe_with_whisperx(
            audio_input_for_pipeline,
            language=language,
            enable_diarization=True,
            enable_emotion_detection=enable_emotion_detection
        )
        app.logger.info(f"JOB {job_id}: Transcription successful. Segments: {len(transcribed_json.get('segments', [])) if transcribed_json else 'N/A'}")

        transcript_file_path = os.path.join(output_dir_job, f"{base_name}_transcription.json")
        write_transcription_to_json(transcribed_json, transcript_file_path)
        processing_files['transcription'] = os.path.relpath(transcript_file_path, app.config['OUTPUT_FOLDER'])
        app.logger.info(f"JOB {job_id}: Transcription saved to {transcript_file_path}")
        num_speakers = 0
        if transcribed_json and "segments" in transcribed_json:
            speakers = set(seg.get("speaker") for seg in transcribed_json["segments"] if seg.get("speaker"))
            num_speakers = len(speakers)
            app.logger.info(f"JOB {job_id}: Detected speakers: {speakers} (Total: {num_speakers})")
            update_status(job_id, 'transcription_complete', current_file=processing_files['transcription'],
                          message=f'Transcription complete. Detected {num_speakers} speaker(s): {", ".join(sorted(speakers))}')
        else:
            update_status(job_id, 'transcription_complete', current_file=processing_files['transcription'],
                          message='Transcription complete. No speakers detected.')
            
        #5. Translation  
        update_status(job_id, 'translating', message='Translating transcription...')
        app.logger.info(f"JOB {job_id}: Starting translation for {transcript_file_path} from {language} to {target_lang}")
        translated_json_path = batch_translate_transcript_json(
            transcript_file_path,
            output_dir=output_dir_job,
            model_name="deepseek/deepseek-chat-v3-0324:free", 
            source_lang=language,
            target_lang=target_lang,
            translated_field="translated_text",
            batch_size=20,
            separator="[SEGMENT_BREAK]",
        )
        processing_files['translated_transcription'] = os.path.relpath(translated_json_path, app.config['OUTPUT_FOLDER'])
        app.logger.info(f"JOB {job_id}: Translation complete. Translated file at {translated_json_path}")

#        # 6. Speech Synthesis
#        transcribed_text = ""
#        if transcribed_json:
#            # Try to get the full text from the JSON (WhisperX format)
#           if "segments" in transcribed_json:
#                transcribed_text = " ".join(seg.get("text", "") for seg in transcribed_json["segments"])

#        update_status(job_id, 'synthesizing_speech', message='Synthesizing speech...')
#        app.logger.info(f"JOB {job_id}: Starting speech synthesis with Fish Speech. Reference: {audio_input_for_pipeline}")
#        final_synthesized_audio_path = synthesize_speech_with_fish_speech(
#            text=transcribed_text,
#            reference_audio_path=audio_input_for_pipeline,
#            output_dir=output_dir_job,
#            job_id=job_id
#        )
#        app.logger.info(f"JOB {job_id}: Speech synthesis successful. Output: {final_synthesized_audio_path}")
#        processing_files['final_output'] = os.path.relpath(final_synthesized_audio_path, app.config['OUTPUT_FOLDER'])

        # --- SPEAKER SEPARATION & SYNTHESIS ---
        try:
            app.logger.info(f"JOB {job_id}: Starting speaker separation pipeline (inlined)...")
            # Paths for speaker separation
            root_dir = output_dir_job
            # Use the correct transcript file path (should match where you saved it above)
            transcript_json = os.path.join(root_dir, f"{base_name}_transcription_batched.json")
            original_audio = processing_files['extracted_audio'] and os.path.join(app.config['OUTPUT_FOLDER'], processing_files['extracted_audio']) or input_path
            audio_output_dir = os.path.join(root_dir, "separated_audio")
            transcript_output_dir = os.path.join(root_dir, "separated_transcripts")
            synthesized_audio_output_dir = os.path.join(root_dir, "synthesized_audio")

            # Use the main transcript JSON
            if not os.path.exists(transcript_json):
                app.logger.error(f"❌ Transcript JSON not found for speaker separation at {transcript_json}.")
                update_status(job_id, 'speaker_sep_failed', message=f'Transcript JSON not found at {transcript_json}.')
                return

            if not os.path.exists(original_audio):
                app.logger.error(f"❌ Original audio not found for speaker separation at {original_audio}.")
                update_status(job_id, 'speaker_sep_failed', message=f'Original audio not found at {original_audio}.')
                return

            with open(transcript_json, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            update_status(job_id, 'speaker_sep_transcripts', message='Creating individual speaker transcripts...')
            app.logger.info("📝 Creating individual speaker transcripts...")
            speaker_transcript_files = create_speaker_transcript_files(transcript_json, transcript_output_dir)

            update_status(job_id, 'speaker_sep_audio', message='Separating audio by speaker...')
            app.logger.info("🔊 Separating audio by speaker...")
            speaker_audio_files = separate_audio_by_speaker(transcript_json, original_audio, audio_output_dir)

            update_status(job_id, 'speaker_sep_synth', message='Synthesizing speech for each speaker...')
            app.logger.info("🎤 Synthesizing speech for each speaker...")
            os.makedirs(synthesized_audio_output_dir, exist_ok=True)
            synthesized_audio_files = {}

            for idx, speaker_id in enumerate(speaker_audio_files.keys()):
                try:
                    update_status(job_id, 'speaker_sep_synth', message=f'Synthesizing for {speaker_id} ({idx+1}/{len(speaker_audio_files)})...')
                    app.logger.info(f"🔄 Processing {speaker_id} ({idx+1}/{len(speaker_audio_files)})...")
                    denoise_audio(speaker_audio_files[speaker_id], speaker_audio_files[speaker_id])
                    app.logger.info(f"   Denoised audio for {speaker_id}")
                    reference_segment_path = select_representative_segment(speaker_audio_files[speaker_id])
                    app.logger.info(f"   Selected representative segment for {speaker_id}: {reference_segment_path}")
                    speaker_output_dir = os.path.join(synthesized_audio_output_dir, speaker_id)
                    os.makedirs(speaker_output_dir, exist_ok=True)
                    app.logger.info(f"   Synthesizing chunked speech for {speaker_id}...")
                    synthesized_audio_path = synthesize_chunked_speech(
                        transcript_data=transcript_data,
                        reference_audio_path=reference_segment_path,
                        output_dir=speaker_output_dir,
                        target_speaker=speaker_id,
                        cleanup_existing=True
                    )
                    if synthesized_audio_path:
                        synthesized_audio_files[speaker_id] = synthesized_audio_path
                        app.logger.info(f"✅ Synthesized audio for {speaker_id} saved to: {synthesized_audio_path}")
                    else:
                        app.logger.error(f"❌ Failed to synthesize audio for {speaker_id}")
                except Exception as e:
                    app.logger.error(f"❌ Error processing speaker {speaker_id}: {e}")

            update_status(job_id, 'speaker_sep_complete', message='Speaker separation and synthesis complete.')
            app.logger.info("🎉 Speaker separation and synthesis complete!")
        except Exception as e:
            app.logger.error(f"JOB {job_id}: Speaker separation pipeline failed: {e}")
            update_status(job_id, 'speaker_sep_failed', message=f"Speaker separation pipeline failed: {e}")
            return

        # --- RECONSTRUCT FINAL VIDEO WITH SYNTHESIZED AUDIO ---
        try:
            final_video_path = None
            if os.path.exists(original_audio) and os.path.exists(synthesized_audio_output_dir):
                app.logger.info(f"JOB {job_id}: Reconstructing final video with synthesized audio...")
                final_video_path = reconstruct_layered_audio(transcript_data, synthesized_audio_output_dir, muted_video_path, output_dir_job, root_dir)
                processing_files['final_output'] = os.path.relpath(final_video_path, app.config['OUTPUT_FOLDER'])
                app.logger.info(f"JOB {job_id}: Final video with audio saved to: {final_video_path}")
            else:
                app.logger.warning(f"JOB {job_id}: Original audio or synthesized audio directory not found. Skipping video reconstruction.")
        except Exception as e:
            app.logger.error(f"JOB {job_id}: Final video reconstruction failed: {e}")
            update_status(job_id, 'failed', message=f"Final video reconstruction failed: {e}")
            return

    except Exception as e:
        error_detail = f"JOB {job_id}: Pipeline error: {str(e)}"
        app.logger.error(error_detail, exc_info=True)
        if transcribed_text:
            error_detail += f" | Transcription (partial): {transcribed_text[:100]}..."

        try:
            processing_files_record_path = os.path.join(output_dir_job, "processing_files.json")
            with open(processing_files_record_path, 'w') as f:
                json.dump(processing_files, f, indent=4)
            app.logger.info(f"JOB {job_id}: Saved partial processing_files.json after error.")
        except Exception as dump_err:
            app.logger.error(f"JOB {job_id}: Could not save processing_files.json after error: {dump_err}")

        update_status(job_id, 'failed', message=error_detail)
        app.logger.error(f"JOB {job_id}: Background processing failed.")
   

@app.route('/progress/<job_id>')
def progress(job_id):
    status = get_status(job_id)
    return jsonify(status)

@app.route('/process_files/<job_id>')
def process_files(job_id):
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    files_record_path = os.path.join(output_dir, "processing_files.json")
    
    if not os.path.exists(files_record_path):
        return jsonify({'error': 'Processing files record not found. Job may still be in progress or failed early.'}), 404
        
    try:
        with open(files_record_path, 'r') as f:
            files_data = json.load(f)
        
        # Ensure paths are suitable for web (relative to output_folder, forward slashes)
        # The paths in processing_files.json are already relative to app.config['OUTPUT_FOLDER']
        # We just need to ensure they are formatted nicely for URLs/display
        web_friendly_files = {}
        for key, path_val in files_data.items():
            if path_val: # If path is not None
                 # Ensure job_id is part of the path for download URL construction
                if not path_val.startswith(job_id):
                     web_friendly_files[key] = f"{job_id}/{path_val.replace(os.sep, '/')}"
                else:
                     web_friendly_files[key] = path_val.replace(os.sep, '/')
            else:
                web_friendly_files[key] = None
        
        return jsonify(web_friendly_files)
    except Exception as e:
        app.logger.error(f"Error reading processing_files.json for job {job_id}: {str(e)}")
        return jsonify({'error': f'Could not retrieve file list: {str(e)}'}), 500
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                job_id = str(uuid.uuid4())[:8] # Short, unique-enough ID
                # filename_secure = secure_filename(file.filename) # Not using timestamp prefix for job folder
                
                # Job-specific UPLOAD subfolder (e.g., /app/uploads/<job_id>/original.mp3)
                job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
                os.makedirs(job_upload_dir, exist_ok=True)
                # Save uploaded file with its original (secured) name inside job_upload_dir
                input_file_path = os.path.join(job_upload_dir, secure_filename(file.filename))
                file.save(input_file_path)
                
                # Job-specific OUTPUT folder (e.g., /app/output/<job_id>/...)
                job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
                os.makedirs(job_output_dir, exist_ok=True) # process_file_in_background will use this
                
                # Get form parameters for processing
                language = request.form.get('language', 'auto')
                whisper_model = request.form.get('whisper_model', 'base')
                reference_choice = request.form.get('reference_audio', 'same') # Passed to background, but effectively ignored by current notebook alignment
                min_speakers = int(request.form.get('min_speakers', 1))
                max_speakers = int(request.form.get('max_speakers', 2))
                target_lang = request.form.get('target_lang')
                # --- Get emotion detection toggle ---
                emotion_detection = request.form.get('emotion_detection', 'off').lower() == 'on'
                app.logger.info(f"Job {job_id}: File '{file.filename}' uploaded. Language: {language}, Model: {whisper_model}, RefChoice: {reference_choice}")

                from threading import Thread
                thread = Thread(target=process_file_in_background, args=(
                    input_file_path,      # Full path to the uploaded original file
                    job_output_dir,       # Base output directory for this job
                    job_id,
                    language,
                    target_lang,
                    whisper_model,
                    reference_choice,
                    min_speakers,
                    max_speakers,
                    # Add as kwarg for backward compatibility
                ), kwargs={'enable_emotion_detection': emotion_detection})
                thread.start()
                
                return redirect(url_for('processing', job_id=job_id))
                
            except Exception as e:
                app.logger.error(f"Error during file upload or job start: {str(e)}", exc_info=True)
                flash(f'An error occurred: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed.')
            return redirect(request.url)
            
    return render_template('index.html')


@app.route('/processing/<job_id>')
def processing(job_id):
    return render_template('processing.html', job_id=job_id)

@app.route('/results/<job_id>')
def results(job_id):
    status = get_status(job_id)
    if status.get('status') != 'complete':
        if status.get('status') == 'failed':
            flash(f"Processing failed for job {job_id}: {status.get('message', 'Unknown error')}", 'error')
        return redirect(url_for('processing', job_id=job_id))
    
    output_dir_job = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    files_record_path = os.path.join(output_dir_job, "processing_files.json")
    
    try:
        with open(files_record_path, 'r') as f:
            raw_files_info = json.load(f) # Load raw info
    except FileNotFoundError:
        flash(f"Results file record for job {job_id} not found.", "error")
        return redirect(url_for('index'))
    except json.JSONDecodeError:
        flash(f"Error reading results for job {job_id}.", "error")
        return redirect(url_for('index'))

    # Pre-process all paths in files_info for the template to use forward slashes
    files_info_for_template = {}
    for key, path_str in raw_files_info.items():
        if isinstance(path_str, str):
            files_info_for_template[key] = path_str.replace(os.sep, '/')
        else:
            files_info_for_template[key] = path_str # Handles None or other types

    transcription_rel_path = files_info_for_template.get('transcription')
    final_audio_rel_path = files_info_for_template.get('final_output')

    if not transcription_rel_path or not final_audio_rel_path:
        flash(f"Essential result files are missing for job {job_id}.", "error")
        return redirect(url_for('index'))

    # Construct absolute path to read transcription (original rel path might have os.sep)
    # Use raw_files_info for file system operations before normalization for URLs
    transcription_abs_path_fs = os.path.join(app.config['OUTPUT_FOLDER'], raw_files_info.get('transcription'))
    
    try:
        with open(transcription_abs_path_fs, 'r', encoding='utf-8') as t_file:
            transcription_text = t_file.read()
    except FileNotFoundError:
        flash(f"Transcription file not found for job {job_id}.", "error")
        return redirect(url_for('processing', job_id=job_id))
    
    return render_template(
        'result.html',
        job_id=job_id,
        audio_file_url_path=final_audio_rel_path, # Already processed to use '/'
        transcription_file_url_path=transcription_rel_path, # Already processed to use '/'
        transcription_text=transcription_text,
        model_used=status.get('whisper_model', 'N/A'),
        files_info=files_info_for_template # Pass the processed version
    )



@app.route('/download/<path:filepath_in_output>')
def download_file(filepath_in_output):
    # filepath_in_output is expected to be like "job_id/some_folder/file.wav"
    # send_from_directory needs directory and then the filename relative to that directory
    # Here, app.config['OUTPUT_FOLDER'] is the base directory.
    app.logger.info(f"Download request for: {filepath_in_output} from {app.config['OUTPUT_FOLDER']}")
    return send_from_directory(
        app.config['OUTPUT_FOLDER'],
        filepath_in_output,
        as_attachment=True
    )

# --- New functions for advanced speaker separation and synthesis ---

def separate_audio_by_speaker(transcript_json_path: str, original_audio_path: str, output_dir: str) -> Dict[str, str]:
    """
    Separate audio file into individual speaker segments based on WhisperX transcript.

     Args:
        transcript_json_path: Path to the JSON transcript file
        original_audio_path: Path to the original audio file
        output_dir: Directory to save separated audio files
    
    Returns:
        Dictionary mapping speaker IDs to their audio file paths
    """
    #Load transcript
    with open(transcript_json_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)

    # Extract segments form the nested structure    
    segments = transcript_data.get('result', {}).get('segments', [])

    #Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    #Group segments by speaker
    speaker_segments = defaultdict(list)

    for segment in segments:
        speaker = segment.get('speaker')
        if not speaker:
            continue
        speaker_segments[speaker].append({
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'text': segment.get('text', '').strip()
        })
    
    app.logger.info(f"🎤 Processing speakers: {list(speaker_segments.keys())}")
    
    # Print speaker statistics
    for speaker, segs in speaker_segments.items():
        total_duration = sum(seg['end'] - seg['start'] for seg in segs)
        app.logger.info(f"   {speaker}: {len(segs)} segments, {total_duration:.1f}s total")
    
    # Create audio files for each speaker
    speaker_files = {}
    for speaker_id, segments in speaker_segments.items():
        app.logger.info(f"\n🔄 Processing {speaker_id} ({len(segments)} segments)...")
        filter_parts = []
        for i, segment in enumerate(segments):
            start_time = segment['start']
            duration = segment['end'] - segment['start']
            if duration < 0.1:
                continue
            filter_parts.append(f"[0:a]atrim=start={start_time}:duration={duration},asetpts=PTS-STARTPTS[a{i}]")
        if not filter_parts:
            app.logger.warning(f"   ⚠️ No valid segments found for {speaker_id}")
            continue
        if len(filter_parts) == 1:
            filter_complex = filter_parts[0].replace(f"[a0]", "[out]")
        else:
            filter_complex = ";".join(filter_parts)
            inputs_list = "".join([f"[a{i}]" for i in range(len(filter_parts))])
            filter_complex += f";{inputs_list}concat=n={len(filter_parts)}:v=0:a=1[out]"
        speaker_file = os.path.join(output_dir, f"{speaker_id}.wav")
        speaker_files[speaker_id] = speaker_file
        cmd = [
            'ffmpeg', '-y',
            '-i', original_audio_path,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:a', 'pcm_s16le',
            '-ar', '48000',
            speaker_file
        ]
        try:
            app.logger.info(f"   Extracting audio for {speaker_id}...")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            app.logger.info(f"   ✅ Saved: {speaker_file}")
        except subprocess.CalledProcessError as e:
            app.logger.error(f"   ❌ Error processing {speaker_id}: {e}")
            app.logger.error(f"   Command: {' '.join(cmd)}")
            continue
    return speaker_files

def create_speaker_transcript_files(transcript_json_path: str, output_dir: str) -> Dict[str, str]:
    """
    Create individual transcript files for each speaker.
    
    Args:
        transcript_json_path: Path to the JSON transcript file
        output_dir: Directory to save speaker transcript files
    
    Returns:
        Dictionary mapping speaker IDs to their transcript file paths
    """
    
    # Load transcript
    with open(transcript_json_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    segments = transcript_data.get('result', {}).get('segments', [])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group segments by speaker
    speaker_segments = defaultdict(list)
    
    for segment in segments:
        speaker = segment.get('speaker')
        if not speaker:
            continue
            
        speaker_segments[speaker].append({
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'text': segment.get('text', '').strip(),
            'words': segment.get('words', [])
        })
    
    speaker_transcript_files = {}
    
    for speaker_id, segments in speaker_segments.items():
        # Create text transcript
        txt_file = os.path.join(output_dir, f"{speaker_id}_transcript_batched.txt")
        srt_file = os.path.join(output_dir, f"{speaker_id}_transcript_batched.srt")
        json_file = os.path.join(output_dir, f"{speaker_id}_transcript_batched.json")
        
        # Write plain text transcript
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Speaker: {speaker_id}\n")
            f.write("=" * 50 + "\n\n")
            
            for segment in segments:
                timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
                f.write(f"{timestamp} {segment['text']}\n")
        
        # Write SRT format
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = format_timestamp_srt(segment['start'])
                end_time = format_timestamp_srt(segment['end'])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
        
        # Write JSON format with detailed info
        with open(json_file, 'w', encoding='utf-8') as f:
            speaker_data = {
                'speaker_id': speaker_id,
                'total_segments': len(segments),
                'total_duration': sum(seg['end'] - seg['start'] for seg in segments),
                'segments': segments
            }
            json.dump(speaker_data, f, indent=2, ensure_ascii=False)
        
        speaker_transcript_files[speaker_id] = {
            'txt': txt_file,
            'srt': srt_file,
            'json': json_file
        }
        
        app.logger.info(f"📝 Created transcripts for {speaker_id}:")
        app.logger.info(f"   Text: {txt_file}")
        app.logger.info(f"   SRT: {srt_file}")
        app.logger.info(f"   JSON: {json_file}")
    
    return speaker_transcript_files

def select_representative_segment(audio_path: str, duration_sec: int = 10) -> str:
    """
    Select a continuous 10-second segment from the audio file with improved robustness.
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        if len(audio) < duration_sec * 1000:
            if len(audio) > 1000:
                segment = audio
                app.logger.warning(f"⚠️ Audio is shorter than {duration_sec} seconds. Using available audio.")
            else:
                raise ValueError(f"Audio is too short (only {len(audio)/1000:.1f}s)")
        else:
            most_active_segment = None
            highest_energy = 0
            for start in range(0, len(audio) - duration_sec * 1000, 1000):
                segment = audio[start:start + duration_sec * 1000]
                samples = np.array(segment.get_array_of_samples())
                if len(samples) == 0:
                    continue
                rms_energy = np.sqrt(np.mean(samples ** 2))
                if np.isnan(rms_energy) or rms_energy < 100:
                    continue
                if rms_energy > highest_energy:
                    highest_energy = rms_energy
                    most_active_segment = segment
            if most_active_segment is None:
                app.logger.warning("⚠️ No high-energy segment found. Using middle segment as fallback.")
                mid_point = len(audio) // 2
                segment = audio[mid_point:mid_point + duration_sec * 1000]
            else:
                segment = most_active_segment
        samples = np.array(segment.get_array_of_samples())
        if len(samples) == 0 or np.max(np.abs(samples)) < 100:
            raise ValueError("Selected segment has no audible content")
        output_path = os.path.splitext(audio_path)[0] + "_segment.wav"
        segment.export(output_path, format="wav")
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ValueError("Failed to save valid segment file")
        return output_path
    except Exception as e:
        app.logger.error(f"⚠️ Critical error in segment selection: {str(e)}")
        app.logger.warning("⚠️ Creating silent segment as last resort")
        silent_segment = AudioSegment.silent(duration=duration_sec * 1000)
        output_path = os.path.splitext(audio_path)[0] + "_segment.wav"
        silent_segment.export(output_path, format="wav")
        return output_path

def chunk_transcript_advanced(transcript_data: dict) -> list:
    """
    Split the transcript into chunks based on segments with accurate speaker detection.
    """
    # --- CHANGED: Always extract segments from correct location ---
    segments = None
    if "result" in transcript_data and isinstance(transcript_data["result"], dict):
        if "segments" in transcript_data["result"]:
            segments = transcript_data["result"]["segments"]
    elif "segments" in transcript_data:
        segments = transcript_data["segments"]
    if segments is None:
        return []
    def get_dominant_speaker(segment):
        # If "speaker" field exists, use it directly (for batched JSONs)
        if "speaker" in segment and segment["speaker"]:
            return segment["speaker"]
        words = segment.get("words", [])
        if not words:
            return "UNKNOWN"
        speaker_word_counts = {}
        speaker_duration = {}
        for word in words:
            speaker = word.get("speaker", "UNKNOWN")
            start = word.get("start", 0)
            end = word.get("end", 0)
            duration = end - start
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1
            speaker_duration[speaker] = speaker_duration.get(speaker, 0) + duration
        if speaker_duration:
            dominant_speaker = max(speaker_duration, key=speaker_duration.get)
            return dominant_speaker
        return "UNKNOWN"
    chunks = []
    for i, segment in enumerate(segments):
        segment_text = segment.get("text", "").strip()
        speaker = segment.get("speaker") or get_dominant_speaker(segment)
        # --- EMOTION: propagate emotion field if present ---
        emotion = segment.get("emotion", "none")
        if segment_text:
            chunks.append({
                "index": i,
                "id": f"seg_{i:04d}",
                "text": segment_text,
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "speaker": speaker,
                "emotion": emotion
            })
    return chunks

def synthesize_chunked_speech(transcript_data: dict, reference_audio_path: str, output_dir: str, target_speaker: str = None, cleanup_existing: bool = True) -> Optional[str]:
    app.logger.info(f"🔍 Starting synthesis process for speaker: {target_speaker or 'ALL'}")
    os.makedirs(output_dir, exist_ok=True)
    segments_dir = os.path.join(output_dir, "segments")
    if cleanup_existing and os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
        app.logger.info("🧹 Cleaned up existing segments directory")
    os.makedirs(segments_dir, exist_ok=True)
    all_chunks = chunk_transcript_advanced(transcript_data)
    if not all_chunks:
        app.logger.error("❌ No segments found in transcript")
        return None
    # --- CHANGED: Filter by actual speaker field in chunk ---
    if target_speaker:
        chunks = [chunk for chunk in all_chunks if chunk["speaker"] == target_speaker]
        app.logger.info(f"🎭 Filtered to {len(chunks)} segments for {target_speaker} (out of {len(all_chunks)} total)")
    else:
        chunks = all_chunks
        app.logger.info(f"📄 Processing all {len(chunks)} segments")
    if not chunks:
        app.logger.error(f"❌ No segments found for speaker: {target_speaker}")
        return None
    separated_audio_dir = os.path.dirname(reference_audio_path)
    speaker_audio_map = {}
    for speaker_file in os.listdir(separated_audio_dir):
        if speaker_file.endswith('.wav') and 'SPEAKER_' in speaker_file and '_segment' not in speaker_file:
            speaker_name = speaker_file.replace('.wav', '')
            speaker_audio_map[speaker_name] = os.path.join(separated_audio_dir, speaker_file)
    app.logger.info(f"🎭 Available speaker audio files: {list(speaker_audio_map.keys())}")
    synthesized_chunks = []
    failed_segments = []
    for i, chunk in enumerate(chunks):
        segment_index = chunk["index"]
        segment_id = f"{target_speaker}_{chunk['id']}" if target_speaker else chunk["id"]
        # --- EMOTION: prepend emotion to text if present ---
        emotion = chunk.get("emotion", "none")
        segment_text = chunk["text"]
        if emotion and emotion not in ["none", "unknown", ""]:
            segment_text = f"({emotion}) {segment_text}"
        speaker = chunk["speaker"]
        start_time = chunk["start"]
        end_time = chunk["end"]
        app.logger.info(f"\n🔄 Synthesizing segment {i + 1}/{len(chunks)}: {segment_id}")
        app.logger.info(f"   Original index: {segment_index}, Speaker: {speaker}, Time: {start_time:.2f}s - {end_time:.2f}s")
        app.logger.info(f"   Text: {segment_text[:100]}{'...' if len(segment_text) > 100 else ''}")
        speaker_reference_audio = speaker_audio_map.get(speaker, reference_audio_path)
        app.logger.info(f"🎤 Using reference audio: {os.path.basename(speaker_reference_audio)}")
        synthesized_path = synthesize_speech(
            text=segment_text,
            reference_audio_path=speaker_reference_audio,
            output_dir=segments_dir,
            segment_id=segment_id
        )
        if synthesized_path:
            synthesized_chunks.append({
                "path": synthesized_path,
                "original_index": segment_index,
                "segment_id": segment_id,
                "speaker": speaker,
                "start": start_time,
                "end": end_time
            })
            app.logger.info(f"✅ Segment {segment_id} saved")
        else:
            failed_segments.append(segment_index)
            app.logger.error(f"❌ Failed to synthesize segment {segment_id}")
    synthesized_chunks.sort(key=lambda x: x["original_index"])
    if not synthesized_chunks:
        app.logger.error("❌ No valid segments were synthesized.")
        return None
    final_audio = AudioSegment.empty()
    for chunk_data in synthesized_chunks:
        chunk_path = chunk_data["path"]
        chunk_audio = AudioSegment.from_file(chunk_path)
        final_audio += chunk_audio
    speaker_suffix = f"_{target_speaker}" if target_speaker else ""
    final_output_path = os.path.join(output_dir, f"final_output{speaker_suffix}.wav")
    final_audio.export(final_output_path, format="wav", parameters=["-ac", "1"])
    if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1024:
        return final_output_path
    else:
        return None

def reconstruct_layered_audio(transcript_data, synthesized_output_dir, video_path, output_dir=None, root_dir=None):
    # 1. Load video to get duration and fps
    print("Hallo test wir sind in reconstruct")
    print(video_path)
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration

    # 2. Prepare per-speaker timeline
    if 'result' in transcript_data and 'segments' in transcript_data['result']:
        segments = transcript_data['result']['segments']
    elif 'segments' in transcript_data:
        segments = transcript_data['segments']
    else:
        raise KeyError("No segments found in transcript data! Top-level keys: " + str(list(transcript_data.keys())))
    speakers = [d for d in os.listdir(synthesized_output_dir) if d.startswith("SPEAKER_")]
    speaker_tracks = []

    for speaker in speakers:
        speaker_dir = os.path.join(synthesized_output_dir, speaker, "segments")
        # Blank track (silence)
        track = AudioSegment.silent(duration=int(video_duration * 1000)) # pydub uses ms

        for i, seg in enumerate(segments):
            if seg.get('speaker', seg.get('speaker_label')) == speaker:
                wav_name = f"{speaker}_seg_{i:04d}.wav"

                seg_path = os.path.join(speaker_dir, wav_name)
                if os.path.exists(seg_path):
                    seg_audio = AudioSegment.from_wav(seg_path)
                    start_ms = int(seg['start'] * 1000)
                    # Overlay the segment at the correct time
                    track = track.overlay(seg_audio, position=start_ms)
        # Export this speaker track as temp wav
        temp_path = f"{speaker}_layer.wav"
        track.export(temp_path, format="wav")
        speaker_tracks.append(temp_path)

    print("speakers doneso")

    # 3. Mix all speaker tracks
    mixed = AudioSegment.silent(duration=int(video_duration * 1000))
    for track_path in speaker_tracks:
        tr = AudioSegment.from_wav(track_path)
        mixed = mixed.overlay(tr)
    print(f"Speaker tracks: {speaker_tracks}")
    for path in speaker_tracks:
        print(f"Track exists? {os.path.exists(path)}, Size: {os.path.getsize(path) if os.path.exists(path) else 0}")
    
    
    # Export mixed
    final_audio_path = os.path.join(root_dir, "final_mixed_audio.wav")
    mixed.export(final_audio_path, format="wav")
    print(f"Will export mixed audio to: {final_audio_path}")
    print(f"Mixed audio length: {len(mixed)/1000.0:.2f} seconds")
    # 4. Combine with original video
    output_video = "final_video_with_audio.mp4" if output_dir is None else os.path.join(output_dir, "final_video_with_audio.mp4")
    print(output_video)
    # 1. Get directory name
    dir_path = os.path.dirname(output_video)

    # 2. Create directory if it does not exist
    os.makedirs(dir_path, exist_ok=True) 
    final_audio = AudioFileClip(final_audio_path)
    new_video = video_clip.with_audio(final_audio)
    new_video.write_videofile(output_video, codec="libx264", audio_codec="aac")

    print(f"✅ Done! Output: {output_video}")
    return output_video  




if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5000, debug=True) # Debug=True is not for production
