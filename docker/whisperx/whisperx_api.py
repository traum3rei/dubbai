import os
# Force enable HuggingFace Hub online mode at the very beginning
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from flask import Flask, request, jsonify, send_file
import torch
import uuid
import json
import traceback
import sys
import tempfile
import shutil
from werkzeug.utils import secure_filename
import gc
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Supported audio formats
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'mp4', 'avi', 'mov', 'webm'}

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables to cache models
whisper_model = None
diarization_model = None
alignment_model = None
alignment_metadata = None
model_device = None
model_compute_type = None
whisperx_available = False

HF_TOKEN = os.environ.get('HF_TOKEN', None)
os.environ['HF_HUB_OFFLINE'] = '0'  # Enable online access
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Enable online access

# Try to import WhisperX
try:
    import whisperx
    whisperx_available = True
    print("✅ WhisperX imported successfully")
except ImportError as e:
    print(f"❌ WhisperX import failed: {e}")
    print("Trying to install WhisperX...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "whisperx"])
        import whisperx
        whisperx_available = True
        print("✅ WhisperX installed and imported successfully")
    except Exception as e2:
        print(f"❌ Failed to install WhisperX: {e2}")
        whisperx_available = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model(model_name="large-v3", device="cuda", compute_type="float16"):
    """Load the Whisper model - FORCED to use large-v3 only"""
    global whisper_model, model_device, model_compute_type
    
    if not whisperx_available:
        raise Exception("WhisperX is not available")
    
    # FORCE model to be large-v3 regardless of input
    forced_model = "large-v3"
    if model_name != forced_model:
        print(f"⚠️  Model '{model_name}' requested but forcing to use '{forced_model}'")
        logger.warning(f"Model '{model_name}' requested but forcing to use '{forced_model}'")
    
    model_name = forced_model  # Override any other model choice
    
    # Check if we need to reload the model
    if (whisper_model is None or 
        model_device != device or 
        model_compute_type != compute_type):
        
        print(f"Loading Whisper model: {model_name} on {device}")
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            compute_type = "int8"
        
        # Clean up existing model
        if whisper_model is not None:
            del whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Only try large-v3 model
        try:
            print(f"Loading model: {model_name}")
            logger.info(f"Loading model: {model_name}")
            
            whisper_model = whisperx.load_model(
                model_name, 
                device=device, 
                compute_type=compute_type,
                asr_options={
                    "beam_size": 1,
                    "best_of": 1,
                    "compression_ratio_threshold": 2.4,
                    "log_prob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_previous_text": True,
                },
                vad_options={
                    "vad_onset": 0.500,
                    "vad_offset": 0.363
                }
            )
            
            model_device = device
            model_compute_type = compute_type
            print(f"✅ Model '{model_name}' loaded successfully on {device}")
            logger.info(f"Model '{model_name}' loaded successfully on {device}")
            
        except Exception as e:
            # Try CPU if CUDA failed
            if device == "cuda":
                try:
                    print(f"Retrying '{model_name}' on CPU...")
                    whisper_model = whisperx.load_model(
                        model_name, 
                        device="cpu", 
                        compute_type="int8"
                    )
                    model_device = "cpu"
                    model_compute_type = "int8"
                    print(f"✅ Model '{model_name}' loaded successfully on CPU")
                    logger.info(f"Model '{model_name}' loaded successfully on CPU")
                except Exception as e2:
                    error_msg = f"Failed to load {model_name} on both CUDA and CPU. CUDA error: {e}, CPU error: {e2}"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                error_msg = f"Failed to load {model_name} on {device}: {e}"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                raise Exception(error_msg)
        
        if whisper_model is None:
            error_msg = f"Failed to load {model_name} model"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    return whisper_model

def load_alignment_model(language_code, device):
    """Load the alignment model for the given language"""
    global alignment_model, alignment_metadata
    
    try:
        print(f"Loading alignment model for language: {language_code}")
        # The load_align_model returns a tuple: (model, metadata)
        alignment_model, alignment_metadata = whisperx.load_align_model(
            language_code=language_code, 
            device=device
        )
        print(f"✅ Alignment model loaded successfully")
        return alignment_model, alignment_metadata
    except Exception as e:
        print(f"❌ Failed to load alignment model: {e}")
        logger.error(f"Failed to load alignment model: {e}")
        return None, None

def load_diarization_model(device):
    """Load the speaker diarization model"""
    global diarization_model
    
    if diarization_model is None:
        try:
            print("Loading speaker diarization model...")
            if not HF_TOKEN:
                raise Exception("HF_TOKEN not set. Please set HF_TOKEN environment variable.")
                
            print(f"Using HF_TOKEN: {HF_TOKEN[:10]}...")  # Show first 10 chars for verification
            
            # Use the correct import path for DiarizationPipeline
            diarization_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=HF_TOKEN, 
                device=device
            )
            print("✅ Diarization model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load diarization model: {e}")
            logger.error(f"Failed to load diarization model: {e}")
            raise e
    
    return diarization_model

def transcribe_audio_with_diarization(audio_path, language="en", model_name="large-v3", device="cuda", compute_type="float16", enable_diarization=True):
    """Complete transcription with alignment and diarization - FORCED to use large-v3"""
    try:
        # Force model to large-v3
        if model_name != "large-v3":
            print(f"⚠️  Model '{model_name}' requested but forcing to use 'large-v3'")
            logger.warning(f"Model '{model_name}' requested but forcing to use 'large-v3'")
        model_name = "large-v3"  # Force override
        
        print(f"Starting transcription with diarization of: {audio_path}")
        logger.info(f"Starting transcription with diarization using model: {model_name}")
        
        if not whisperx_available:
            raise Exception("WhisperX is not available")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        print("Loading audio...")
        audio = whisperx.load_audio(audio_path)
        audio_duration = len(audio) / 16000
        print(f"Audio loaded: {len(audio)} samples, duration: {audio_duration:.2f}s")
        logger.info(f"Audio loaded: duration: {audio_duration:.2f}s")
        
        # Check audio levels
        import numpy as np
        audio_max = np.max(np.abs(audio))
        audio_rms = np.sqrt(np.mean(audio**2))
        print(f"Audio stats - Max: {audio_max:.6f}, RMS: {audio_rms:.6f}")
        
        # Step 1: Load and run Whisper transcription with FORCED large-v3
        print(f"Step 1: Loading Whisper model (forced: {model_name})...")
        model = load_whisper_model(model_name, device, compute_type)
        
        print("Step 1: Starting Whisper transcription...")
        result = model.transcribe(
            audio, 
            batch_size=4,
            language=language if language != "auto" else None
        )
        
        print(f"Step 1 completed: {len(result.get('segments', []))} segments")
        
        # Step 2: Align whisper output (optional but recommended)
        try:
            print("Step 2: Loading alignment model...")
            align_model, metadata = load_alignment_model(language_code=language, device=device)
            
            if align_model is not None and metadata is not None:
                print("Step 2: Aligning transcription...")
                # Use the correct function signature for align
                result = whisperx.align(
                    result["segments"], 
                    align_model, 
                    metadata,  # Use the metadata returned from load_align_model
                    audio, 
                    device, 
                    return_char_alignments=False
                )
                print("Step 2 completed: Transcription aligned")
            else:
                print("Step 2 skipped: Alignment model not available")
                
        except Exception as e:
            print(f"Step 2 warning: Alignment failed: {e}")
            logger.warning(f"Alignment failed: {e}")
        
        # Step 3: Speaker Diarization (if enabled and HF token available)
        if enable_diarization:
            try:
                print("Step 3: Loading diarization model...")
                diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token= HF_TOKEN ,device= device)
                
                if diarize_model and HF_TOKEN:
                    print("Step 3: Running speaker diarization...")
                    diarize_segments = diarize_model(audio)
                    
                    print("Step 3: Assigning speakers to segments...")
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print("Step 3 completed: Speaker diarization assigned")
                    
                    # Count unique speakers
                    speakers = set()
                    for segment in result.get("segments", []):
                        if "speaker" in segment:
                            speakers.add(segment["speaker"])
                    
                    print(f"✅ Detected {len(speakers)} unique speakers: {sorted(speakers)}")
                    
                else:
                    print("Step 3 skipped: Diarization model not available or no HF token")
                    
            except Exception as e:
                print(f"Step 3 warning: Diarization failed: {e}")
                logger.warning(f"Diarization failed: {e}")
        else:
            print("Step 3 skipped: Diarization disabled")
        
        # Process results
        segments = result.get('segments', [])
        
        # Debug first few segments
        print("\nFirst few segments:")
        for i, segment in enumerate(segments[:3]):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            print(f"  {i}: [{start:.1f}-{end:.1f}s] {speaker}: {text[:100]}...")
        
        # Create summary
        full_text = ""
        speaker_stats = {}
        for segment in segments:
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'Unknown')
            
            if text:
                full_text += text + " "
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'segments': 0, 'total_time': 0}
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['total_time'] += segment.get('end', 0) - segment.get('start', 0)
        
        full_text = full_text.strip()
        
        print(f"\n✅ Transcription completed:")
        print(f"  - {len(segments)} total segments")
        print(f"  - {len(speaker_stats)} speakers detected")
        print(f"  - {len(full_text)} characters of text")
        print(f"  - Model used: {model_name} (forced)")
        
        # Add metadata to result
        if 'language' not in result:
            result['language'] = language
        
        result['speaker_stats'] = speaker_stats
        result['model_used'] = model_name  # Add this to track which model was actually used
        result['debug_info'] = {
            'audio_duration': audio_duration,
            'audio_max': float(audio_max),
            'audio_rms': float(audio_rms),
            'segments_count': len(segments),
            'speakers_count': len(speaker_stats),
            'total_text_length': len(full_text),
            'text_preview': full_text[:200] if full_text else "NO TEXT FOUND",
            'diarization_enabled': enable_diarization,
            'hf_token_available': bool(HF_TOKEN),
            'model_forced': model_name
        }
        
        # Clean up GPU memory
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        error_msg = f"Error during transcription with diarization: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        traceback.print_exc()
        
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else "None"
        
        return jsonify({
            'status': 'healthy',
            'whisperx_available': whisperx_available,
            'cuda_available': cuda_available,
            'cuda_device': cuda_device,
            'model_loaded': whisper_model is not None,
            'model_forced': 'large-v3',  # Indicate forced model
            'diarization_model_loaded': diarization_model is not None,
            'alignment_model_loaded': alignment_model is not None,
            'model_device': model_device,
            'model_compute_type': model_compute_type,
            'hf_token_available': bool(HF_TOKEN),
            'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE', '0'),
            'upload_dir': UPLOAD_DIR,
            'output_dir': OUTPUT_DIR
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to pre-load model - FORCED to large-v3"""
    try:
        data = request.get_json() or {}
        requested_model = data.get('model', 'large-v3')
        device = data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        compute_type = data.get('compute_type', 'float16' if device == 'cuda' else 'int8')
        
        # Force model to large-v3
        forced_model = 'large-v3'
        if requested_model != forced_model:
            print(f"⚠️  Model '{requested_model}' requested but forcing to use '{forced_model}'")
        
        print(f"Loading model via API: {forced_model} (forced)")
        model = load_whisper_model(forced_model, device, compute_type)
        
        return jsonify({
            'status': 'success',
            'message': f'Model {forced_model} loaded successfully (forced)',
            'model_requested': requested_model,
            'model_loaded': forced_model,
            'device': model_device,
            'compute_type': model_compute_type
        })
        
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'error': error_msg
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe uploaded audio file with speaker diarization - FORCED to use large-v3"""
    job_id = str(uuid.uuid4())
    file_path = None
    
    try:
        print(f"=== Job {job_id}: Starting transcription with diarization (FORCED large-v3) ===")
        logger.info(f"Job {job_id}: Starting transcription request")
        
        if not whisperx_available:
            return jsonify({
                'job_id': job_id,
                'error': 'WhisperX is not available on this server',
                'status': 'failed'
            }), 500
        
        # Validate file upload
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get parameters
        language = request.form.get('language', 'en')
        requested_model = request.form.get('model', 'large-v3')
        device = request.form.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        compute_type = request.form.get('compute_type', 'float16' if device == 'cuda' else 'int8')
        enable_diarization = request.form.get('diarization', 'true').lower() == 'true'
        
        # Force model to large-v3
        forced_model = 'large-v3'
        if requested_model != forced_model:
            print(f"⚠️  Job {job_id}: Model '{requested_model}' requested but forcing to use '{forced_model}'")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
        
        print(f"Job {job_id}: Saving file to {file_path}")
        file.save(file_path)
        
        # Verify file was saved
        if not os.path.exists(file_path):
            raise Exception(f"Failed to save uploaded file to {file_path}")
        
        file_size = os.path.getsize(file_path)
        print(f"Job {job_id}: File saved successfully, size: {file_size} bytes")
        
        # Transcribe with diarization using FORCED model
        print(f"Job {job_id}: Starting transcription with diarization (model forced: {forced_model}, diarization: {enable_diarization})")
        result = transcribe_audio_with_diarization(
            file_path,
            language=language,
            model_name=forced_model,  # Use forced model
            device=device,
            compute_type=compute_type,
            enable_diarization=enable_diarization
        )
        
        print(f"Job {job_id}: ✅ Transcription completed successfully")
        logger.info(f"Job {job_id}: Transcription completed successfully")
        
        # Return result
        response_data = {
            'job_id': job_id,
            'status': 'completed',
            'filename': filename,
            'language': result.get('language', language),
            'model_requested': requested_model,
            'model_used': forced_model,
            'model_forced': True,
            'device': model_device,
            'segments_count': len(result.get('segments', [])),
            'speaker_stats': result.get('speaker_stats', {}),
            'debug_info': result.get('debug_info', {}),
            'diarization_enabled': enable_diarization,
            'result': result
        }
        
        # Log the response for debugging
        print(f"Job {job_id}: Response segments count: {response_data['segments_count']}")
        print(f"Job {job_id}: Speakers detected: {len(response_data['speaker_stats'])}")
        print(f"Job {job_id}: Model used: {forced_model} (forced from {requested_model})")
        print(f"Job {job_id}: Debug info: {response_data['debug_info']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Job {job_id}: Transcription failed: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        traceback.print_exc()
        return jsonify({
            'job_id': job_id,
            'error': error_msg,
            'status': 'failed'
        }), 500
        
    finally:
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Job {job_id}: Cleaned up file {file_path}")
            except Exception as e:
                print(f"Job {job_id}: Warning - Could not remove upload file: {e}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    error_msg = f"Unhandled exception: {e}"
    print(f"❌ {error_msg}")
    logger.error(error_msg)
    traceback.print_exc()
    return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("WhisperX API Server with Diarization Starting")
    print("🔒 MODEL FORCED TO: large-v3 ONLY")
    print("=" * 60)
    print(f"WhisperX available: {whisperx_available}")
    print(f"HF_TOKEN available: {bool(HF_TOKEN)}")
    print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
    print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'not set')}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if not HF_TOKEN:
        print("⚠️  WARNING: HF_TOKEN not set - diarization will be disabled")
        print("   Set environment variable: export HF_TOKEN='your_huggingface_token'")
        print("   Get token from: https://huggingface.co/settings/tokens")
    
    print("🔒 IMPORTANT: All model requests will be FORCED to use 'large-v3'")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    except Exception as e:
        print(f"FATAL: Failed to start server: {e}")
        traceback.print_exc()
        sys.exit(1)
