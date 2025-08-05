# Main controller for the media processing pipeline 
# The Steps are as follows:
# Von MP4 zu Audio + Background
# Original MP4
#   ↓ Extract audio (keep original SR)
#   ↓ Separate vocals/background (at original SR)
#   ↓ Convert vocals & background stems → 48 kHz WAV
#   ↓ Denoise vocals (48 kHz WAV) for Synthesis
#   ↓ Transcripe text using the original Audio
# Audio + Background  

import shutil
from flask import current_app as app
import os
import threading
from media_pipeline.audio_extraction import separate_audio_from_video
from media_pipeline.vocal_separation import separate_vocals
from frontend.media_pipeline.audio_conversion import convert_to_48k_wav
from media_pipeline.audio_denoise import denoise_audio
from media_pipeline.transcription import load_whisper_model, transcribe_audio
from media_pipeline.utils import write_transcription_to_json, generate_pipeline_paths
import json

class MediaPipelineController:
    def __init__(self, whisper_model_size="base", base_output="../output"):
        self.whisper_model_size = whisper_model_size
        self.base_output = base_output
        self.whisper_model = load_whisper_model(self.whisper_model_size)
        
    def run_full_pipeline_background(self, video_path, language=None, job_id=None, status_callback=None):
        def target():
            with app.app_context():  # <-- Add this line
                try:
                    if status_callback and job_id:
                        status_callback(job_id, 'starting', message='Pipeline started.')
                    self.run_full_pipeline(video_path, language)
                    if status_callback and job_id:
                        status_callback(job_id, 'complete', message='Pipeline complete.')
                except Exception as e:
                    if status_callback and job_id:
                        status_callback(job_id, 'failed', message=str(e))
                    app.logger.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        thread = threading.Thread(target=target)
        thread.start()
        return thread

    def is_video_file(self, filepath):
        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
        return os.path.splitext(filepath)[1].lower() in video_exts

    def run_full_pipeline(self, video_path, language=None, job_id=None, status_callback=None):
        paths = generate_pipeline_paths(video_path, self.base_output)
        os.makedirs(paths["root"], exist_ok=True)
        
        processing_files = {
            # fill this dict as you go through the pipeline
            # e.g. 'original': ..., 'separated_vocals': ..., etc.
        }
        try:
            # Step 1: Extract audio and muted video (only if input is a video)
            if self.is_video_file(video_path):
                audio_raw, video_muted = separate_audio_from_video(video_path, paths["audio_raw"], paths["video_muted"])
            else:
                # Just copy the audio file to the expected location
                shutil.copy(video_path, paths["audio_raw"])
                audio_raw = paths["audio_raw"]
                video_muted = None  # No muted video for audio files

            processing_files['original'] = audio_raw

            # Step 2: Vocal separation
            app.logger.info(f"Separating vocals from {paths['audio_raw']}...")
            vocals_raw, _ = separate_vocals(audio_raw, output_dir=os.path.dirname(paths["audio_vocals"]))
            # Move/rename vocals output to expected path
            os.rename(vocals_raw, paths["audio_vocals"])
            app.logger.info(f"Vocals separated and saved to {paths['audio_vocals']}.")
            
            processing_files['separated_vocals'] = paths["audio_vocals"]
            
            # Step 3: Convert vocals to 48kHz WAV
            app.logger.info(f"Converting {paths['audio_vocals']} to 48kHz WAV at {paths['audio_vocals_48k']}...")
            convert_to_48k_wav(paths["audio_vocals"], paths["audio_vocals_48k"])
            app.logger.info("Conversion done.")

            processing_files['vocals_48k'] = paths["audio_vocals_48k"]

            # Step 4: Denoise vocals
            app.logger.info(f"Denoising audio {paths['audio_vocals_48k']} to {paths['audio_vocals_denoised']}...")
            denoise_audio(paths["audio_vocals_48k"], paths["audio_vocals_denoised"])
            app.logger.info("Denoising done.")

            processing_files['denoised_vocals'] = paths["audio_vocals_denoised"]

            # Step 5: Transcribe audio
            # You can choose to transcribe raw audio or denoised vocals but the raw audio is more accurate
            app.logger.info(f"Transcribing audio {paths['audio_raw']}...")
            transcript_result = transcribe_audio(self.whisper_model, paths["audio_raw"], language=language)
            app.logger.info("Transcription done.")

            # Step 6: Save transcript JSON
            app.logger.info(f"Writing transcription to JSON at {paths['transcript_json']}...")
            write_transcription_to_json(transcript_result, paths["transcript_json"])
            app.logger.info("Transcription JSON written.")
            
            processing_files['transcript'] = paths["transcript_json"]
            
            # At the end, write the file
            files_record_path = os.path.join(paths["root"], "processing_files.json")
            with open(files_record_path, 'w') as f:
                json.dump(processing_files, f, indent=4)
            if status_callback and job_id:
                status_callback(job_id, 'complete', message='Pipeline complete.')
        except Exception as e:
            # On error, write whatever you have so far for debugging
            files_record_path = os.path.join(paths["root"], "processing_files.json")
            try:
                with open(files_record_path, 'w') as f:
                    json.dump(processing_files, f, indent=4)
            except Exception:
                pass
            if status_callback and job_id:
                status_callback(job_id, 'failed', message=str(e))
            app.logger.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
            raise
