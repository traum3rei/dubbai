import os
import subprocess
from threading import Lock
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from media_pipeline.pipeline_controller import MediaPipelineController
import shlex
import shutil
import json
from threading import Lock
from datetime import datetime
import uuid
from gradio_client import Client, handle_file
import torch
import whisper as whisper 


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/uploads'
app.config['OUTPUT_FOLDER'] = '/app/output'
app.config['STATUS_FOLDER'] = '/app/status'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATUS_FOLDER'], exist_ok=True)

status_lock = Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav', 'ogg', 'flac', 'mp4', 'mkv', 'avi'}


pipeline = MediaPipelineController(base_output=app.config['OUTPUT_FOLDER'])

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
                
                app.logger.info(f"Job {job_id}: File '{file.filename}' uploaded. Language: {language}, Model: {whisper_model}, RefChoice: {reference_choice}")

                from threading import Thread
                pipeline.run_full_pipeline_background(
                    input_file_path,
                    language=language,
                    job_id=job_id,
                    status_callback=update_status
                )
                
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

if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5000, debug=True) # Debug=True is not for production
