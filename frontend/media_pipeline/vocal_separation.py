import os
from flask import current_app as app
import shlex
import shutil
import demucs.separate

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


