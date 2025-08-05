import os
import shutil
from gradio_client import Client, handle_file
from flask import current_app as app

def synthesize_speech_with_fish_speech(text, reference_audio_path, output_dir, job_id=None, api_url="http://fish-speech:7860/"):
    """
    Synthesize speech using Fish Speech via Gradio client.
    Returns the path to the synthesized WAV file.
    """
    
    app.logger.info(f"JOB {job_id}: Synthesizing speech. Text: '{text[:50]}...', Reference: {reference_audio_path}, Output Dir: {output_dir}")
    try:
        client = Client(api_url)
        reference_text = ""
        app.logger.info(f"JOB {job_id}: Calling Fish Speech API predict...")
        result = client.predict(
            text=text,
            reference_id="",
            reference_audio=handle_file(reference_audio_path),
            reference_text=reference_text,
            max_new_tokens=0, chunk_length=200, top_p=0.5,
            repetition_penalty=1.7, temperature=0.6, seed=0,
            use_memory_cache="on",
            api_name="/partial"
        )
        app.logger.info(f"JOB {job_id}: Fish Speech raw response: {result}")

        wav_path_from_fish_speech = None
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
            raise ValueError(f"JOB {job_id}: No .wav file path found in Fish Speech response: {result}")

        app.logger.info(f"JOB {job_id}: Fish Speech returned WAV path: {wav_path_from_fish_speech}")

        if not os.path.exists(wav_path_from_fish_speech):
            app.logger.error(f"JOB {job_id}: Synthesized WAV file NOT FOUND at {wav_path_from_fish_speech}. CWD: {os.getcwd()}")
            try:
                gradio_tmp_dir = os.path.dirname(wav_path_from_fish_speech)
                if "gradio" in gradio_tmp_dir and os.path.exists(gradio_tmp_dir):
                    app.logger.info(f"JOB {job_id}: Contents of {gradio_tmp_dir}: {os.listdir(gradio_tmp_dir)}")
            except Exception as list_err:
                app.logger.warning(f"JOB {job_id}: Could not list Gradio temp dir: {list_err}")
            raise FileNotFoundError(f"Synthesized WAV file not found: {wav_path_from_fish_speech}")

        os.makedirs(output_dir, exist_ok=True)
        final_output_wav = os.path.join(output_dir, "output.wav")
        app.logger.info(f"JOB {job_id}: Moving synthesized file from {wav_path_from_fish_speech} to {final_output_wav}")
        shutil.move(wav_path_from_fish_speech, final_output_wav)
        app.logger.info(f"JOB {job_id}: Successfully synthesized speech saved to {final_output_wav}")
        return final_output_wav

    except Exception as e:
        error_msg = f"JOB {job_id}: Speech synthesis failed: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e