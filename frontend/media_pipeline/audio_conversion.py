import subprocess

def convert_to_48k_wav(input_audio_path, output_wav_path):
    subprocess.run([
        'ffmpeg', '-y', '-i', input_audio_path,
        '-ar', '48000',
        '-ac', '2',
        output_wav_path
    ], check=True, capture_output=True)
    return output_wav_path


