import os
import subprocess

def separate_audio_from_video(video_path, audio_out_path, muted_video_out_path):
    os.makedirs(os.path.dirname(audio_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(muted_video_out_path), exist_ok=True)

    audio_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        audio_out_path
    ]

    muted_video_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-an", muted_video_out_path
    ]

    subprocess.run(audio_cmd, check=True)
    subprocess.run(muted_video_cmd, check=True)
    return audio_out_path, muted_video_out_path

