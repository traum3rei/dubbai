import os
import json

def write_transcription_to_json(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path

def generate_pipeline_paths(video_path, base_output="../output"):
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    root_dir = os.path.join(base_output, video_title)

    return {
        "root": root_dir,
        "audio_raw": os.path.join(root_dir, "audio", "original_unprocessed.wav"),
        "audio_vocals": os.path.join(root_dir, "audio", "vocals_raw.wav"),
        "audio_vocals_48k": os.path.join(root_dir, "audio", "vocals_48k.wav"),
        "audio_vocals_denoised": os.path.join(root_dir, "audio", "vocals_denoised.wav"),
        "video_muted": os.path.join(root_dir, "video", "video_muted.mp4"),
        "video_final": os.path.join(root_dir, "video", "video_final.mp4"),
        "transcript_json": os.path.join(root_dir, "transcripts", "transcript.json"),
        "demucs_out": os.path.join(root_dir, "demucs")
    }
