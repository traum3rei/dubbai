import whisper

def load_whisper_model(model_size="base"):
    return whisper.load_model(model_size)

def transcribe_audio(model, audio_file_path, language=None):
    options = {"verbose": True, "task": "transcribe"}
    if language:
        options["language"] = language
    return model.transcribe(audio_file_path, **options)

