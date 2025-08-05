from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_audio(input_wav_path, output_wav_path):
    model, df_state, _ = init_df()
    audio, _ = load_audio(input_wav_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(output_wav_path, enhanced, df_state.sr())
    return output_wav_path
