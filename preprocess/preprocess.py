import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence


def check_audio_extension(input_file):
    ext = os.path.splitext(input_file)[-1].lower()
    if ext != ".wav":
        audio = AudioSegment.from_file(input_file)
        output_file = os.path.splitext(input_file)[0] + ".wav"
        audio.export(output_file, format="wav")
        print(f"Converted to WAV: {output_file}")
        return output_file
    return input_file


def load_audio_with_soundfile(input_file):
    y, sr = sf.read(input_file, always_2d=True)  # Ensure 2D output
    y = np.mean(y, axis=1)  # Convert stereo to mono
    print(f"Sampeling rate: {sr} Hz")
    return y, sr


def get_unique_output_dir(base_dir):
    if not base_dir:
        return None

    counter = 1
    output_dir = base_dir
    while os.path.exists(output_dir):
        output_dir = f"{base_dir.rstrip('/')}_{counter}/"
        counter += 1
    os.makedirs(output_dir)
    return output_dir


def remove_silence(input_file, silence_thresh=-40, min_silence_len=500):
    """
    Removes silent segments form the audio file.
    """
    audio = AudioSegment.from_file(input_file)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100,  # Keep 100 ms of silence at the start/end of each chunk
    )
    return sum(chunks)


# Define preprocessing function
def preprocess_audio(input_file, output_dir="", segment_length=15, target_sr=16000):
    """
    Preprocesses an audio file by performing noise reduction, segmentation (15s), amplitude normalization, silence removal, and downsampling (44.1kHz to 16kHz).

    - Converts non-WAV files to WAV
    - Performs noise reduction
    - Downsamples (44.1kHz → 16kHz)
    - Segments audio into 15s chunks
    - Handles both single files and directories

    Args:
        input_path (str): Path to an audio file or directory.
        output_dir (str): Directory to save the processed audio segments.
        segment_length (int): Length of each segment in seconds (default is 15s).
        target_sr (int): Target sampling rate (default is 16kHz).

    Returns:
        lists: A list of processed audio segments (NumPy arrays).
        int: The sampling rate of the processed segments.
    """

    output_dir = get_unique_output_dir(output_dir)
    input_file = check_audio_extension(input_file)

    # Remove silence from the audio
    audio_no_silence = remove_silence(input_file)
    temp_file = os.path.join(output_dir, "temp_no_silence.wav")
    audio_no_silence.export(temp_file, format="wav")

    y, sr = load_audio_with_soundfile(input_file)

    # y, sr = librosa.load(input_file, sr=None)

    # Apply noise reduction using spectral gating
    # y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Noise reduction in chunks (if audio is long)
    chunk_size = sr * 5  # Process 5-second chunks
    y_denoised = np.concatenate(
        [
            nr.reduce_noise(y[i : i + chunk_size], sr=sr)
            for i in range(0, len(y), chunk_size)
        ]
    )

    # Normalize amplitudes to [-1, 1]
    y_normalized = y_denoised / np.max(np.abs(y_denoised))

    # Resample from 44.1kHz to 16kHz if not in target sampling rate
    if sr != target_sr:
        y_normalized = librosa.resample(y_denoised, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # total_samples = len(y_denoised)

    # Get the base filename of the audio (excluding extension)
    audio_filename = os.path.splitext(os.path.basename(input_file))[0]

    # Calculate segment length in samples
    segment_samples = segment_length * sr

    # Split and save segments
    segments = []
    for i, start in enumerate(range(0, len(y_normalized), segment_samples)):
        end = start + segment_samples
        segment = y_normalized[start:end]
        if len(segment) == segment_samples:  # includes full-length segments only
            segments.append(segment)
            # Save segment to disk if output_dir is provided
            if output_dir:
                sf.write(
                    os.path.join(output_dir, f"{audio_filename}_segment_{i + 1}.wav"),
                    segment,
                    sr,
                )

    return segments, sr

    # # Split and save segments
    # for i, start in enumerate(range(1, total_samples, segment_samples)):
    #     end = start + segment_samples
    #     segment = y_normalized[start:end]
    #
    #     # Save only full-length segments
    #     if len(segment) == segment_samples:
    #         output_file = os.path.join(output_dir, f"segment_{i + 2}.wav")
    #         sf.write(output_file, segment, sr)
    #         print(f"Saved: {output_file}")


# input_audio = "Predi-COVID_0099_20200624100830_1_m4a_W_0.wav"
# input_audio = os.path.expanduser("~/Downloads/moby_mono.wav")
# input_audio = "../data/209.mp3"
# check_audio_extension(input_audio)
# preprocess_audio(source_dir, output_base_dir)

# source_dir = "/home/christian/Desktop/C_006/THESIS/Datasets/Predi_COVID19_Fatigue_Voice_Recording/fatigue/TypeW/Type1/"
output_base_dir = "preprocessed_audio/preprocess_audio/"
input_audio = "../data/Eriks_Voice.m4a"

segments, sr = preprocess_audio(input_audio, output_base_dir)

# Segmented Audio
# for file in os.listdir(source_dir):
#     if file.endswith(".wav") or file.endswith(".m5a") or file.endswith(".3gp"):
#        input_path = os.path.join(source_dir, file)
#         preprocess_audio(input_path, output_base_dir)
