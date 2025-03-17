import os
import numpy as np
import utils
import auditory
import plotslib
import pickle
import matplotlib.pyplot as plt

# Define the directory containing the audio files
# audio_dir = "../data"
audio_dir = "../preprocess/preprocessed_audio/preprocess_audio_erik/"

# Define the rates and scales vectors
rates_vec = [
    -32,
    -22.6,
    -16,
    -11.3,
    -8,
    -5.70,
    -4,
    -2,
    -1,
    -0.5,
    -0.25,
    0.25,
    0.5,
    1,
    2,
    4,
    5.70,
    8,
    11.3,
    16,
    22.6,
    32,
]
scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]

# Initialize accumulators for scale-rate, freq-rate, and freq-scale
total_scale_rate = np.zeros((len(scales_vec), len(rates_vec)))
total_freq_rate = np.zeros((128, len(rates_vec)))
total_freq_scale = np.zeros((128, len(scales_vec)))

# Initialize a counter for the number of files
num_files = 0

# Loop through all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        # Construct the full file path
        wav_file = os.path.join(audio_dir, filename)

        # Load the audio file
        audio, fs = utils.audio_data(wav_file)

        # Compute the STRF
        strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(
            audio, audio_fs=fs, duration=15, rates=rates_vec, scales=scales_vec
        )

        # Compute the average STRF
        magnitude_strf = np.abs(strf)
        real_valued_strf = np.mean(magnitude_strf, axis=0)

        # Convert STRF to average vectors
        avgvec = plotslib.strf2avgvec(strf)
        strf_scale_rate, strf_freq_rate, strf_freq_scale = plotslib.avgvec2strfavg(
            avgvec, nbScales=len(scales_vec), nbRates=len(rates_vec)
        )

        # Accumulate the results
        total_scale_rate += strf_scale_rate
        total_freq_rate += strf_freq_rate
        total_freq_scale += strf_freq_scale

        # Increment the file counter
        num_files += 1

# Average the results
avg_scale_rate = total_scale_rate / num_files
avg_freq_rate = total_freq_rate / num_files
avg_freq_scale = total_freq_scale / num_files

# Plot the averaged results
plotslib.plotStrfavgEqual(avg_scale_rate, avg_freq_rate,
                          avg_freq_scale, cmap="viridis")

# Save the averaged results
with open("avg_strf_data.pkl", "wb") as f:
    pickle.dump(
        {
            "avg_scale_rate": avg_scale_rate,
            "avg_freq_rate": avg_freq_rate,
            "avg_freq_scale": avg_freq_scale,
        },
        f,
    )

# averaged scale-rate representation
plt.figure(figsize=(8, 6))
plt.imshow(
    avg_scale_rate,
    aspect="auto",
    origin="lower",
    extent=[rates_vec[0], rates_vec[-1], scales_vec[0], scales_vec[-1]],
    interpolation="gaussian",
    cmap="viridis",
)
plt.colorbar(label="Modulation Energy (Amplitude)")
plt.xlabel("Temporal Modulation Rate (Hz)")
plt.ylabel("Spectral Modulation Scale (cyc/oct)")
plt.title("Averaged Rate-Scale Representation (avg_scale_rate)")
plt.show()
