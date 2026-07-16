"""
preprocess.py
Converts raw .wav files (real/ and fake/ folders) into fixed-size log-Mel
spectrogram tensors saved as .npy files, plus a metadata CSV (path, label).

Usage:
    python src/preprocess.py --data_dir data --out_dir outputs/features

Expected input layout:
    data/
      real/  *.wav
      fake/  *.wav
"""
import argparse
import os
import glob
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE = 16000
DURATION_SEC = 4.0          # fixed window length
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def load_fixed_length_audio(path, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """Load audio, resample, then pad/trim to a fixed length."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) >= n_samples:
        # take a random-ish but deterministic centered window
        start = max(0, (len(y) - n_samples) // 2)
        y = y[start:start + n_samples]
    else:
        pad = n_samples - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    return y


def wav_to_logmel(y, sr=SAMPLE_RATE):
    """Convert waveform to a normalized log-Mel spectrogram (N_MELS x T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def process_folder(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for label_name, label_id in [("real", 0), ("fake", 1)]:
        folder = os.path.join(data_dir, label_name)
        files = (
            glob.glob(os.path.join(folder, "*.wav"))
            + glob.glob(os.path.join(folder, "*.flac"))
            + glob.glob(os.path.join(folder, "*.mp3"))
        )
        print(f"[{label_name}] found {len(files)} files")

        for fp in tqdm(files, desc=f"Processing {label_name}"):
            try:
                y = load_fixed_length_audio(fp)
                feat = wav_to_logmel(y)
                out_name = os.path.splitext(os.path.basename(fp))[0] + ".npy"
                out_path = os.path.join(out_dir, f"{label_name}_{out_name}")
                np.save(out_path, feat)
                records.append({"feature_path": out_path, "label": label_id})
            except Exception as e:
                print(f"  skipped {fp}: {e}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} feature files. Metadata -> {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                         help="folder containing real/ and fake/ subfolders")
    parser.add_argument("--out_dir", type=str, default="outputs/features",
                         help="where to save .npy features + metadata.csv")
    args = parser.parse_args()
    process_folder(args.data_dir, args.out_dir)
