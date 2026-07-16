"""
dataset.py
PyTorch Dataset for loading precomputed log-Mel spectrogram .npy features.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioSpectrogramDataset(Dataset):
    def __init__(self, dataframe, fixed_frames=251):
        """
        dataframe: pandas DataFrame with columns ['feature_path', 'label']
        fixed_frames: time dimension to pad/crop spectrograms to, so all
                      samples in a batch have identical shape.
        """
        self.df = dataframe.reset_index(drop=True)
        self.fixed_frames = fixed_frames

    def __len__(self):
        return len(self.df)

    def _fix_time_dim(self, mel):
        n_mels, t = mel.shape
        if t >= self.fixed_frames:
            mel = mel[:, :self.fixed_frames]
        else:
            pad_width = self.fixed_frames - t
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["feature_path"])
        mel = self._fix_time_dim(mel)
        mel = torch.from_numpy(mel).unsqueeze(0).float()  # (1, n_mels, time)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return mel, label
