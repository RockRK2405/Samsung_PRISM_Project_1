"""
model.py
A lightweight CNN for REAL vs FAKE classification from log-Mel spectrograms.
Parameter budget (~1.2M) is intentionally kept small, in the same
cost-aware spirit as the EfficientNet-B0 choice used for the Image module.

Track B (optional, free, higher accuracy):
  Replace this model with a frozen `facebook/wav2vec2-base` feature
  extractor (via HuggingFace `transformers`) + a small MLP head
  (Wav2Vec2Head below). Only the head is trained -> cheap even on CPU.
"""
import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Wav2Vec2Head(nn.Module):
    """
    Track B (optional): small MLP head trained on top of FROZEN Wav2Vec2
    embeddings (768-dim mean-pooled). Wire this in only if you want to try
    the higher-accuracy backbone; still free on Colab/Kaggle since the
    backbone itself is never fine-tuned.
    """
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = AudioCNN()
    dummy = torch.randn(2, 1, 128, 251)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", out.shape)
    print(f"Total parameters: {n_params:,}")
