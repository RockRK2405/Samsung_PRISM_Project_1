"""
infer.py
Runs the trained Audio Detector on a single .wav file and prints an output
in the schema expected by the project's architecture (Step 3.3 Audio
Analysis -> "Audio authenticity score, Synthetic speech probability").

Usage:
    python src/infer.py --wav path/to/clip.wav --ckpt outputs/best_audio_model.pt
"""
import argparse
import json
import torch

from preprocess import load_fixed_length_audio, wav_to_logmel
from dataset import AudioSpectrogramDataset
from model import AudioCNN


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y = load_fixed_length_audio(args.wav)
    mel = wav_to_logmel(y)

    # reuse the dataset's fixed-time-dim logic for consistent shape
    fixer = AudioSpectrogramDataset.__new__(AudioSpectrogramDataset)
    fixer.fixed_frames = 251
    mel = fixer._fix_time_dim(mel)

    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,n_mels,time)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        fake_prob = probs[1].item()
        real_prob = probs[0].item()
        pred_label = "FAKE" if fake_prob >= 0.5 else "REAL"

    result = {
        "file": args.wav,
        "label": pred_label,
        "audio_authenticity_score": round(real_prob, 4),
        "synthetic_speech_probability": round(fake_prob, 4),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="outputs/best_audio_model.pt")
    args = parser.parse_args()
    main(args)
