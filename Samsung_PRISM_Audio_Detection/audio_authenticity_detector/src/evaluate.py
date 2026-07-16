"""
evaluate.py
Loads the trained checkpoint and reports the same metric set your
Image/Text modules report: Accuracy, F1, Precision, Recall, ROC-AUC,
False Positive Rate, plus a confusion matrix plot.

Usage:
    python src/evaluate.py --features_dir outputs/features --ckpt outputs/best_audio_model.pt
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

from dataset import AudioSpectrogramDataset
from model import AudioCNN


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(args.features_dir, "metadata.csv"))
    _, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_ds = AudioSpectrogramDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("=== Audio Detector Evaluation ===")
    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"F1 Score            : {f1*100:.2f}%")
    print(f"Precision           : {precision*100:.2f}%")
    print(f"Recall               : {recall*100:.2f}%")
    print(f"ROC-AUC              : {roc_auc*100:.3f}%")
    print(f"False Positive Rate : {fpr*100:.2f}%  (target <5%)")
    print(f"Confusion Matrix:\n{cm}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Audio Detector Confusion Matrix")
    out_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Confusion matrix plot saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--ckpt", type=str, default="outputs/best_audio_model.pt")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
