"""Training script for the Image Detection module (EfficientNet-B0 on CIFAKE).

Converts the ad-hoc Colab logic from ``Image_Detection_Model(V1).ipynb`` into a
reusable, non-notebook script so the checkpoint can be retrained beyond the
original 1-epoch / 20k-subset proof of concept.

Example
-------
::

    python scripts/train.py \\
        --data-root /path/to/CIFAKE \\
        --epochs 8 \\
        --subset-size 0 \\
        --out checkpoints/image_detector_v2.pth

``--subset-size 0`` trains on the full training split. The original run used
``--subset-size 20000 --epochs 1``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
import torchvision.transforms as transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model() -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return train_transform, test_transform


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_transform, test_transform = build_transforms()
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    print(f"Classes: {train_dataset.classes} (index order matters — this is the label convention)")

    if args.subset_size and args.subset_size > 0:
        indices = random.sample(range(len(train_dataset)), min(args.subset_size, len(train_dataset)))
        train_dataset = Subset(train_dataset, indices)
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={running_loss:.4f}  ({elapsed:.1f}s)")

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes, output_dict=True)
    matrix = confusion_matrix(all_labels, all_preds).tolist()
    print(f"Test accuracy: {accuracy:.4f}")
    print(json.dumps(report, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to: {out_path.resolve()}")

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix,
                "epochs": args.epochs,
                "subset_size": args.subset_size,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
            },
            indent=2,
        )
    )
    print(f"Saved metrics to: {metrics_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Image Detection EfficientNet-B0 model on CIFAKE.")
    parser.add_argument("--data-root", required=True, help="Path to CIFAKE root containing train/ and test/ subfolders.")
    parser.add_argument("--out", default="checkpoints/image_detector_v2.pth", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (V1 used 1).")
    parser.add_argument("--subset-size", type=int, default=0, help="Cap on training samples; 0 = use full training split (V1 used 20000).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
