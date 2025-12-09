# classifier/train_resnet50.py
"""
Train a ResNet-50 classifier on cropped bird / no-bird patches.

Input (from prepare_crops.py):
    classifier/crop_dataset/
        train/bird/*.jpg
        train/no-bird/*.jpg
        val/bird/*.jpg
        val/no-bird/*.jpg
        test/bird/*.jpg
        test/no-bird/*.jpg

Output:
    classifier/resnet50_best.pth  – best validation-accuracy model
    classifier/resnet50_last.pth  – last epoch model
"""

import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --- make project root importable so "from utils.config" works ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------

from utils.config import (
    CROP_TRAIN_DIR,
    CROP_VAL_DIR,
    CROP_TEST_DIR,
    CLASS_NAMES,
    CLF_IMAGE_SIZE,
)


# ----------------- Hyperparameters -----------------
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = PROJECT_ROOT / "classifier"
BEST_MODEL_PATH = OUT_DIR / "resnet50_best.pth"
LAST_MODEL_PATH = OUT_DIR / "resnet50_last.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------


def get_dataloaders():
    """
    Create train / val / test dataloaders from ImageFolder directories.
    """

    # standard ImageNet-style transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((CLF_IMAGE_SIZE, CLF_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((CLF_IMAGE_SIZE, CLF_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(str(CROP_TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(CROP_VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(CROP_TEST_DIR), transform=eval_transform)

    print("[INFO] Class-to-index mapping:", train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_model(num_classes: int):
    """
    Load ResNet-50 (pretrained on ImageNet), replace the final layer for our classes.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    print("[INFO] Device:", DEVICE)

    train_loader, val_loader, test_loader = get_dataloaders()

    num_classes = len(CLASS_NAMES)
    print("[INFO] Number of classes:", num_classes, CLASS_NAMES)

    model = build_model(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # save last
        torch.save(model.state_dict(), LAST_MODEL_PATH)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[INFO] 🔥 New best val acc: {best_val_acc:.4f}, saved to {BEST_MODEL_PATH}")

    print("\n[INFO] Training finished.")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")

    # ---------- Final test evaluation (using best model) ----------
    best_model = build_model(num_classes=num_classes).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    test_loss, test_acc = evaluate(best_model, test_loader, criterion)
    print(f"[INFO] Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
