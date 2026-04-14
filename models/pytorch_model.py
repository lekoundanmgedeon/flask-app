"""
PyTorch CNN Pipeline for Intel Image Classification
Architecture: Custom CNN with BatchNorm, Dropout, and residual-style connections
Saved to: your_firstname_model.pth
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─────────────────────────────────────────────────────────────
# 1.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)


def get_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int):
    """
    Build train / val / test DataLoaders with augmentation for train set.

    Expected directory structure:
        data_dir/
            seg_train/seg_train/<class>/...
            seg_test/seg_test/<class>/...
            seg_pred/seg_pred/...          (unlabelled – skipped here)
    """

    # ── Augmentation for training ──────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        # ImageNet-style normalisation works well as a starting point
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Only resize + centre-crop for val / test ───────────────
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Locate dataset folders (handle both flat and nested Kaggle layouts)
    def _find_folder(root, candidates):
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(
            f"Could not find any of {candidates} inside '{root}'. "
            "Please check --data_dir."
        )

    train_dir = _find_folder(data_dir, ["seg_train/seg_train", "seg_train", "train"])
    test_dir  = _find_folder(data_dir, ["seg_test/seg_test",  "seg_test",  "test"])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    # Split 10 % of training data for validation
    val_size  = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Val subset should use eval transforms – wrap it
    val_subset.dataset = copy.deepcopy(train_dataset)
    val_subset.dataset.transform = eval_transform

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"\n[Data] Train samples : {len(train_subset)}")
    print(f"[Data] Val   samples : {len(val_subset)}")
    print(f"[Data] Test  samples : {len(test_dataset)}")
    print(f"[Data] Classes       : {train_dataset.classes}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# 2.  MODEL DEFINITION
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU (→ optional MaxPool)."""
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class IntelCNN(nn.Module):
    """
    Custom CNN for 6-class scene classification.

    Architecture overview
    ─────────────────────
    Stage 1  :  3 → 32 → 64   (150×150 → 75×75)
    Stage 2  : 64 → 128 → 128 (75×75  → 37×37)
    Stage 3  :128 → 256 → 256 (37×37  → 18×18)
    Stage 4  :256 → 512       (18×18  →  9×9 )
    Head     : GAP → FC(512→256) → Dropout(0.4) → FC(256→6)
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBlock(3,  32),
            ConvBlock(32, 64, pool=True),   # /2
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64,  128),
            ConvBlock(128, 128, pool=True),  # /2
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=True),  # /2
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512, pool=True),  # /2
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────
# 3.  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PyTorch] Training plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_pytorch_pipeline(args):
    # ── Device ────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n[PyTorch] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────
    model = IntelCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PyTorch] Model parameters: {total_params:,}")

    # ── Loss / Optimiser / Scheduler ──────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[PyTorch] Starting training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  │  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best val acc: {best_val_acc:.2f}%")

    # ── Load best weights & evaluate on test set ──────────────
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[PyTorch] Test Accuracy: {test_acc:.2f}%  |  Test Loss: {test_loss:.4f}")

    # ── Save model ────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "your_firstname_model.pth")
    torch.save({
        "model_state_dict":      best_model_wts,
        "optimizer_state_dict":  optimizer.state_dict(),
        "class_names":           CLASS_NAMES,
        "img_size":              args.img_size,
        "best_val_acc":          best_val_acc,
        "test_acc":              test_acc,
        "args":                  vars(args),
    }, save_path)
    print(f"[PyTorch] Model saved → {save_path}")

    # ── Plot ──────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "pytorch_training_history.png")
    plot_history(history, plot_path)

    print(f"\n[PyTorch] Best Val Acc : {best_val_acc:.2f}%")
    print(f"[PyTorch] Test Acc     : {test_acc:.2f}%")
    return model
