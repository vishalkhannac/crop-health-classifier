# src/preprocess.py — Phase 2: Build train/val/test generators, save labels.txt
import os, sys, json, shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42

# ── Discover classes ──────────────────────────────────────────
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and any(d.glob("*.jpg"))])
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

print(f"Found {num_classes} classes.")

# ── Collect all image paths and labels ───────────────────────
all_paths, all_labels = [], []
for cls in classes:
    imgs = list((DATA_DIR / cls).glob("*.jpg"))
    all_paths.extend([str(p) for p in imgs])
    all_labels.extend([class_to_idx[cls]] * len(imgs))

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
total = len(all_paths)
print(f"Total images: {total}")

# ── Stratified 70/15/15 split ────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Save splits to disk so train.py / evaluate.py can reuse them ──
splits_dir = ROOT / "model" / "splits"
splits_dir.mkdir(exist_ok=True)
np.save(splits_dir / "X_train.npy", X_train); np.save(splits_dir / "y_train.npy", y_train)
np.save(splits_dir / "X_val.npy",   X_val);   np.save(splits_dir / "y_val.npy",   y_val)
np.save(splits_dir / "X_test.npy",  X_test);  np.save(splits_dir / "y_test.npy",  y_test)

# ── Save class label list ─────────────────────────────────────
labels_path = MODEL_DIR / "labels.txt"
labels_path.write_text("\n".join(classes), encoding="utf-8")
print(f"Saved {num_classes} class labels to {labels_path}")

print("\nPreprocessing complete.")
print(f"  Classes : {num_classes}")
print(f"  Train   : {len(X_train)}")
print(f"  Val     : {len(X_val)}")
print(f"  Test    : {len(X_test)}")
