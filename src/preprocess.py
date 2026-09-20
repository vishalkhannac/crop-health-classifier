# src/preprocess.py — Phase 2: Build train/val/test generators, save labels.txt
import os, sys, json, shutil, random
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
IMG_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")

def get_class_images(class_dir):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(class_dir.glob(ext))
    return imgs

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and len(get_class_images(d)) > 0])
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

print(f"Found {num_classes} classes.")

# ── Collect balanced image paths and labels across all classes ──
MAX_PER_CLASS = 150  # Balanced representation across all 108 classes
all_paths, all_labels = [], []
for cls in classes:
    imgs = get_class_images(DATA_DIR / cls)
    if len(imgs) > MAX_PER_CLASS:
        # Prioritize sliced/multi/user augmented samples
        priority = [p for p in imgs if any(k in p.name.lower() for k in ("user_", "sliced_", "multi_"))]
        others = [p for p in imgs if p not in priority]
        random.seed(SEED)
        sampled_others = random.sample(others, max(0, MAX_PER_CLASS - len(priority))) if len(priority) < MAX_PER_CLASS else []
        imgs = (priority + sampled_others)[:MAX_PER_CLASS]
    all_paths.extend([str(p) for p in imgs])
    all_labels.extend([class_to_idx[cls]] * len(imgs))

all_paths  = np.array(all_paths, dtype=object)
all_labels = np.array(all_labels, dtype=np.int32)
total = len(all_paths)
print(f"Total balanced images selected: {total}")

# ── Stratified 70/15/15 split ────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Save splits to disk so train.py / evaluate.py can reuse them ──
splits_dir = ROOT / "model" / "splits"
splits_dir.mkdir(exist_ok=True)
np.save(splits_dir / "X_train.npy", X_train, allow_pickle=True)
np.save(splits_dir / "y_train.npy", y_train)
np.save(splits_dir / "X_val.npy",   X_val, allow_pickle=True)
np.save(splits_dir / "y_val.npy",   y_val)
np.save(splits_dir / "X_test.npy",  X_test, allow_pickle=True)
np.save(splits_dir / "y_test.npy",  y_test)

# ── Save class label list ─────────────────────────────────────
labels_path = MODEL_DIR / "labels.txt"
labels_path.write_text("\n".join(classes), encoding="utf-8")
print(f"Saved {num_classes} class labels to {labels_path}")

print("\nPreprocessing complete.")
print(f"  Classes : {num_classes}")
print(f"  Train   : {len(X_train)}")
print(f"  Val     : {len(X_val)}")
print(f"  Test    : {len(X_test)}")
