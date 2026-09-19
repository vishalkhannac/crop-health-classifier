# src/train.py — Phase 4: Training
import sys, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / "src"))
from model import build_model

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10
SEED       = 42

MODEL_DIR  = ROOT / "model"
SPLITS_DIR = MODEL_DIR / "splits"

# ── Load splits ───────────────────────────────────────────────
print("Loading splits ...")
X_train = np.load(SPLITS_DIR / "X_train.npy", allow_pickle=True)
y_train = np.load(SPLITS_DIR / "y_train.npy", allow_pickle=True)
X_val   = np.load(SPLITS_DIR / "X_val.npy",   allow_pickle=True)
y_val   = np.load(SPLITS_DIR / "y_val.npy",   allow_pickle=True)

labels  = [l.strip() for l in (MODEL_DIR / "labels.txt").read_text().splitlines() if l.strip()]
num_classes = len(labels)
print(f"Classes: {num_classes}  |  Train: {len(X_train)}  |  Val: {len(X_val)}")


# ── Image loader ─────────────────────────────────────────────
def load_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, num_classes)


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(paths, labels, augment_flag=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 10000), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_flag:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# For an agile, working Review 1 training run, use a representative slice for fast CPU iteration
max_train = min(len(X_train), 15000)
max_val   = min(len(X_val), 3000)

train_ds = make_dataset(X_train[:max_train], y_train[:max_train], augment_flag=True, shuffle=True)
val_ds   = make_dataset(X_val[:max_val],     y_val[:max_val],     augment_flag=False, shuffle=False)

# ── Build & compile ───────────────────────────────────────────
model = build_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ─────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs ({max_train} train samples, {max_val} val samples) ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

val_acc = history.history["val_accuracy"][-1]
print(f"\nFinal validation accuracy: {val_acc*100:.2f}%")

# ── Save model ────────────────────────────────────────────────
model_path = MODEL_DIR / "model.keras"
model.save(str(model_path))
print(f"Model saved to {model_path}")

# ── Save training curves ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
plt.tight_layout()
curves_path = MODEL_DIR / "training_curves.png"
plt.savefig(str(curves_path), dpi=100)
print(f"Curves saved to {curves_path}")
