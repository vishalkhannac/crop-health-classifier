# src/evaluate.py — Phase 5: Evaluate trained model on test set
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE   = 224
BATCH_SIZE = 32

MODEL_DIR  = ROOT / "model"
SPLITS_DIR = MODEL_DIR / "splits"

# ── Load test set ─────────────────────────────────────────────
X_test = list(np.load(SPLITS_DIR / "X_test.npy", allow_pickle=True).astype(str))
y_test = np.load(SPLITS_DIR / "y_test.npy").astype(np.int32)

labels = [l.strip() for l in (MODEL_DIR / "labels.txt").read_text().splitlines() if l.strip()]
num_classes = len(labels)

def load_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, num_classes)

eval_X = X_test
eval_y = y_test

test_ds = (
    tf.data.Dataset.from_tensor_slices((eval_X, eval_y))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── Load model & evaluate ─────────────────────────────────────
print("Loading model ...")
model = tf.keras.models.load_model(str(MODEL_DIR / "model.keras"))

print("Evaluating on test set ...")
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\n{'='*40}")
print(f"TEST ACCURACY : {acc*100:.2f}%")
print(f"TEST LOSS     : {loss:.4f}")
print(f"{'='*40}\n")

# ── Per-class report ──────────────────────────────────────────
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(eval_y, y_pred, target_names=labels, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────
cm = confusion_matrix(eval_y, y_pred)
fig, ax = plt.subplots(figsize=(max(12, num_classes // 2), max(10, num_classes // 2)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation="vertical", colorbar=False)
plt.title(f"Confusion Matrix — Test Accuracy: {acc*100:.1f}%")
plt.tight_layout()
cm_path = MODEL_DIR / "confusion_matrix.png"
plt.savefig(str(cm_path), dpi=80)
print(f"Confusion matrix saved to {cm_path}")
