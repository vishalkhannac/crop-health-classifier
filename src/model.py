# src/model.py — Phase 3: MobileNetV2 transfer-learning model
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_model(num_classes: int, img_size: int = 224) -> tf.keras.Model:
    """
    MobileNetV2 base (frozen) + custom softmax head.
    Input: (img_size, img_size, 3), pixels in [0, 1].
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="plant_veg_health")
    model.base_model = base
    return model


if __name__ == "__main__":
    # Quick smoke-test: build for a dummy num_classes
    import json
    labels_path = ROOT / "model" / "labels.txt"
    if labels_path.exists():
        classes = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        num_classes = len(classes)
    else:
        num_classes = 54   # fallback for testing before Phase 2
    m = build_model(num_classes)
    m.summary()
    trainable = sum(tf.keras.backend.count_params(w) for w in m.trainable_weights)
    non_train  = sum(tf.keras.backend.count_params(w) for w in m.non_trainable_weights)
    print(f"\nNum classes : {num_classes}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {non_train:,}")
