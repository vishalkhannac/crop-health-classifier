# src/explainability.py — Grad-CAM Visual Attention Heatmap Engine
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


def generate_gradcam(pil_img: Image.Image, model: tf.keras.Model, img_size: int = 224, alpha: float = 0.42) -> Image.Image:
    """
    Computes a Gradient-weighted Class Activation Map (Grad-CAM) from the MobileNetV2
    backbone's final convolutional feature map (7x7x1280) with respect to the top predicted class,
    and returns an RGB PIL Image with the JET heatmap superimposed onto the original image.
    """
    try:
        orig_rgb = np.array(pil_img.convert("RGB"))
        orig_h, orig_w = orig_rgb.shape[:2]

        resized = pil_img.convert("RGB").resize((img_size, img_size))
        img_tensor = tf.cast(np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0), tf.float32)

        # Find MobileNetV2 base model layer and classifier head layers
        base_model = None
        head_layers = []
        found_base = False
        for layer in model.layers:
            if not found_base and isinstance(layer, tf.keras.Model):
                base_model = layer
                found_base = True
            elif found_base:
                head_layers.append(layer)

        if base_model is None:
            # Fallback if model is flat
            return pil_img

        with tf.GradientTape() as tape:
            conv_outputs = base_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            x = conv_outputs
            for hl in head_layers:
                x = hl(x, training=False)
            preds = x
            top_class_idx = tf.argmax(preds[0])
            top_class_Channel = preds[:, top_class_idx]

        grads = tape.gradient(top_class_Channel, conv_outputs)
        if grads is None:
            return pil_img

        # Global average pooling of gradients over spatial dimensions (1, 7, 7, 1280) -> (1280,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]  # (7, 7, 1280)

        # Weight feature map channels by gradient importance
        heatmap = tf.reduce_sum(tf.multiply(conv_out, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_u8 = np.uint8(255 * heatmap_resized)
        colormap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)

        # Blend original image with heatmap, emphasizing high-activation regions
        weight_map = np.clip(heatmap_resized[:, :, np.newaxis] * alpha + 0.12, 0.12, 0.65)
        blended = (orig_rgb * (1.0 - weight_map) + colormap_rgb * weight_map).astype(np.uint8)
        return Image.fromarray(blended)
    except Exception:
        return pil_img
