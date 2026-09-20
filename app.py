# app.py — Phase 7: Polished Streamlit App
# Plant & Vegetable Health Detection
# Predicts class, shows confidence meter, disease name, safe-to-consume verdict.

import sys
from pathlib import Path

# ── Resolve venv site-packages ────────────────────────────────
ROOT = Path(__file__).parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import streamlit as st
import numpy as np
from PIL import Image
import io

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Plant & Vegetable Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Gradient header bar */
  .header-bar {
    background: linear-gradient(135deg, #2d7a2d 0%, #4caf50 60%, #81c784 100%);
    border-radius: 12px;
    padding: 24px 32px 18px 32px;
    margin-bottom: 24px;
    color: white;
  }
  .header-bar h1 { color: white; margin: 0 0 4px 0; font-size: 2.1rem; }
  .header-bar p  { color: rgba(255,255,255,0.88); margin: 0; font-size: 1.05rem; }

  /* Result cards */
  .card {
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    font-size: 1.05rem;
  }
  .card-safe   { background: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20; }
  .card-unsafe { background: #fff3e0; border-left: 6px solid #e65100; color: #bf360c; }
  .card-class  { background: #e3f2fd; border-left: 6px solid #1565c0; color: #0d47a1; }
  .card-disease{ background: #fce4ec; border-left: 6px solid #880e4f; color: #560027; }
  .card-honesty{ background: #f5f5f5; border-left: 6px solid #9e9e9e; color: #616161; font-size: 0.9rem; }
  .card-label  { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.08em;
                 text-transform: uppercase; margin-bottom: 4px; opacity: 0.7; }
  .card-value  { font-size: 1.25rem; font-weight: 700; }
  .card-note   { font-size: 0.9rem; margin-top: 4px; opacity: 0.85; }

  /* Upload area */
  [data-testid="stFileUploadDropzone"] {
    border: 2px dashed #4caf50 !important;
    border-radius: 10px !important;
    background: #f1f8f1 !important;
  }

  /* Confidence bar label */
  .conf-label { font-size: 1rem; font-weight: 600; color: #333; margin-bottom: 2px; }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 48px 24px;
    border: 2px dashed #b0bec5;
    border-radius: 12px;
    color: #78909c;
    margin-top: 16px;
  }
  .empty-state h3 { color: #546e7a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model …")
def load_model_and_labels():
    """Load the trained Keras model and class labels — cached after first load."""
    import tensorflow as tf
    model_path  = ROOT / "model" / "model.keras"
    labels_path = ROOT / "model" / "labels.txt"
    if not model_path.exists() or not labels_path.exists():
        return None, None
    model  = tf.keras.models.load_model(str(model_path))
    labels = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return model, labels


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize to 224×224, normalise to [0,1], add batch dim."""
    img = pil_image.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, labels, img_array):
    """Run inference. Returns (class_name, confidence, sorted top-5)."""
    probs = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    class_name  = labels[top_idx]
    confidence  = float(probs[top_idx])
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    top5 = [(labels[i], float(p)) for i, p in top5]
    return class_name, confidence, top5


def confidence_colour(conf: float) -> str:
    if conf >= 0.75: return "#2e7d32"
    if conf >= 0.45: return "#f57c00"
    return "#c62828"


# ─── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🌿 Plant & Vegetable Health Detection</h1>
  <p>Upload a photo of a plant leaf or vegetable to get the health status,
     disease name (if any), confidence score, and a safe-to-consume verdict.</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────
model, labels = load_model_and_labels()

if model is None:
    st.error(
        "⚠️ Model not found. Please run the training pipeline first:\n"
        "```\npython src/preprocess.py\npython src/train.py\n```"
    )
    st.stop()

# ─── Layout: upload left, results right ───────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload an image")
    uploaded = st.file_uploader(
        "Choose a leaf or vegetable photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded photo", use_container_width=True)

with col_right:
    if not uploaded:
        st.markdown("""
        <div class="empty-state">
          <h3>🌱 No image yet</h3>
          <p>Upload a photo on the left to get your health report here.</p>
          <p><b>Works with:</b> plant leaves (healthy or diseased) and vegetables (fresh or rotten)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🔬 Health Report")
        try:
            with st.spinner("Analysing image …"):
                img_array = preprocess_image(pil_img)
                class_name, confidence, top5 = predict(model, labels, img_array)

            # Import verdicts
            sys.path.insert(0, str(ROOT / "src"))
            from verdicts import get_verdict, HONESTY_LINE
            verdict_info = get_verdict(class_name)

            # ── Class name ──────────────────────────────────────
            display_cls = class_name.replace("___", " — ").replace("_", " ")
            st.markdown(f"""
            <div class="card card-class">
              <div class="card-label">Identified As</div>
              <div class="card-value">{display_cls}</div>
            </div>""", unsafe_allow_html=True)

            # ── Confidence meter ────────────────────────────────
            bar_colour = confidence_colour(confidence)
            conf_pct = confidence * 100
            st.markdown(f'<div class="conf-label">Confidence: <span style="color:{bar_colour};font-size:1.2rem">{conf_pct:.1f}%</span></div>', unsafe_allow_html=True)
            st.progress(confidence, text=None)

            # ── Disease ─────────────────────────────────────────
            disease = verdict_info["disease"]
            if disease and disease.lower() not in ("none", ""):
                st.markdown(f"""
                <div class="card card-disease">
                  <div class="card-label">Disease Detected</div>
                  <div class="card-value">🦠 {disease}</div>
                </div>""", unsafe_allow_html=True)

            # ── Safe-to-consume verdict ─────────────────────────
            safe   = verdict_info["safe"]
            vtxt   = verdict_info["verdict"]
            note   = verdict_info["note"]
            card_c = "card-safe" if safe else "card-unsafe"
            icon   = "✅" if safe else "🚫"
            st.markdown(f"""
            <div class="card {card_c}">
              <div class="card-label">Safe to Consume?</div>
              <div class="card-value">{icon} {vtxt}</div>
              <div class="card-note">{note}</div>
            </div>""", unsafe_allow_html=True)

            # ── Honesty line ────────────────────────────────────
            st.markdown(f"""
            <div class="card card-honesty">
              <div class="card-label">⚠️ Important Note</div>
              {HONESTY_LINE}
            </div>""", unsafe_allow_html=True)

            # ── Top-5 predictions expander ──────────────────────
            with st.expander("📊 View top-5 predictions"):
                for i, (lbl, prob) in enumerate(top5, 1):
                    nice = lbl.replace("___", " — ").replace("_", " ")
                    st.markdown(f"**{i}. {nice}**")
                    st.progress(float(prob), text=f"{prob*100:.1f}%")

        except Exception as e:
            st.error(f"⚠️ Something went wrong during analysis: {e}")
            st.info("Please try a different image, or re-run the training pipeline.")

# ─── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "Plant & Vegetable Health Detection · MobileNetV2 + PlantVillage + Fresh/Rotten dataset · "
    "Review 1 build — model is an early version; accuracy will be improved post-review."
)
