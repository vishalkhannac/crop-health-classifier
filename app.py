# app.py — Plant & Vegetable Health Detection with Multi-Object Support
import sys
from pathlib import Path

# Resolve venv site-packages and src paths
ROOT = Path(__file__).resolve().parent

candidate_site_packages = [
    ROOT / "venv" / "Lib" / "site-packages",
    Path(r"D:\plant-veg-health\venv\Lib\site-packages"),
    Path(r"C:\Users\kurtz\Downloads\MLMProjectFeed\venv\Lib\site-packages"),
]

for p in candidate_site_packages:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if Path(r"D:\plant-veg-health\src").exists() and str(Path(r"D:\plant-veg-health\src")) not in sys.path:
    sys.path.insert(0, str(Path(r"D:\plant-veg-health\src")))

import streamlit as st
import numpy as np
from PIL import Image
import io

import detector
from verdicts import get_verdict, HONESTY_LINE

# Page config
st.set_page_config(
    page_title="Plant & Vegetable Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown("""
<style>
  .header-bar {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }
  .header-bar h1 { color: white; margin: 0 0 6px 0; font-size: 2.1rem; }
  .header-bar p  { color: rgba(255,255,255,0.9); margin: 0; font-size: 1.05rem; }

  .card {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1.02rem;
  }
  .card-safe   { background: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20; }
  .card-unsafe { background: #ffebee; border-left: 6px solid #c62828; color: #b71c1c; }
  .card-mixed  { background: #fff8e1; border-left: 6px solid #f57f17; color: #e65100; }
  .card-class  { background: #e3f2fd; border-left: 6px solid #1565c0; color: #0d47a1; }
  .card-disease{ background: #fce4ec; border-left: 6px solid #880e4f; color: #560027; }
  .card-honesty{ background: #f5f5f5; border-left: 6px solid #9e9e9e; color: #616161; font-size: 0.88rem; }
  
  .card-label  { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.08em;
                 text-transform: uppercase; margin-bottom: 4px; opacity: 0.75; }
  .card-value  { font-size: 1.25rem; font-weight: 700; }
  .card-note   { font-size: 0.9rem; margin-top: 4px; opacity: 0.88; }

  [data-testid="stFileUploadDropzone"] {
    border: 2px dashed #4caf50 !important;
    border-radius: 12px !important;
    background: #f1f8f1 !important;
  }

  .empty-state {
    text-align: center;
    padding: 48px 24px;
    border: 2px dashed #b0bec5;
    border-radius: 12px;
    color: #78909c;
    margin-top: 16px;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model …")
def load_model_and_labels():
    import tensorflow as tf
    candidate_model_dirs = [
        ROOT / "model",
        Path(r"D:\plant-veg-health\model"),
        Path(r"C:\Users\kurtz\Downloads\MLMProjectFeed\model"),
    ]
    model_path = None
    labels_path = None
    for d in candidate_model_dirs:
        m = d / "model.keras"
        l = d / "labels.txt"
        if m.exists() and l.exists():
            model_path = m
            labels_path = l
            break
            
    if model_path is None or labels_path is None:
        return None, None
    model  = tf.keras.models.load_model(str(model_path))
    labels = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return model, labels


st.markdown("""
<div class="header-bar">
  <h1>🌿 Plant & Produce Health Analyzer</h1>
  <p>Upload a photo of plants, vegetables, or fruits (single or multiple objects in one image) to inspect disease, rot, confidence, and consumption safety.</p>
</div>
""", unsafe_allow_html=True)

model, labels = load_model_and_labels()

if model is None:
    st.error("⚠️ Model not found. Please run preprocessing and training first.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Upload photo of plant leaves, fruits or vegetables",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        
        with st.spinner("Analyzing objects and disease status..."):
            analysis = detector.analyze_image_multiobject(pil_img, model, labels, get_verdict)
            
        annotated_img = analysis["annotated_image"]
        items = analysis["items"]
        summary = analysis["summary"]
        
        caption_text = f"Annotated Image — {len(items)} object(s) localized" if summary["is_multi"] else "Uploaded Image"
        st.image(annotated_img, caption=caption_text, use_container_width=True)

with col_right:
    if not uploaded:
        st.markdown("""
        <div class="empty-state">
          <h3>🌱 No image uploaded</h3>
          <p>Upload a photo on the left to analyze health, rot, and consumption safety.</p>
          <p><b>Supports:</b> Single items or multiple items (e.g. apples, tomatoes, carrots, leaves) in a single photo.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🔬 Health Report")
        
        if summary["is_multi"] and len(items) > 1:
            # Multi-Object Batch Mode
            card_class = "card-safe" if summary["fresh_count"] == len(items) else ("card-unsafe" if summary["rotten_count"] == len(items) else "card-mixed")
            icon = "✅" if summary["fresh_count"] == len(items) else ("🚫" if summary["rotten_count"] == len(items) else "⚠️")
            
            st.markdown(f"""
            <div class="card {card_class}">
              <div class="card-label">Batch Overview ({len(items)} Objects Detected)</div>
              <div class="card-value">{icon} {summary['fresh_count']} Fresh &nbsp;|&nbsp; {summary['rotten_count']} Rotten / Diseased</div>
              <div class="card-note">{summary['batch_verdict']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 Individual Item Breakdown")
            for item in items:
                idx = item["index"]
                v_info = item["verdict_info"]
                safe = v_info["safe"]
                status = v_info["status"]
                disease = v_info["disease"]
                conf = item["confidence"]
                badge_col = "#2e7d32" if safe else "#c62828"
                badge_text = "SAFE" if safe else "NOT SAFE"
                
                with st.expander(f"Item #{idx}: {status} ({conf:.1f}%) — {badge_text}", expanded=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(item["crop"], use_container_width=True)
                    with c2:
                        st.markdown(f"**Classification:** {status}")
                        st.markdown(f"**Confidence:** `{conf:.1f}%`")
                        if disease and disease.lower() not in ("none", ""):
                            st.markdown(f"**Disease/Condition:** <span style='color:#880e4f;font-weight:600;'>{disease}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Safety Verdict:** <span style='color:{badge_col};font-weight:700;'>{v_info['verdict']}</span>", unsafe_allow_html=True)
                        st.caption(v_info["note"])
        else:
            # Single Object Mode
            item = items[0]
            v_info = item["verdict_info"]
            class_name = item["class_name"]
            confidence = item["confidence"]
            top3 = item["top3"]
            
            display_cls = v_info["status"]
            st.markdown(f"""
            <div class="card card-class">
              <div class="card-label">Identified As</div>
              <div class="card-value">{display_cls}</div>
            </div>
            """, unsafe_allow_html=True)
            
            bar_color = "#2e7d32" if confidence >= 75 else ("#f57c00" if confidence >= 45 else "#c62828")
            st.markdown(f"<div style='font-size:1rem;font-weight:600;margin-bottom:2px;'>Confidence: <span style='color:{bar_color};font-size:1.2rem;'>{confidence:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(min(1.0, confidence / 100.0))
            
            disease = v_info["disease"]
            if disease and disease.lower() not in ("none", ""):
                st.markdown(f"""
                <div class="card card-disease">
                  <div class="card-label">Disease / Condition Detected</div>
                  <div class="card-value">🦠 {disease}</div>
                </div>
                """, unsafe_allow_html=True)
                
            safe = v_info["safe"]
            vtxt = v_info["verdict"]
            note = v_info["note"]
            card_c = "card-safe" if safe else "card-unsafe"
            icon = "✅" if safe else "🚫"
            st.markdown(f"""
            <div class="card {card_c}">
              <div class="card-label">Safe to Consume?</div>
              <div class="card-value">{icon} {vtxt}</div>
              <div class="card-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📊 View top predictions"):
                for t in top3:
                    st.markdown(f"**{t['status']}** (`{t['confidence']:.1f}%`)")
                    st.progress(min(1.0, t['confidence'] / 100.0))

        st.markdown(f"""
        <div class="card card-honesty">
          <div class="card-label">⚠️ Food Safety Notice</div>
          {HONESTY_LINE}
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("Plant & Produce Health Analyzer · Multi-Object Localization & Health Assessment Engine")

