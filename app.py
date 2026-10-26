# app.py — Plant & Vegetable Health Detection (Full Presentation Build)
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
import explainability
import report_generator
from verdicts import get_verdict, HONESTY_LINE

# Page config
st.set_page_config(
    page_title="Plant & Produce Health Analyzer",
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
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
  }
  .header-bar h1 { color: white; margin: 0 0 6px 0; font-size: 2.1rem; }
  .header-bar p  { color: rgba(255,255,255,0.92); margin: 0; font-size: 1.05rem; }

  .card {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1.02rem;
  }
  .card-safe    { background: #e8f5e9; border-left: 6px solid #2e7d32; color: #1b5e20; }
  .card-unsafe  { background: #ffebee; border-left: 6px solid #c62828; color: #b71c1c; }
  .card-mixed   { background: #fff8e1; border-left: 6px solid #f57f17; color: #e65100; }
  .card-class   { background: #e3f2fd; border-left: 6px solid #1565c0; color: #0d47a1; }
  .card-disease { background: #fce4ec; border-left: 6px solid #880e4f; color: #560027; }
  .card-advice  { background: #f1f8e9; border-left: 6px solid #558b2f; color: #33691e; }
  .card-honesty { background: #f5f5f5; border-left: 6px solid #9e9e9e; color: #616161; font-size: 0.88rem; }
  
  .card-label  { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.08em;
                 text-transform: uppercase; margin-bottom: 4px; opacity: 0.78; }
  .card-value  { font-size: 1.25rem; font-weight: 700; }
  .card-note   { font-size: 0.92rem; margin-top: 4px; opacity: 0.90; }

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


@st.cache_resource(show_spinner="Loading deep learning model & class index …")
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
        return None, None, None
    model  = tf.keras.models.load_model(str(model_path))
    labels = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return model, labels, model_path.parent


def find_sample_file(filename: str):
    for d in [ROOT, Path(r"D:\plant-veg-health"), Path(r"C:\Users\kurtz\Downloads\MLMProjectFeed")]:
        p = d / filename
        if p.exists():
            return p
    return None


st.markdown("""
<div class="header-bar">
  <h1>🌿 Plant & Produce Health Analyzer</h1>
  <p>AI-Powered Multi-Object Localization, Disease Diagnosis, Grad-CAM Explainability & Consumption Safety Engine</p>
</div>
""", unsafe_allow_html=True)

model, labels, model_dir = load_model_and_labels()

if model is None:
    st.error("⚠️ Model not found. Please run preprocessing and training first.")
    st.stop()

tab_diagnose, tab_metrics = st.tabs([
    "🔍 Live Diagnosis & Batch Inspector",
    "📊 Model Architecture & Evaluation Metrics"
])

with tab_diagnose:
    col_left, col_right = st.columns([1, 1.08], gap="large")

    with col_left:
        st.markdown("### 📸 Image Input")
        input_mode = st.radio(
            "Select Input Method:",
            ["📤 Upload Photo(s)", "📷 Live Camera Capture", "🧪 Quick Demo Samples"],
            horizontal=True,
            label_visibility="collapsed"
        )

        active_images = []  # list of (name, PIL.Image)

        if input_mode == "📤 Upload Photo(s)":
            uploaded_files = st.file_uploader(
                "Upload one or more photos of plant leaves, fruits, or vegetables",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded_files:
                for uf in uploaded_files:
                    try:
                        active_images.append((uf.name, Image.open(uf).convert("RGB")))
                    except Exception as e:
                        st.warning(f"Could not read {uf.name}: {e}")

        elif input_mode == "📷 Live Camera Capture":
            cam_photo = st.camera_input("Take a live photo of a leaf, fruit, or vegetable")
            if cam_photo:
                active_images.append(("Camera_Capture.jpg", Image.open(cam_photo).convert("RGB")))

        elif input_mode == "🧪 Quick Demo Samples":
            demo_options = {
                "🍎 Multi-Object Batch: 3 Apples (2 Fresh, 1 Rotten)": "test_multi_apples.jpg",
                "🍏 Sliced Rotten Apple (Internal Decay)": "test_user_apple.jpg",
                "🍅 Rotten Tomato (Fungal Blight & Spoilage)": "test_user_tomato.jpg",
                "🥕 Fresh Carrot (Healthy Root Produce)": "test_user_carrot.jpg",
                "🫑 Rotten Capsicum (Bacterial Soft Rot)": "test_user_capsicum.jpg",
                "🍃 Diseased Leaf: Tomato Late Blight": "test_diseased_leaf.jpg",
                "🌿 Healthy Leaf: Apple Foliage": "test_healthy_leaf.jpg",
            }
            selected_demo = st.selectbox("Choose a pre-loaded test specimen:", list(demo_options.keys()))
            sample_path = find_sample_file(demo_options[selected_demo])
            if sample_path:
                active_images.append((sample_path.name, Image.open(sample_path).convert("RGB")))
            else:
                st.warning("Sample file not found on disk.")

        show_gradcam = st.checkbox("🔥 Show AI Attention Heatmap (Grad-CAM Explainability)", value=True)

        # If multiple files uploaded, let user select which one to inspect in detail
        selected_idx = 0
        if len(active_images) > 1:
            st.info(f"📁 **{len(active_images)} files uploaded.** Showing multi-file batch table on the right. Select a file below for detailed visual inspection:")
            selected_idx = st.selectbox(
                "Inspect file:",
                range(len(active_images)),
                format_func=lambda i: f"File #{i+1}: {active_images[i][0]}"
            )

        if active_images:
            cur_name, pil_img = active_images[selected_idx]
            with st.spinner("Running localization, classification, and Grad-CAM analysis..."):
                analysis = detector.analyze_image_multiobject(pil_img, model, labels, get_verdict)
                gradcam_img = explainability.generate_gradcam(pil_img, model) if show_gradcam else None

            annotated_img = analysis["annotated_image"]
            items = analysis["items"]
            summary = analysis["summary"]

            caption_text = (
                f"Annotated Detection — {len(items)} object(s) localized ({cur_name})"
                if summary["is_multi"] else f"Analyzed Specimen ({cur_name})"
            )

            if show_gradcam and gradcam_img is not None:
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.image(annotated_img, caption=caption_text, use_container_width=True)
                with img_c2:
                    st.image(gradcam_img, caption="Grad-CAM Attention Heatmap (Red = High Activation)", use_container_width=True)
            else:
                st.image(annotated_img, caption=caption_text, use_container_width=True)

            # PDF Report Export Button
            try:
                pdf_bytes = report_generator.generate_pdf_report(pil_img, analysis, gradcam_img)
                st.download_button(
                    label="📥 Download Official PDF Diagnostic Report",
                    data=pdf_bytes,
                    file_name=f"Health_Report_{Path(cur_name).stem}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")

    with col_right:
        if not active_images:
            st.markdown("""
            <div class="empty-state">
              <h3>🌱 Ready for Inspection</h3>
              <p>Upload one or more photos, capture from camera, or pick a <b>Quick Demo Sample</b> on the left.</p>
              <p><b>Capabilities:</b></p>
              <p>• 70 Plant Leaf Disease & Fresh/Rotten Produce Classes<br>
                 • Multi-Object Bounding Box Localization & Batch Verdicts<br>
                 • Grad-CAM Visual Attention Heatmaps<br>
                 • Treatment, Storage & Prevention Advisory + PDF Report Export</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Multi-file upload summary table if > 1 file uploaded
            if len(active_images) > 1:
                st.markdown("### 📊 Multi-File Batch Summary")
                batch_rows = []
                for fname, fimg in active_images:
                    res = detector.analyze_image_multiobject(fimg, model, labels, get_verdict)
                    it0 = res["items"][0]
                    v0 = it0["verdict_info"]
                    batch_rows.append({
                        "File Name": fname,
                        "Objects": res["summary"]["total_items"],
                        "Primary Prediction": v0["status"],
                        "Confidence": f"{it0['confidence']:.1f}%",
                        "Disease / Condition": v0["disease"],
                        "Safety Verdict": "✅ Safe" if res["summary"]["batch_safe"] else "🚫 Not Safe",
                    })
                st.dataframe(batch_rows, use_container_width=True, hide_index=True)
                st.divider()

            st.markdown(f"### 🔬 Diagnostic Report — `{cur_name}`")

            if summary["is_multi"] and len(items) > 1:
                # Multi-Object Single-Image Batch Mode
                card_class = (
                    "card-safe" if summary["fresh_count"] == len(items)
                    else ("card-unsafe" if summary["rotten_count"] == len(items) else "card-mixed")
                )
                icon = (
                    "✅" if summary["fresh_count"] == len(items)
                    else ("🚫" if summary["rotten_count"] == len(items) else "⚠️")
                )

                st.markdown(f"""
                <div class="card {card_class}">
                  <div class="card-label">Multi-Object Batch Overview ({len(items)} Objects Localized)</div>
                  <div class="card-value">{icon} {summary['fresh_count']} Fresh / Healthy &nbsp;|&nbsp; {summary['rotten_count']} Rotten / Diseased</div>
                  <div class="card-note">{summary['batch_verdict']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📋 Individual Object Breakdown")
                for item in items:
                    idx = item["index"]
                    v_info = item["verdict_info"]
                    safe = v_info["safe"]
                    status = v_info["status"]
                    disease = v_info["disease"]
                    conf = item["confidence"]
                    badge_col = "#2e7d32" if safe else "#c62828"
                    badge_text = "SAFE" if safe else "NOT SAFE"

                    with st.expander(f"Object #{idx}: {status} ({conf:.1f}%) — {badge_text}", expanded=True):
                        c1, c2 = st.columns([1, 2.2])
                        with c1:
                            st.image(item["crop"], caption=f"Crop #{idx}", use_container_width=True)
                        with c2:
                            st.markdown(f"**Classification:** {status}")
                            st.markdown(f"**Confidence:** `{conf:.1f}%`")
                            st.progress(min(1.0, conf / 100.0))
                            if disease and not disease.lower().startswith("none"):
                                st.markdown(f"**Disease / Condition:** <span style='color:#880e4f;font-weight:600;'>{disease}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Safety Verdict:** <span style='color:{badge_col};font-weight:700;'>{v_info['verdict']}</span>", unsafe_allow_html=True)
                            st.caption(v_info["note"])
                            if v_info.get("treatment"):
                                st.markdown(f"💡 **Action:** {v_info['treatment']}")
            else:
                # Single Object Mode
                item = items[0]
                v_info = item["verdict_info"]
                confidence = item["confidence"]
                top3 = item["top3"]

                if v_info.get("is_ood"):
                    st.warning("⚠️ **Out-of-Distribution / Low Confidence Alert:** The image may be blurry, obstructed, or not a supported plant/produce item.")

                display_cls = v_info["status"]
                st.markdown(f"""
                <div class="card card-class">
                  <div class="card-label">Identified Specimen & Status</div>
                  <div class="card-value">{display_cls}</div>
                </div>
                """, unsafe_allow_html=True)

                bar_color = "#2e7d32" if confidence >= 75 else ("#f57c00" if confidence >= 45 else "#c62828")
                st.markdown(
                    f"<div style='font-size:1rem;font-weight:600;margin-bottom:4px;'>"
                    f"Model Confidence: <span style='color:{bar_color};font-size:1.25rem;'>{confidence:.1f}%</span></div>",
                    unsafe_allow_html=True
                )
                st.progress(min(1.0, confidence / 100.0))

                disease = v_info["disease"]
                if disease and not disease.lower().startswith("none"):
                    st.markdown(f"""
                    <div class="card card-disease">
                      <div class="card-label">Disease / Spoilage Condition</div>
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

                # Actionable Treatment & Storage Advisory Card
                treatment = v_info.get("treatment")
                prevention = v_info.get("prevention")
                if treatment or prevention:
                    with st.expander("🌱 Recommended Actions, Treatment & Storage Advisory", expanded=True):
                        if treatment:
                            st.markdown(f"**🛠️ Immediate Action / Treatment:**\n{treatment}")
                        if prevention:
                            st.markdown(f"**🛡️ Prevention & Optimal Storage:**\n{prevention}")

                with st.expander("📊 Top-3 Candidate Predictions"):
                    for t in top3:
                        st.markdown(f"**{t['status']}** (`{t['confidence']:.1f}%`)")
                        st.progress(min(1.0, t['confidence'] / 100.0))

            st.markdown(f"""
            <div class="card card-honesty">
              <div class="card-label">⚠️ Food Safety Disclaimer</div>
              {HONESTY_LINE}
            </div>
            """, unsafe_allow_html=True)

with tab_metrics:
    st.markdown("### 🧠 Model Architecture & Performance Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Backbone Architecture", "MobileNetV2", "ImageNet Transfer")
    m_col2.metric("Supported Classes", f"{len(labels)} Classes", "38 Leaf + 32 Produce")
    m_col3.metric("Validation Accuracy", "89.51%", "2-Stage Fine-Tuned")
    m_col4.metric("Input Resolution", "224 × 224 px", "RGB Normalized [0,1]")

    st.divider()
    c_curve, c_cm = st.columns(2)
    with c_curve:
        st.markdown("#### 📈 Training & Validation Curves")
        curves_path = model_dir / "training_curves.png" if model_dir else None
        if curves_path and curves_path.exists():
            st.image(str(curves_path), caption="2-Stage Transfer Learning & Fine-Tuning Curves", use_container_width=True)
        else:
            st.info("Training curves plot not found.")

    with c_cm:
        st.markdown("#### 🎯 Multi-Class Confusion Matrix")
        cm_path = model_dir / "confusion_matrix.png" if model_dir else None
        if cm_path and cm_path.exists():
            st.image(str(cm_path), caption="Test Split Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with st.expander(f"📋 View All {len(labels)} Supported Plant & Produce Classes"):
        cols = st.columns(3)
        for idx, lbl in enumerate(labels):
            v = get_verdict(lbl)
            icon = "🟢" if v["safe"] else "🔴"
            cols[idx % 3].markdown(f"{icon} **{v['status']}** (`{lbl}`)")

st.divider()
st.caption("Plant & Produce Health Analyzer · MobileNetV2 Transfer Learning · Multi-Object Detection & Grad-CAM Explainability")
