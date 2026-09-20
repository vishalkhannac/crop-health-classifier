# todo.md — Collaborator Tasks (Antigravity Navigator)

**Antigravity Agent:** Read this document carefully. You are collaborating on the **Plant & Vegetable Health Detection** repository (`crop-health-classifier`).

---

## 📌 Repository Overview & Architecture
- **Language / Framework:** Python 3.8+, TensorFlow 2.13.0, Streamlit.
- **Model:** MobileNetV2 transfer learning with 76 fine-tuned classes (38 leaf disease/health classes + 38 fresh/rotten vegetable & fruit classes).
- **Core Files:**
  - `app.py`: Streamlit frontend application.
  - `src/model.py`: Model architecture definition.
  - `src/verdicts.py`: Deterministic verdict and disease mapping engine.
  - `src/evaluate.py`: Evaluation metrics and confusion matrix generator.
  - `model/model.keras` & `model/labels.txt`: Current trained weights & label index.

---

## 🎯 Collaborator Milestone Checklist

Pick and implement the following tasks in order. **Do not modify the core model weights or retrain the full network unless explicitly required.**

---

### [ ] Task 1: Grad-CAM Visual Explainability Heatmap
**Objective:** Provide visual explainability showing *where* the model detected rot, lesions, or disease symptoms.

- [ ] Create `src/explainability.py`:
  - Implement a `generate_gradcam(img_array, model, last_conv_layer_name="Conv_1")` function using Keras gradient computation.
  - Generate an overlaid heatmap on the original input image.
- [ ] Update `app.py`:
  - Add an interactive toggle: *"Show AI Attention Heatmap (Grad-CAM)"*.
  - Render the superimposed heatmap side-by-side with the uploaded photo.
- [ ] Test with `test_user_tomato.jpg` and `test_user_carrot.jpg` to verify heatmaps highlight the lesion and produce body accurately.

---

### [ ] Task 2: Actionable Treatment & Storage Advisory Engine
**Objective:** Give farmers and consumers practical steps when disease or decay is detected.

- [ ] Update `src/verdicts.py`:
  - Expand the return tuple or dictionary to include `treatment` (e.g., recommended organic fungicide, copper spray schedule, pruning advice) and `prevention` (e.g., crop rotation, optimal storage temperature/humidity).
- [ ] Update `app.py`:
  - Display a styled **"Recommended Actions and Prevention"** accordion card below the Safe-to-Consume verdict.

---

### [ ] Task 3: Mobile Camera Input & Multi-Image Batch Scan
**Objective:** Improve real-world field usability for mobile devices and bulk inspections.

- [ ] Update `app.py`:
  - Add `st.radio` or tabs for input mode: **"Upload Photo"** vs. **"Take Live Photo (Camera)"** using `st.camera_input()`.
  - Allow uploading multiple files simultaneously; display an aggregated summary table of batch health diagnoses.

---

### [ ] Task 4: Out-of-Distribution (OOD) Object Guardrails
**Objective:** Prevent confident false predictions when non-plant/non-produce photos are uploaded (e.g., cars, household objects, humans).

- [ ] Add a confidence and entropy threshold checker in `src/verdicts.py`:
  - If top-1 confidence is below 35% or prediction entropy is high, return an *"Unrecognized Object / Low Confidence"* status with advice to upload a clearer photo of a plant leaf or vegetable.

---

### [ ] Task 5: Exportable Diagnostic PDF Report
**Objective:** Allow users to download and share official diagnosis reports.

- [ ] Create `src/report_generator.py`:
  - Implement PDF generation (using `fpdf2` or `reportlab`) containing timestamp, uploaded image thumbnail, predicted disease, confidence %, consumption verdict, and treatment advice.
- [ ] Add a **"📥 Download PDF Health Report"** button in `app.py`.

---

## 🛠️ Operating Rules for Collaborator Agent
1. Work on one task at a time and verify locally before moving to the next.
2. Run the Streamlit app locally via `python run_app.py` and test end-to-end.
3. Keep `HOW_TO_RUN.md` updated if new dependencies or commands are introduced.
4. Commit with descriptive git commit messages (e.g., `git commit -m "feat: add Grad-CAM explainability in Streamlit"`).
5. Push your branch / commits to `origin main` when verified.