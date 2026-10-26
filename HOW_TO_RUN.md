# How to Run — Plant & Vegetable Health Detection (Final Presentation Guide)

Upload one or more photos, take a live camera snapshot, or select a built-in demo specimen. The application reports:
- **Identified Specimen & Status** (across 70+ plant leaf disease and fresh/rotten produce classes)
- **Calibrated Confidence Meter (%)** + Top-3 breakdown
- **Multi-Object Bounding Box Localization & Batch Verdict**
- **Grad-CAM AI Attention Heatmap** showing where disease or rot is detected
- **Disease Diagnosis + Actionable Treatment, Prevention & Storage Advisory**
- **Safe-to-Consume Verdict** + **1-Click Downloadable PDF Health Report**

> `Safe to consume` is a **visual estimate from exterior appearance only — not a laboratory microbiological food-safety test.**

---

## 1. Fastest Way to Launch (100% Offline)

### Option A: Double-Click `start.bat`
Simply double-click **`start.bat`** inside `D:\plant-veg-health\` or `C:\Users\kurtz\Downloads\MLMProjectFeed\`.

### Option B: Run via PowerShell
```powershell
& "C:\Users\kurtz\Downloads\python\Python\python.exe" "D:\plant-veg-health\run_app.py"
```
Then open [**http://localhost:8501**](http://localhost:8501) in your browser.

---

## 2. Presentation Demo Walkthrough (For Tomorrow's Review)

Once the app is open at `http://localhost:8501`:
1. **Tab 1 — `🔍 Live Diagnosis & Batch Inspector`**:
   - Select **`🧪 Quick Demo Samples`** to immediately demonstrate:
     - **Multi-Object Detection (`test_multi_apples.jpg`)**: Shows 3 apples localized with green/red bounding boxes, individual crop breakdowns, and a Mixed Batch warning.
     - **Sliced Rotten Apple (`test_user_apple.jpg`)**: Shows `Rotten Apple` at **99.7% confidence** alongside the **Grad-CAM Heatmap** highlighting the internal brown decay.
     - **Rotten Tomato / Fresh Carrot / Rotten Capsicum / Fresh Pear**: Instant 99%+ confidence diagnoses with custom storage & treatment advice.
   - Click **`📥 Download Official PDF Diagnostic Report`** to showcase the automated PDF export.
   - Switch to **`📤 Upload Photo(s)`** (supports selecting multiple images at once for a batch summary table) or **`📷 Live Camera Capture`**.
2. **Tab 2 — `📊 Model Architecture & Evaluation Metrics`**:
   - Displays the MobileNetV2 architecture summary, **89.51% validation accuracy**, **Training & Validation Curves**, **Confusion Matrix**, and the searchable directory of all supported classes.

---

## 3. Full Data & Training Pipeline Commands (Reference)

```powershell
$py = "C:\Users\kurtz\Downloads\python\Python\python.exe"

# Step 1: Preprocess data & generate stratified 70/15/15 splits
& $py D:\plant-veg-health\src\preprocess.py

# Step 2: Train 2-stage MobileNetV2 transfer learning model
& $py D:\plant-veg-health\src\train.py

# Step 3: Evaluate on test split & generate confusion matrix
& $py D:\plant-veg-health\src\evaluate.py
```
