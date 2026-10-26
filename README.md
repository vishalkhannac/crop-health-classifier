# 🌿 Plant & Vegetable Health Analyzer

An end-to-end deep learning & computer vision web application for **automated plant leaf disease diagnosis**, **fruit/vegetable freshness & rot inspection**, **multi-object batch localization**, **Grad-CAM visual explainability**, and **exportable PDF diagnostic reports**.

---

## ✨ Key Features & Capabilities

1. **70-Class Deep Learning Classifier (MobileNetV2 Transfer Learning)**
   - **38 PlantVillage Leaf Disease & Health Classes:** Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
   - **32 Fresh & Rotten Produce Classes:** Fresh and Rotten varieties of Apple, Banana, Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Chilli Pepper, Cucumber, Eggplant, Garlic, Ginger, Grape, Guava, Jujube, Kiwi, Lemon, Lettuce, Mango, Okra, Onion, Orange, Pear, Peas, Pineapple, Pomegranate, Potato, Pumpkin, Radish, Spinach, Strawberry, Sweetpotato, Tomato, Turnip, and Watermelon.

2. **Multi-Object Localization & Batch Health Diagnosis (`src/detector.py`)**
   - Uses HSV/LAB color-space segmentation, adaptive morphological filtering, Canny contour proposals, and Non-Maximum Suppression (NMS, IoU = 0.35) to automatically isolate multiple fruits, vegetables, or leaves in a single photo.
   - Draws **color-coded bounding boxes** (Green = Safe/Fresh, Red = Rotten/Diseased) and computes an overall **Batch Safety Verdict**.

3. **Grad-CAM Visual Attention Heatmaps (`src/explainability.py`)**
   - Computes Gradient-weighted Class Activation Maps from the final convolutional layer of MobileNetV2 (`Conv_1`) to highlight the exact lesions, mold spots, or tissue areas driving the AI's decision.

4. **Actionable Treatment, Prevention & Storage Advisory (`src/verdicts.py`)**
   - Provides pathogen-specific organic/chemical treatment protocols for diseased crops and temperature/humidity/ethylene storage guidelines for fresh produce.
   - Includes **Out-of-Distribution (OOD) & Entropy Guardrails** to flag blurry or non-produce images.

5. **1-Click Official PDF Health Report (`src/report_generator.py`)**
   - Generates a downloadable A4 PDF report containing the annotated specimen image, Grad-CAM heatmap, confidence metrics, consumption safety verdict, and treatment/storage recommendations.

6. **100% Offline & Standalone Ready**
   - Runs completely offline via `start.bat` or `start.ps1` with zero external cloud API dependencies.

> ⚠️ **Food Safety Disclaimer:** *The safe-to-consume verdict is a visual estimate based strictly on exterior appearance and visible signs of rot or blight. It is not a laboratory microbiological food-safety test.*

---

## 🚀 Quick Start (Offline Execution)

### Option 1: One-Click Windows Launcher (Recommended)
Double-click **`start.bat`** in the project folder. It automatically configures the Python environment and opens the web app at **`http://localhost:8501`**.

### Option 2: PowerShell / Terminal
```powershell
& "C:\Users\kurtz\Downloads\python\Python\python.exe" "D:\plant-veg-health\run_app.py"
```

---

## 📂 Project Architecture

```text
plant-veg-health/
├── app.py                      # Main Streamlit web application (Diagnosis + Metrics tabs)
├── run_app.py                  # Standalone Streamlit launcher with fallback path resolution
├── start.bat / start.ps1       # One-click offline Windows launchers
├── requirements.txt            # Python package dependencies
├── HOW_TO_RUN.md               # Step-by-step setup, execution & presentation guide
├── todo.md                     # Project milestone tracker (Phases 1–8)
├── COLLABORATOR_TODO.md        # Advanced feature checklist (All 5 tasks completed)
├── src/
│   ├── get_data.py             # Dataset downloader & organizer
│   ├── preprocess.py           # Priority-weighted sampling, augmentation & 70/15/15 splits
│   ├── model.py                # 2-stage MobileNetV2 transfer learning architecture
│   ├── train.py                # Warm-up + fine-tuning training pipeline
│   ├── evaluate.py             # Test set evaluation & confusion matrix generator
│   ├── detector.py             # Multi-object region proposal, NMS & bounding box annotator
│   ├── explainability.py       # Grad-CAM convolutional heatmap generator
│   ├── report_generator.py     # Standalone A4 PDF diagnostic report builder
│   └── verdicts.py             # Deterministic rule engine, advisory & OOD guardrails
└── model/
    ├── model.keras             # Trained MobileNetV2 weights
    ├── labels.txt              # Class label index
    ├── training_curves.png     # Accuracy & loss curves across training stages
    └── confusion_matrix.png    # Multi-class evaluation confusion matrix
```

---

## 📊 Model Performance Benchmarks

| Benchmark Specimen | Scene Type | Objects Localized | Top-1 Prediction | Confidence | Safety Verdict |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **`test_multi_apples.jpg`** | Multi-Object Cluster | 3 | 2× `fresh_apple`, 1× `rotten_apple` | 85.1% – 98.4% | ⚠️ Mixed Batch (Discard 1) |
| **`test_user_apple.jpg`** | Sliced Rotten Apple | 1 | `rotten_apple` | **99.7%** | 🚫 Not Safe — Discard |
| **`test_user_tomato.jpg`** | Decayed Tomato | 1 | `rotten_tomato` | **100.0%** | 🚫 Not Safe — Discard |
| **`test_user_carrot.jpg`** | Fresh Carrot | 1 | `fresh_carrot` | **99.7%** | ✅ Likely Safe |
| **`test_user_capsicum.jpg`** | Decayed Bell Pepper | 1 | `rotten_capsicum` | **99.9%** | 🚫 Not Safe — Discard |
| **`test_user_pear.jpg`** | Fresh Pear | 1 | `fresh_pear` | **99.5%** | ✅ Likely Safe |
