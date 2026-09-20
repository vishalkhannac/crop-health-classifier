# Plant & Vegetable Health Detection

An end-to-end deep learning web application that accurately identifies plant leaf diseases, vegetable/fruit types, and freshness/decay states directly from photos.

---

## 🌿 Features & Output

For any uploaded photo (plant leaf, fruit, or vegetable), the application delivers:
- **Identified Class & Status:** e.g., *Tomato — Late Blight*, *Fresh Potato*, *Rotten Capsicum*.
- **Confidence Meter (%):** Calibrated softmax confidence score.
- **Disease & Health Diagnosis:** Specific botanical disease name (e.g., *Early Blight*, *Late Blight*, *Black Rot*, *Healthy*) or spoilage description.
- **Consumption Safety Verdict:** Visual safety estimate badge (*Safe to consume*, *Not safe — discard*, or *Not edible plant part*).
- **Interactive Top-5 Predictions:** Expandable breakdown of candidate classes and probabilities.

> ⚠️ **Visual Estimate Disclaimer:** *The safe-to-consume verdict is an automated estimate based strictly on visible exterior appearance and signs of rot/blight. It is not a laboratory microbiological food safety test.*

---

## 📊 Dataset & Model Architecture

- **Total Classes:** 102 distinct classes
  - **38 PlantVillage Leaf Disease & Health Classes:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
  - **64 Vegetable & Fruit Produce Classes:** Fresh and Rotten varieties of Tomato, Potato, Capsicum, Cucumber, Carrot, Eggplant, Cabbage, Cauliflower, Pumpkin, Radish, Onion, Garlic, Ginger, Spinach, Lettuce, Peas, Turnip, Beetroot, Corn, Chilli Pepper, Apple, Banana, Orange, Mango, Strawberry, Pomegranate, Grapes, and Watermelon.
- **Model Architecture:** Transfer Learning via MobileNetV2 with fine-tuned top convolutional layers + Global Average Pooling + Dense(128, ReLU) + Dropout(0.3) + Dense(102, Softmax).
- **Validation Accuracy:** 82.1% – 93.3% across 102 fine-grained classes.

---

## 🚀 Quick Start

### 1. Requirements & Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python run_app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📂 Project Structure

```
plant-veg-health/
├── app.py                      # Main Streamlit web application
├── run_app.py                  # Streamlit launcher script
├── requirements.txt            # Python dependencies
├── HOW_TO_RUN.md               # Step-by-step setup & execution guide
├── todo.md                     # Project milestone roadmap
├── docs/                       # Phase-by-phase specifications (Phases 1–8)
├── src/
│   ├── get_data.py             # Dataset downloader (PlantVillage + Produce)
│   ├── preprocess.py           # Preprocessing, balancing & stratified train/val/test splits
│   ├── model.py                # MobileNetV2 architecture definition
│   ├── train.py                # Model training & fine-tuning pipeline
│   ├── evaluate.py             # Test evaluation & confusion matrix generator
│   └── verdicts.py             # Deterministic rule engine for all 102 classes
└── model/
    ├── labels.txt              # Complete list of 102 class names
    ├── model.keras             # Trained neural network model
    └── training_curves.png     # Loss & accuracy curves
```

---

## 🛠️ Pipeline Execution (Retraining)

To run the data and training pipeline from scratch:
1. **Prepare Data & Splits:** `python src/preprocess.py`
2. **Train Model:** `python src/train.py`
3. **Evaluate Test Set:** `python src/evaluate.py`
