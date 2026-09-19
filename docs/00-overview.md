# 00 — Project Overview (context for every phase)

**What we are building:** an image-classification model plus a web app that takes a photo of a plant leaf or a vegetable and reports:
1. **What it is / its status** — the class name (e.g. "Tomato — Late Blight", "Fresh Potato", "Rotten Tomato", "Healthy Apple leaf").
2. **A confidence meter** — how sure the model is, as a percentage / bar.
3. **A disease name** where the class is a plant disease.
4. **A "safe to consume" verdict in words** — e.g. "Likely safe to consume" or "Not safe — discard", with a short note.

**Important honesty note (keep this wording in the app and report):** the model judges only by **appearance**. "Safe to consume" means *looks safe based on visible disease or rot* — it is a visual estimate, **not** a real food-safety test. Always show this as guidance, not a guarantee.

**Approach:** transfer learning with **MobileNetV2** (a lightweight CNN), trained on public datasets, served through a **Streamlit** app.

## This is a MULTI-CLASS project (not just healthy/unhealthy)
Keep the detailed class names from the datasets instead of merging into two folders. The plant dataset already has disease names; the vegetable dataset has fresh/rotten-per-item names. The model predicts one class out of all of them.

## Fixed technical choices — use these unless a phase file says otherwise
- Image size: **224 x 224**, pixels normalised to **0–1**
- Base model: **MobileNetV2** pre-trained on ImageNet, base **frozen**
- Head: global average pooling → one small dense layer → **softmax output with one node per class**
- Loss: **categorical cross-entropy** (multi-class) | Optimiser: **Adam** | Epochs: **~10**
- Split: **70% train / 15% validation / 15% test**
- Stack: **Python, TensorFlow/Keras, OpenCV, Matplotlib, scikit-learn, Streamlit**

## The "safe to consume" logic is a rule, not learned
The model only predicts the class. A small lookup table (`src/verdicts.py`) maps **each class → {disease name (or "none"), safe/unsafe verdict, short note}**. The app looks up the predicted class in this table to show the words. Healthy leaves and fresh vegetables → "Likely safe"; diseased leaves and rotten vegetables → "Not safe — discard".

## Data sources (see phase 1)
- **Plant leaves → PlantVillage from GitHub (no login needed).**
- **Vegetables → Fresh/Rotten dataset from Kaggle** (needs a free Kaggle API token, or download manually).

## Target folder structure
```
plant-veg-health/
├── todo.md                 # navigator (already here)
├── HOW_TO_RUN.md           # keep this updated every phase
├── docs/                   # these phase files
├── data/                   # one subfolder per class (e.g. Tomato___Late_blight, Fresh_Potato, ...)
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── verdicts.py         # class -> disease + safe/unsafe + note
├── model/                  # saved trained model + class-label list
├── app.py                  # Streamlit app
├── requirements.txt
└── README.md
```

## Keep it simple
Prefer short, readable scripts. Add a `requirements.txt` entry whenever you introduce a new library. Commit after every phase.
