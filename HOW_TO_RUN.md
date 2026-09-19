# How to Run — Plant & Vegetable Health Detection

Upload a photo of a plant leaf or vegetable. The app reports the **status / class**, a **confidence %**, the **disease** (if any), and whether it is **safe to consume** — in plain words.

> `Safe to consume` is a **visual estimate from appearance only, not a real food-safety test.**

---

## Project location
All project files live in `D:\plant-veg-health\`.

## Python environment
This project uses the Python 3.8 embedded in Spyder (`C:\Users\kurtz\Downloads\python\Python\python.exe`).
All project packages are installed in `D:\plant-veg-health\venv\Lib\site-packages`.

**To run any script, use:**
```
C:\Users\kurtz\Downloads\python\Python\python.exe <script>
```

---

## 1. Prerequisites
- Python 3.8 (Spyder-bundled) with packages installed in `venv\Lib\site-packages`
- If running fresh: see the install section below

## 2. Install packages (once — already done if venv/Lib/site-packages is populated)
```powershell
 = "C:\Users\kurtz\Downloads\python\Python\python.exe"
 = "D:\plant-veg-health\venv\Lib\site-packages"
C:\Users\kurtz\AppData\Local\Temp = "D:\tmp"
C:\Users\kurtz\AppData\Local\Temp  = "D:\tmp"
&  -m pip install tensorflow==2.13.0 opencv-python-headless scikit-learn streamlit datasets huggingface_hub --target  --cache-dir "D:\pip-cache"
```

## 3. Get the data (only needed for training)
```powershell
 = "C:\Users\kurtz\Downloads\python\Python\python.exe"
# Downloads PlantVillage + Fresh/Rotten datasets from HuggingFace (no login needed)
&  D:\plant-veg-health\src\get_data.py
```
Result: `data/` has one subfolder per class (~38 plant classes + fresh/rotten fruit/veg classes).

## 4. Run the pipeline (in this order)
```powershell
 = "C:\Users\kurtz\Downloads\python\Python\python.exe"
&  D:\plant-veg-health\src\preprocess.py   # builds train/val/test splits, saves model/labels.txt
&  D:\plant-veg-health\src\train.py         # trains, saves model/model.keras + curves
&  D:\plant-veg-health\src\evaluate.py      # test accuracy + confusion matrix
```

## 5. Run the app ← the main thing
```powershell
# In PowerShell:
& "C:\Users\kurtz\Downloads\python\Python\python.exe" "D:\plant-veg-health\run_app.py"
```
Open **http://localhost:8501**, upload a photo, and read the result. Stop with **Ctrl+C**.

---

## Troubleshooting
- **ModuleNotFoundError** → make sure `sys.path` includes `venv\Lib\site-packages`; each script adds it automatically.
- **App can't find model or labels** → run `preprocess.py` then `train.py` first.
- **Training is slow** → normal without GPU; 10 epochs on CPU takes ~1–3 hours depending on dataset size.
- **Disk space** → all data, cache, and model files are on D: (which has ~150 GB free).

## Current status
**Review 1 build in progress.** All source files created; packages installed. Running data → train → evaluate pipeline.
