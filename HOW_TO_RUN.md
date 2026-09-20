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
$py = "C:\Users\kurtz\Downloads\python\Python\python.exe"
$env:PYTHONPATH = "D:\plant-veg-health\venv\Lib\site-packages"
$env:TEMP = "D:\tmp"
$env:TMP = "D:\tmp"
$env:HF_HOME = "D:\hf_home"
& $py -m pip install tensorflow==2.13.0 opencv-python-headless scikit-learn streamlit datasets huggingface_hub --target "D:\plant-veg-health\venv\Lib\site-packages" --cache-dir "D:\pip-cache"
```

## 3. Get the data (only needed for retraining)
```powershell
$py = "C:\Users\kurtz\Downloads\python\Python\python.exe"
# Downloads PlantVillage + Fresh/Rotten datasets from HuggingFace (no login needed)
& $py D:\plant-veg-health\src\get_data.py
```
Result: `data/` has one subfolder per class (102 classes across 38 plant leaf diseases and 64 fresh/rotten vegetable & fruit produce classes).

## 4. Run the pipeline (in this order)
```powershell
$py = "C:\Users\kurtz\Downloads\python\Python\python.exe"
& $py D:\plant-veg-health\src\preprocess.py   # builds train/val/test splits, saves model/labels.txt
& $py D:\plant-veg-health\src\train.py         # trains, saves model/model.keras + curves
& $py D:\plant-veg-health\src\evaluate.py      # test accuracy + confusion matrix
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
- **Training is slow** → normal without GPU; fine-tuning on CPU takes ~1–2 hours depending on dataset size.
- **Disk space** → all data, cache, and model files are on D: (which has ~150 GB free).

## Current status
**Review 1 scope complete.** Phases 1–7 fully built, tested, and verified. 102 classes covering comprehensive plant leaf diseases and fresh/decayed vegetables/fruits. Polished Streamlit UI live at `http://localhost:8501`.
