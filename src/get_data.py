# src/get_data.py — Phase 1: Download datasets to data/
# Downloads PlantVillage and Fresh/Rotten fruit from HuggingFace (no login needed).
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

print("=" * 60)
print("Phase 1 — Downloading datasets")
print("=" * 60)

# ── 1. PlantVillage ───────────────────────────────────────────
print("\n[1/2] PlantVillage (mohanty/PlantVillage) ...")
try:
    import os as _os
    _os.environ["HF_DATASETS_CACHE"] = str(ROOT / "hf_cache")
    from datasets import load_dataset
    pv = load_dataset("mohanty/PlantVillage", "color", trust_remote_code=True)
    label_names = pv["train"].features["label"].names
    total_pv = 0
    for split in ("train", "test"):
        for row in pv[split]:
            cls = label_names[row["label"]]
            folder = cls.replace(",", "").replace(" ", "_").replace("(","").replace(")","")
            dest_dir = DATA_DIR / folder
            dest_dir.mkdir(exist_ok=True)
            n = len(list(dest_dir.glob("*.jpg")))
            row["image"].save(dest_dir / f"{n}.jpg")
            total_pv += 1
    print(f"  Saved {total_pv} images across {len(label_names)} PlantVillage classes.")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# ── 2. Fresh/Rotten fruit ─────────────────────────────────────
print("\n[2/2] Fresh/Rotten fruit (Project-AgML/fresh_rotten_fruit_classification) ...")
try:
    veg = load_dataset(
        "Project-AgML/fresh_rotten_fruit_classification",
        "augmented",
        trust_remote_code=True,
    )
    lab = veg["train"].features["label"].names  # ["fresh","rotten"]
    total_veg = 0
    for row in veg["train"]:
        freshness = lab[row["label"]]
        fruit = row["fruit_type"].lower().replace(" ", "_")
        cls_name = f"{freshness}_{fruit}"
        dest_dir = DATA_DIR / cls_name
        dest_dir.mkdir(exist_ok=True)
        n = len(list(dest_dir.glob("*.jpg")))
        row["image"].save(dest_dir / f"{n}.jpg")
        total_veg += 1
    print(f"  Saved {total_veg} images.")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# ── 3. Report ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Class list with image counts:")
print("=" * 60)
classes = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
total = 0
for c in classes:
    n = len(list(c.glob("*.jpg")))
    if n > 0:
        print(f"  {c.name:<52s} {n:>5d}")
        total += n
print(f"\n  TOTAL: {total} images across {len(classes)} classes")
