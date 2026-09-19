# src/get_data.py — Phase 1: Organise data into data/ folder
# Source 1: PlantVillage git clone at hf_cache/plantvillage_repo/raw/color/
# Source 2: Fresh/Rotten fruit from HuggingFace datasets library
#
# Run this AFTER the git clone completes.

import os, sys, shutil
from pathlib import Path

ROOT      = Path(__file__).parent.parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

DATA_DIR  = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(ROOT / "hf_home")

print("=" * 60)
print("Phase 1 — Organising datasets into data/")
print("=" * 60)

# ── 1. PlantVillage from git clone ────────────────────────────
pv_color = ROOT / "hf_cache" / "plantvillage_repo" / "raw" / "color"
print(f"\n[1/2] PlantVillage — reading from {pv_color}")
if not pv_color.exists():
    print(f"  ERROR: {pv_color} not found.")
    print("  Run: git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset hf_cache/plantvillage_repo")
    sys.exit(1)

total_pv = 0
pv_classes = sorted([d for d in pv_color.iterdir() if d.is_dir()])
for src_class_dir in pv_classes:
    cls_name = src_class_dir.name  # e.g. Tomato___Late_blight
    dest_dir = DATA_DIR / cls_name
    dest_dir.mkdir(exist_ok=True)
    imgs = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.JPG")) + list(src_class_dir.glob("*.jpeg"))
    for img in imgs:
        dst = dest_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        total_pv += 1
print(f"  Linked {total_pv} images across {len(pv_classes)} PlantVillage classes.")

# ── 2. Fresh/Rotten fruit from HuggingFace ───────────────────
print("\n[2/2] Fresh/Rotten fruit (Project-AgML/fresh_rotten_fruit_classification) ...")
try:
    from datasets import load_dataset
    veg = load_dataset(
        "Project-AgML/fresh_rotten_fruit_classification",
        "augmented",
        trust_remote_code=True,
    )
    lab = veg["train"].features["label"].names  # ["fresh","rotten"]
    total_veg = 0
    for row in veg["train"]:
        freshness = lab[row["label"]]
        fruit = row["fruit_type"].lower().replace(" ", "_").replace("-", "_")
        cls_name = f"{freshness}_{fruit}"
        dest_dir = DATA_DIR / cls_name
        dest_dir.mkdir(exist_ok=True)
        n = len(list(dest_dir.glob("*.jpg")))
        dest = dest_dir / f"{n}.jpg"
        row["image"].save(str(dest))
        total_veg += 1
    print(f"  Saved {total_veg} fresh/rotten images.")
except Exception as e:
    print(f"  ERROR downloading vegetable dataset: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. Report ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Class list with image counts:")
print("=" * 60)
classes = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
total = 0
tiny = []
for c in classes:
    n = len(list(c.glob("*.jpg")))
    if n > 0:
        flag = "  *** TINY (<20)" if n < 20 else ""
        print(f"  {c.name:<55s} {n:>5d}{flag}")
        total += n
        if n < 20:
            tiny.append(c.name)

print(f"\n  TOTAL: {total} images across {len([c for c in classes if c.is_dir()])} classes")
if tiny:
    print(f"\n  WARNING: {len(tiny)} tiny classes (<20 images) — consider dropping:")
    for t in tiny:
        print(f"    - {t}")
