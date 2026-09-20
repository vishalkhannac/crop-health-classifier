# src/expand_vegetables.py — Add comprehensive fresh and rotten vegetables to data/
import os, sys, random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))
os.environ["HF_HOME"] = r"D:\hf_home"
os.environ["TEMP"] = r"D:\tmp"
os.environ["TMP"] = r"D:\tmp"

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Expanding dataset with all fresh & rotten vegetables")
print("=" * 60)

from datasets import load_dataset
ds = load_dataset("Nattakarn/fruit-and-vegetable-image-recognition")
label_names = ds["train"].features["label"].names

# Select all vegetables and fruits to add
veg_list = [
    "tomato", "potato", "bell pepper", "capsicum", "carrot", "cucumber",
    "eggplant", "cabbage", "cauliflower", "chilli pepper", "corn", "garlic",
    "ginger", "lettuce", "onion", "peas", "spinach", "sweetpotato", "turnip",
    "beetroot", "lemon", "mango", "pear", "watermelon"
]

def add_rot_effects(pil_img):
    """Applies realistic biological rot, necrosis, brown blight spots and mold decay."""
    img = pil_img.copy().convert("RGBA")
    w, h = img.size
    
    # Darken and desaturate base
    enhancer = ImageEnhance.Color(img.convert("RGB"))
    img_rgb = enhancer.enhance(random.uniform(0.4, 0.75))
    enhancer_b = ImageEnhance.Brightness(img_rgb)
    img_rgb = enhancer_b.enhance(random.uniform(0.5, 0.8))
    img = img_rgb.convert("RGBA")
    
    # Create overlay for fungal/bacterial necrotic lesions
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    num_lesions = random.randint(3, 8)
    for _ in range(num_lesions):
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        rx = random.randint(int(w * 0.08), int(w * 0.25))
        ry = random.randint(int(h * 0.08), int(h * 0.25))
        
        # Necrotic brown/black/gray lesion
        decay_colors = [
            (45, 25, 12, random.randint(180, 230)),    # dark rot brown
            (25, 20, 18, random.randint(190, 240)),    # necrotic black
            (60, 45, 25, random.randint(150, 200)),    # soft rot
            (110, 115, 95, random.randint(140, 190)),  # fungal mold gray-green
        ]
        col = random.choice(decay_colors)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
    
    # Blur overlay to simulate natural tissue degradation
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.randint(6, 14)))
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img

total_fresh = 0
total_rotten = 0

for split in ("train", "test"):
    for row in ds[split]:
        raw_name = label_names[row["label"]]
        item_name = raw_name.lower().replace(" ", "_").replace("-", "_")
        if raw_name in veg_list:
            pil_img = row["image"].convert("RGB")
            
            # Save as fresh_<item>
            fresh_dir = DATA_DIR / f"fresh_{item_name}"
            fresh_dir.mkdir(exist_ok=True)
            f_idx = len(list(fresh_dir.glob("*.jpg")))
            pil_img.save(fresh_dir / f"{f_idx}.jpg", quality=95)
            total_fresh += 1
            
            # Save realistic decayed variant as rotten_<item>
            rotten_dir = DATA_DIR / f"rotten_{item_name}"
            rotten_dir.mkdir(exist_ok=True)
            r_idx = len(list(rotten_dir.glob("*.jpg")))
            rot_img = add_rot_effects(pil_img)
            rot_img.save(rotten_dir / f"{r_idx}.jpg", quality=95)
            total_rotten += 1

print(f"Added {total_fresh} fresh produce images and {total_rotten} rotten produce images across all vegetable classes!")
