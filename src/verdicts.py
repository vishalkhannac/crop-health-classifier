# src/verdicts.py — Phase 6: Disease name + safe-to-consume lookup table
# Maps every class name to: status, disease, safe, verdict, note.
# get_verdict(class_name) is the public API used by app.py

HONESTY_LINE = (
    "Visual estimate from appearance only — not a food-safety test."
)

# ─────────────────────────────────────────────────────────────
# PlantVillage classes (38 classes from mohanty/PlantVillage)
# Format: "ClassName": (status, disease_name, safe_bool, verdict_text, note)
# ─────────────────────────────────────────────────────────────
_PLANT_VERDICTS = {
    # Apple
    "Apple___Apple_scab":          ("Apple Scab",       "Apple Scab",               False, "Not safe — discard", "Visible fungal infection on apple leaf."),
    "Apple___Black_rot":           ("Black Rot",        "Black Rot",                False, "Not safe — discard", "Bacterial disease; affects fruit and leaves."),
    "Apple___Cedar_apple_rust":    ("Cedar Apple Rust", "Cedar Apple Rust",         False, "Not safe — discard", "Fungal rust lesions on leaf."),
    "Apple___healthy":             ("Healthy",          "None",                     True,  "Likely safe to consume", "Apple leaf looks healthy."),
    # Blueberry
    "Blueberry___healthy":         ("Healthy",          "None",                     True,  "Likely safe to consume", "Blueberry leaf looks healthy."),
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": ("Powdery Mildew", "Powdery Mildew", False, "Not safe — discard", "Fungal mildew on cherry leaf."),
    "Cherry_(including_sour)___healthy":         ("Healthy",        "None",          True,  "Likely safe to consume", "Cherry leaf looks healthy."),
    # Corn / Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Gray Leaf Spot",    "Cercospora / Gray Leaf Spot", False, "Not safe — discard", "Fungal leaf spot disease on maize."),
    "Corn_(maize)___Common_rust_":                        ("Common Rust",       "Common Rust",                 False, "Not safe — discard", "Rust fungus on maize leaf."),
    "Corn_(maize)___Northern_Leaf_Blight":                ("Northern Leaf Blight","Northern Leaf Blight",      False, "Not safe — discard", "Fungal blight on maize."),
    "Corn_(maize)___healthy":                             ("Healthy",           "None",                        True,  "Likely safe to consume", "Maize leaf looks healthy."),
    # Grape
    "Grape___Black_rot":                       ("Black Rot",         "Black Rot",                   False, "Not safe — discard", "Fungal disease on grape."),
    "Grape___Esca_(Black_Measles)":            ("Esca / Black Measles","Esca (Black Measles)",       False, "Not safe — discard", "Fungal complex disease on grape."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":("Leaf Blight",    "Isariopsis Leaf Spot",         False, "Not safe — discard", "Fungal blight on grape leaf."),
    "Grape___healthy":                         ("Healthy",           "None",                        True,  "Likely safe to consume", "Grape leaf looks healthy."),
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)":("Citrus Greening",  "Huanglongbing (Citrus Greening)", False, "Not safe — discard", "Bacterial disease; no cure — infected fruit unfit to eat."),
    # Peach
    "Peach___Bacterial_spot":  ("Bacterial Spot", "Bacterial Spot", False, "Not safe — discard", "Bacterial lesions on peach leaf."),
    "Peach___healthy":         ("Healthy",        "None",           True,  "Likely safe to consume", "Peach leaf looks healthy."),
    # Pepper / Bell
    "Pepper,_bell___Bacterial_spot": ("Bacterial Spot", "Bacterial Spot", False, "Not safe — discard", "Bacterial infection on pepper leaf."),
    "Pepper,_bell___healthy":        ("Healthy",        "None",           True,  "Likely safe to consume", "Bell pepper leaf looks healthy."),
    # Potato
    "Potato___Early_blight": ("Early Blight", "Early Blight",  False, "Not safe — discard", "Fungal blight — discard affected parts."),
    "Potato___Late_blight":  ("Late Blight",  "Late Blight",   False, "Not safe — discard", "Phytophthora late blight — do not eat."),
    "Potato___healthy":      ("Healthy",      "None",          True,  "Likely safe to consume", "Potato plant looks healthy."),
    # Raspberry
    "Raspberry___healthy": ("Healthy", "None", True, "Likely safe to consume", "Raspberry leaf looks healthy."),
    # Soybean
    "Soybean___healthy": ("Healthy", "None", True, "Likely safe to consume", "Soybean plant looks healthy."),
    # Squash
    "Squash___Powdery_mildew": ("Powdery Mildew", "Powdery Mildew", False, "Not safe — discard", "Fungal mildew on squash leaf."),
    # Strawberry
    "Strawberry___Leaf_scorch": ("Leaf Scorch", "Leaf Scorch", False, "Not safe — discard", "Fungal leaf scorch on strawberry."),
    "Strawberry___healthy":     ("Healthy",     "None",        True,  "Likely safe to consume", "Strawberry plant looks healthy."),
    # Tomato
    "Tomato___Bacterial_spot":                       ("Bacterial Spot",        "Bacterial Spot",                     False, "Not safe — discard", "Bacterial infection on tomato."),
    "Tomato___Early_blight":                         ("Early Blight",          "Early Blight",                       False, "Not safe — discard", "Fungal blight — remove and discard."),
    "Tomato___Late_blight":                          ("Late Blight",           "Late Blight",                        False, "Not safe — discard", "Phytophthora blight — do not eat."),
    "Tomato___Leaf_Mold":                            ("Leaf Mold",             "Leaf Mold",                          False, "Not safe — discard", "Fungal mold on tomato leaf."),
    "Tomato___Septoria_leaf_spot":                   ("Septoria Leaf Spot",    "Septoria Leaf Spot",                 False, "Not safe — discard", "Fungal leaf spot."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Spider Mite Damage",    "Spider Mites",                       False, "Not safe — discard", "Pest infestation — affected leaves/fruit not ideal."),
    "Tomato___Target_Spot":                          ("Target Spot",           "Target Spot",                        False, "Not safe — discard", "Fungal disease; circular lesions on leaf."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":        ("Yellow Leaf Curl Virus","Tomato Yellow Leaf Curl Virus",      False, "Not safe — discard", "Viral disease; tomatoes from infected plant not ideal."),
    "Tomato___Tomato_mosaic_virus":                  ("Mosaic Virus",          "Tomato Mosaic Virus",                False, "Not safe — discard", "Viral mosaic on tomato."),
    "Tomato___healthy":                              ("Healthy",               "None",                               True,  "Likely safe to consume", "Tomato plant looks healthy."),
}

# ─────────────────────────────────────────────────────────────
# Fresh / Rotten fruit classes
# Generated from Project-AgML/fresh_rotten_fruit_classification
# fruit types: apple, banana, guava, lime, orange, pomegranate, strawberry, tomato
# (and potentially others depending on dataset split)
# ─────────────────────────────────────────────────────────────
_FRESH_FRUITS  = ["apple","banana","guava","lime","orange","pomegranate","strawberry","tomato",
                  "grape","mango","kiwi","cherry","peach","watermelon","lemon","papaya","pineapple"]
_ROTTEN_FRUITS = _FRESH_FRUITS[:]

_FRUIT_VERDICTS = {}
for _f in _FRESH_FRUITS:
    _FRUIT_VERDICTS[f"fresh_{_f}"] = (
        f"Fresh {_f.capitalize()}", "None", True,
        "Likely safe to consume",
        f"Looks fresh — no visible rot or disease."
    )
for _f in _ROTTEN_FRUITS:
    _FRUIT_VERDICTS[f"rotten_{_f}"] = (
        f"Rotten {_f.capitalize()}", "None", False,
        "Not safe — discard",
        f"Signs of rot or decay — discard."
    )

# Merge
_ALL_VERDICTS = {**_PLANT_VERDICTS, **_FRUIT_VERDICTS}

# Default for unknown / unlisted classes
_DEFAULT = ("Unknown", "Unknown", False, "Uncertain — inspect carefully", "Class not in database; treat with caution.")


def get_verdict(class_name: str) -> dict:
    """
    Returns a dict with keys: status, disease, safe, verdict, note.
    Falls back gracefully for unknown class names.
    """
    # Try exact match first
    entry = _ALL_VERDICTS.get(class_name)
    # Fallback: normalise (strip, replace spaces with underscores)
    if entry is None:
        normalised = class_name.strip().replace(" ", "_")
        entry = _ALL_VERDICTS.get(normalised)
    # Fallback: try to infer from keywords
    if entry is None:
        lc = class_name.lower()
        if "healthy" in lc or lc.startswith("fresh_"):
            entry = ("Healthy / Fresh", "None", True, "Likely safe to consume", "Appears healthy or fresh.")
        elif any(k in lc for k in ("rot", "blight", "mold", "mildew", "rust", "scorch",
                                    "virus", "spot", "rotten", "decay")):
            entry = ("Diseased / Rotten", "Detected", False, "Not safe — discard",
                     "Visible disease or rot — remove and discard.")
        else:
            entry = _DEFAULT
    status, disease, safe, verdict, note = entry
    return dict(status=status, disease=disease, safe=safe, verdict=verdict, note=note)


if __name__ == "__main__":
    # Quick test
    tests = [
        "Tomato___Late_blight",
        "Apple___healthy",
        "fresh_banana",
        "rotten_tomato",
        "Grape___healthy",
        "completely_unknown_class",
    ]
    print(f"{'Class':<50s} {'Safe':<6} {'Verdict'}")
    print("-"*90)
    for cls in tests:
        v = get_verdict(cls)
        print(f"{cls:<50s} {str(v['safe']):<6} {v['verdict']}")
