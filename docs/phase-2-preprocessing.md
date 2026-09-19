# Phase 2 — Preprocessing (multi-class)

**Objective:** turn the per-class image folders into train/validation/test data, and save the class-label list.

## Tasks
- [ ] Create `src/preprocess.py`.
- [ ] Load images from every class subfolder in `data/`.
- [ ] Resize to **224x224**, normalise pixels to **0–1**.
- [ ] Apply augmentation on the training data: horizontal flip, small rotation, brightness change.
- [ ] Split into **70% train / 15% validation / 15% test**, keeping class balance (stratified).
- [ ] **Save the ordered list of class names** to `model/labels.txt` (or JSON) — the app needs this to turn the model's output number back into a class name.
- [ ] Print the number of classes and the train/val/test sizes.

## Definition of done
Running `src/preprocess.py` prints the class count and set sizes with no errors, and `model/labels.txt` exists.

## When this phase is done
1. Tick Phase 2 in `todo.md` and note the number of classes + set sizes.
2. Update `HOW_TO_RUN.md` (how to run preprocessing).
3. Commit: `git commit -m "Phase 2: multi-class preprocessing + saved labels"`.
