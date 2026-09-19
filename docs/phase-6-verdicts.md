# Phase 6 — Verdict rules (disease name + safe-to-consume in words)

**Objective:** build the lookup table that turns a predicted class into human words: a disease name (if any), a safe/unsafe verdict, and a short note. This is a **rule table, not machine learning** — the model only predicts the class; this file gives it meaning.

## Tasks
- [ ] Create `src/verdicts.py` containing a dictionary that maps **every class name from `model/labels.txt`** to:
  - `status` — short status, e.g. "Healthy", "Late Blight", "Fresh", "Rotten"
  - `disease` — the disease name if it is a plant disease, else `"None"`
  - `safe` — `True` or `False`
  - `verdict` — the words to show, e.g. "Likely safe to consume" or "Not safe — discard"
  - `note` — one short line, e.g. "Looks healthy." / "Visible disease — remove affected parts / do not eat." / "Signs of rot — discard."
- [ ] Rules to apply:
  - Any `*___healthy` leaf class or `fresh_*` vegetable class → safe = True, verdict = "Likely safe to consume".
  - Any disease leaf class or `rotten_*` vegetable class → safe = False, verdict = "Not safe — discard", and set `disease` to the disease name for leaves.
- [ ] Add a helper `get_verdict(class_name)` that returns this info, with a safe default for unknown classes.
- [ ] **Always include the honesty line** as a constant the app will display: *"This is a visual estimate from appearance only, not a food-safety test."*

## Definition of done
`get_verdict()` returns the right words for every class in `model/labels.txt`, including the safe/unsafe verdict and disease name.

## When this phase is done
1. Tick Phase 6 in `todo.md`.
2. Update `HOW_TO_RUN.md` if needed.
3. Commit: `git commit -m "Phase 6: verdict rules (disease + safe-to-consume)"`.
