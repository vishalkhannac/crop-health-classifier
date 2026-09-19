# Phase 5 — Evaluation (multi-class)

**Objective:** measure real performance on the unseen test set.

## Tasks
- [ ] Create `src/evaluate.py`.
- [ ] Evaluate the trained model on the **test set**.
- [ ] Print **overall test accuracy**.
- [ ] Generate and save a **confusion matrix** across all classes.
- [ ] Print a **per-class report** (precision, recall, F1 — use scikit-learn's classification_report).
- [ ] **If accuracy is poor:** add more augmentation, drop tiny classes, and/or train for more epochs (back to Phase 4), then re-evaluate.

## Definition of done
Overall accuracy is printed and recorded, and a confusion-matrix image + per-class report are saved.

## When this phase is done
1. Tick Phase 5 in `todo.md` and **record the exact test accuracy number** in the notes.
2. Update `HOW_TO_RUN.md` (how to evaluate + where results are saved).
3. Commit: `git commit -m "Phase 5: multi-class evaluation and accuracy recorded"`.
