# 00b — Review 1 Scope (read this right after the overview)

**This is the first project review, not the final submission.** Build toward roughly **60–70% completion**, with one exception: the **frontend (the Streamlit app) must be fully built and polished**, even though the model behind it is still an early version.

## Build fully for Review 1
- [ ] **Phase 1 — Data**: datasets downloaded and organised into class folders.
- [ ] **Phase 2 — Preprocessing**: resize/normalise/augment/split pipeline working, labels saved.
- [ ] **Phase 3 — Model**: MobileNetV2 + softmax head builds correctly.
- [ ] **Phase 4 — Training**: trained at least once, end to end, no errors. It does **not** need to be tuned or highly accurate yet — a working first training run is enough for this review.
- [ ] **Phase 5 — Evaluation**: run once, accuracy and confusion matrix recorded, even if the number is modest.
- [ ] **Phase 6 — Verdict rules**: the disease/safe-to-consume lookup table, complete for all classes.
- [ ] **Phase 7 — App (frontend): build this to a genuinely finished, demo-ready standard.**
  - Clean layout, a clear title, instructions, an upload widget, an image preview, a visible confidence meter/progress bar, the class name, disease (if any), the safe-to-consume verdict in a highlighted box, and the honesty line about it being a visual estimate.
  - It should look and feel complete even though the model behind it is only an early version — **spend real effort polishing this part.**
  - Use the model from Phase 4 as-is; do not block the frontend on model accuracy.

## Deliberately skip or defer past Review 1
- [ ] **Do not tune hyperparameters or chase higher accuracy yet** — one clean training run is enough.
- [ ] **Do not do Phase 8 (finalise & push to a new GitHub repo) yet.** Keep everything committed locally only.
- [ ] Skip a polished README/requirements pass — a short one-paragraph note of what exists is enough for now.
- [ ] Skip testing on real-world/messy-background photos — PlantVillage-style clean photos are fine for this review.

## What "done for Review 1" looks like
Running `streamlit run app.py` opens a complete, good-looking app; uploading a sample photo returns a class, a confidence percentage, a disease name where relevant, and a safe-to-consume verdict — even if that verdict is sometimes wrong because the model is still early. That gap between "looks finished" and "isn't fully accurate yet" is expected and fine to say out loud in the review — it shows what's left for the next phase.

## After Review 1
Once the review is over, resume the normal `todo.md` order: go back to Phase 4/5 to improve accuracy, then do Phase 8 to finalise and push.
