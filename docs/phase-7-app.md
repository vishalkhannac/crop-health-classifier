# Phase 7 — App (frontend — build this to a finished standard)

**Objective:** save the model and wrap everything in a Streamlit web app that shows status, confidence, disease, and the safe-to-consume verdict. **This is the one part of the Review 1 build that should look and feel fully complete** — see `docs/00b-review1-scope.md`.

## Tasks
- [ ] Save the trained model to `model/model.keras` (and make sure `model/labels.txt` from Phase 2 is present).
- [ ] Create `app.py`, a Streamlit app that includes:
  - **Header** — app title and a one-line description of what it does.
  - **Upload widget** — lets the user upload a leaf or vegetable photo.
  - **Image preview** — shows the uploaded photo back to the user.
  - **Prediction** — runs the same 224x224 preprocessing, predicts the class.
  - **Confidence meter** — the probability shown as a % *and* a visual progress bar, not just a number.
  - **Status / class name** — e.g. "Tomato — Late Blight".
  - **Disease** — shown clearly where relevant (via `src/verdicts.py`).
  - **Safe-to-consume verdict** — shown in a clearly highlighted box (e.g. green for safe, red/orange for not safe), in words, not just True/False.
  - **Honesty line** — always visible: *"Visual estimate from appearance only, not a food-safety test."*
  - **Empty state** — before any image is uploaded, show a friendly prompt instead of a blank page or an error.
- [ ] Polish pass: consistent spacing, a page-wide layout (`st.set_page_config`), and no raw tracebacks visible to the user even if something goes wrong (wrap prediction in a try/except with a friendly message).
- [ ] Run the app and open it in Antigravity's built-in browser.
- [ ] Test with a healthy leaf, a diseased leaf, a fresh vegetable, and a rotten vegetable — confirm the layout looks good for all four, not just one.

## Definition of done (for Review 1)
The app runs, looks like a finished product (not a bare prototype), and for each test image shows a class name, a confidence % with a visual meter, a disease (where relevant), and a clearly styled safe/unsafe verdict — regardless of whether the underlying model is fully accurate yet.

## When this phase is done
1. Tick Phase 7 in `todo.md`.
2. Update `HOW_TO_RUN.md` (how to launch the app — the main "how to run" for me).
3. Commit: `git commit -m "Phase 7: polished Streamlit app with confidence + disease + safe verdict"`.
4. **Stop here for Review 1** — do not start Phase 8 unless I say the review is done.
