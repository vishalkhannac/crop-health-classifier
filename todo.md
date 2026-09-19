# todo.md — Project Navigator (read me first)

**Antigravity agent: this file is the master plan. Read it fully, then build the project by following it.**

You are building a **Plant & Vegetable Health Detection** app. For any uploaded leaf or vegetable photo it reports: the **status / class name**, a **confidence meter (%)**, the **disease name** (where relevant), and a **"safe to consume" verdict in words** — always shown as a visual estimate, not a food-safety test.

## Read these two files first, in order
1. `docs/00-overview.md` — the fixed technical choices and folder structure.
2. `docs/00b-review1-scope.md` — **this is your current target.** This is a first project review, not the final build: aim for roughly 60–70% overall completion, with the frontend app fully polished even though the model is still an early version. Follow this scope file's checklist over the general "definition of done" in each phase file wherever the two differ.

Then work through the phases below in order, stopping where the scope file says to stop.

---

## Operating rules (follow these the whole way through)
1. **Work in order.** Do not start a phase until the one before it is finished and working.
2. **Open the phase file first.** Before doing a phase, read its file in `docs/` and follow it — but check `docs/00b-review1-scope.md` for what's actually required right now.
3. **One phase = working before you move on.** If a script errors, fix it before continuing.
4. **The frontend (Phase 7) is the one place to go beyond "just working."** Polish the layout, labels, and visuals — this is what will be shown at the review.
5. **Stop after Phase 7 for this review.** Do not start Phase 8 (finalise & push to a new repo) until I say the review is done.
6. **After finishing each phase, always do these three things:**
   - Tick the phase in the *Progress* list below and add a one-line note of what you built.
   - Update `HOW_TO_RUN.md` so it always reflects how to run what exists so far.
   - Commit your work locally: `git commit -m "Phase N: <short description>"`.
7. **Only ask me when you are genuinely blocked** — a decision I must make, or a login you need (Kaggle for the vegetable data, or GitHub later). Plant data (PlantVillage) comes from GitHub and needs no login. Otherwise keep going and make sensible choices consistent with `docs/00-overview.md`.
8. **Keep `HOW_TO_RUN.md` and this `todo.md` accurate at all times** — update them on a regular basis, not just at the end.

---

## Progress (update this as you go)
- [x] **Phase 1 — Data (multi-class, PlantVillage from GitHub)** → `docs/phase-1-data.md` — Complete: 54 classes (38 plant leaf diseases/health + 16 fresh/rotten produce), 66,638 images organised into `data/`.
- [ ] **Phase 2 — Preprocessing (+ save labels)** → `docs/phase-2-preprocessing.md`
- [ ] **Phase 3 — Model (softmax, multi-class)** → `docs/phase-3-model.md`
- [ ] **Phase 4 — Training (one working run — not tuned yet)** → `docs/phase-4-training.md`
- [ ] **Phase 5 — Evaluation (accuracy recorded, even if modest)** → `docs/phase-5-evaluation.md`
- [ ] **Phase 6 — Verdict rules (disease + safe-to-consume)** → `docs/phase-6-verdicts.md`
- [ ] **Phase 7 — App (frontend fully polished)** → `docs/phase-7-app.md`   ⟵ **REVIEW 1 STOPS HERE**
- [ ] *(after the review)* **Phase 8 — Finalise & push to new repo** → `docs/phase-8-finalise-and-push.md`

_Notes (fill in as you finish each phase):_
- Phase 1 (class list + counts): 54 classes, 66,638 images total across PlantVillage (54,303 images) and Fresh/Rotten produce (12,335 images).
- Phase 2 (num classes + set sizes):
- Phase 3:
- Phase 4 (val accuracy):
- Phase 5 (TEST ACCURACY):
- Phase 6:
- Phase 7:
- Phase 8 (repo URL, after the review):

---

### Definition of done for Review 1
Phases 1–7 ticked; the app runs, looks complete and polished, and returns a class + confidence + disease + safe-to-consume verdict for a healthy leaf, a diseased leaf, a fresh vegetable, and a rotten vegetable — even if accuracy is still modest. `HOW_TO_RUN.md` is accurate. Nothing has been pushed to a remote repo yet.

### Definition of done for the whole project (later, after the review)
All 8 phases ticked, accuracy improved from the first pass, and the project pushed to a new GitHub repository.
