# Phase 8 — Finalise & push to a new repository

**Objective:** confirm the whole project works end to end, finish the docs, and push to a brand-new GitHub repo. **This is the only phase that pushes to a remote.**

## Tasks
- [ ] Create `requirements.txt` (every package used) and a `README.md` describing the project, what it reports (status, confidence, disease, safe-to-consume), how to run it, the final test accuracy, and the honesty note about appearance-only judgement.
- [ ] Do a **full end-to-end check**:
  - preprocessing runs and writes `model/labels.txt`
  - the app launches and shows class + confidence + disease + verdict for a healthy leaf, a diseased leaf, a fresh vegetable, and a rotten vegetable
  - `HOW_TO_RUN.md` matches what actually happens when followed step by step
- [ ] Make sure `HOW_TO_RUN.md` and `todo.md` are fully up to date (all phases ticked, accuracy recorded).
- [ ] Add a `.gitignore` (ignore `data/`, large model files if big, and any virtual environment).
- [ ] **Confirm with me that everything works before pushing.** Show me the app result and the accuracy, and ask for a go-ahead.
- [ ] After I confirm, create a **new GitHub repository** and push:
  - If GitHub CLI is available: `gh repo create plant-veg-health --public --source=. --remote=origin --push`
  - Otherwise ask me to create an empty repo and give you the URL, then:
    `git remote add origin <URL>` → `git branch -M main` → `git push -u origin main`
  - If you need my GitHub login/token, ask me for it.

## Definition of done
Everything works end to end, docs are accurate, I have confirmed, and the project is live in a new GitHub repository.

## When this phase is done
1. Tick Phase 8 in `todo.md` and paste the repository URL in the notes.
2. Final commit + push.
