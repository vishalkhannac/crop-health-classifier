# Phase 1 — Data (multi-class)

**Objective:** get the images onto disk with **one subfolder per class**, keeping the detailed names (disease names for leaves, fresh/rotten-per-item for vegetables).

## Tasks
- [ ] **Plant leaves — download PlantVillage from GitHub (no login):**
  - Clone `https://github.com/spMohanty/PlantVillage-Dataset` (or a similar public PlantVillage mirror).
  - Use its colour images. Each class is already its own folder with a disease name (e.g. `Tomato___Late_blight`, `Apple___healthy`).
- [ ] **Vegetables — download a Fresh/Rotten Fruits & Vegetables dataset from Kaggle:**
  - If Kaggle needs authentication, ask me for my Kaggle API token (`kaggle.json`) **or** tell me to download it manually and where to put it.
  - Keep its per-item classes (e.g. `fresh_tomato`, `rotten_potato`).
- [ ] **Combine into one `data/` folder** where every class is its own subfolder. Do **not** merge into two folders — keep the individual class names.
- [ ] Print the **list of all class names** and the image count for each.
- [ ] If some classes have very few images, note them (we may drop the tiniest ones to keep training stable).

## Definition of done
`data/` contains one subfolder per class with the detailed names, and you have printed the full class list with counts.

## When this phase is done
1. Tick Phase 1 in `todo.md` and paste the class list + counts in the notes.
2. Update `HOW_TO_RUN.md` ("Data" section: PlantVillage from GitHub, vegetables from Kaggle/manual, and where it all lives).
3. Commit: `git commit -m "Phase 1: multi-class datasets (PlantVillage from GitHub + vegetables)"`.
