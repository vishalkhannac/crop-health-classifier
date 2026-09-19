# Phase 3 — Model (multi-class)

**Objective:** build the MobileNetV2 transfer-learning model with one output per class.

## Tasks
- [ ] Create `src/model.py` with a function `build_model(num_classes)`.
- [ ] Load **MobileNetV2** pre-trained on ImageNet, without its top, and **freeze** the base.
- [ ] Add: global average pooling → one small dense layer → **softmax output with `num_classes` nodes**.
- [ ] Print the model summary.

## Definition of done
The model builds for the real number of classes from Phase 2 and the summary shows the frozen base + softmax head.

## When this phase is done
1. Tick Phase 3 in `todo.md` and note the number of classes and trainable parameters.
2. Update `HOW_TO_RUN.md` if anything new is runnable.
3. Commit: `git commit -m "Phase 3: MobileNetV2 multi-class model"`.
