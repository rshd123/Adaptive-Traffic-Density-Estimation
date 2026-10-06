# TODO

Remaining work for the project, based on **docs/CONTEXT.md** (the research goal) and **docs/IMPLEMENTATION.md** (the plan). Tasks are written as outcomes, not file names — the code layout can change freely.

---

## Done so far (quick status)

- Data subset built: 3,000 train / 400 val / 700 test images, config generated (`nc: 28`)
- Environment ready: conda env `ATDE`, torch with CUDA 13.0, ultralytics, GPU verified (RTX 5060)
- PCU weights defined (car 1.0 · motorcycle 0.5 · auto-rickshaw 0.75 · bus 3.0 · truck 3.5) with ID-based lookup
- Training entry point written (CLI ready to run)
- Pretrained base weights downloaded

---

## 1. Train the detector  (ready to run now)

- [ ] Run training: `python src/main.py train` (yolo26n, 50 epochs, batch 16, imgsz 640, GPU; early stop patience 15)
- [ ] Check quality: mAP on val for the 5 PCU classes should be good; rare classes may be weak — acceptable for the demo
- [ ] Confirm the best checkpoint is saved for later use

## 2. PCU-weighted density  (core of the project)

- [ ] **Virtual spatial zones** — split the road into width-wise strips (3/5/7, configurable), wider at the bottom to handle perspective; put each detected vehicle into a zone by its bounding-box center
- [ ] **Density estimation** — zone density = sum of PCU weights in that zone ÷ zone area; overall density = all PCU ÷ road area; smooth over time for video
- [ ] **Occlusion correction** — make up for vehicles hidden behind each other in crowded zones (under-counting)

## 3. Evaluation  (prove it actually works)

- [ ] Detection accuracy (mAP) on the test split
- [ ] Density error compared to ground truth
- [ ] **Baseline 1:** plain vehicle counting vs PCU-weighted density — this is the project's main comparison (from CONTEXT.md)
- [ ] **Baseline 2:** fixed lane-based zones vs virtual spatial zones
- [ ] Zone assignment accuracy

## 4. Visualization & demo

- [ ] Show detections with their PCU values
- [ ] Zone-wise density heatmap drawn over the road image
- [ ] Comparison plots: PCU density vs raw count
- [ ] Image/video inference that outputs an annotated result + density numbers
- [ ] Simple demo app: upload an image/video → get the density result

## 5. Research validation  (the risk flagged in CONTEXT.md)

- [ ] Justify the PCU weights properly (IRC-based values + references) — CONTEXT.md calls unvalidated/arbitrary PCU weights the main risk to publication
- [ ] Later: final names for all 28 class IDs (cosmetic only — never requires retraining)
- [ ] Later: real-world evaluation on Chennai traffic footage (external validation)

---

**Suggested order:** 1 → 2 → 3 → 4, with 5 running alongside. Item 1 can start immediately.
