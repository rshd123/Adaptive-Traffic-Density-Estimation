# Implementation Plan

## Pipeline Overview

```
Traffic Image/Video → YOLO26n → Vehicle Detection + Classification → PCU Weighting → Virtual Spatial Zones → Zone-wise PCU Density → Overall Traffic Density
```

Demo scope: single model (`yolo26n`), DriveIndia dataset only, no Kaggle/FGVD merging.

---

## Phase 1: Data Preparation

### 1.1 Source Data (already extracted)

Data lives directly in `datasets/` (extracted from the DriveIndia inner zips — extraction is done):

| Split | Source | Counts |
|---|---|---|
| train | `datasets/train{1,2,3}/{images,labels}/` | 8,333 + 8,333 + 8,334 = 25,000 imgs/labels |
| val | `datasets/val/{images_2500,labels_2500}/` | 2,500 imgs, 2,411 labels (89 unlabeled) |
| test | `datasets/test/{images_2500,labels_2500}/` | 2,500 imgs, 2,398 labels (102 unlabeled) |

The preprocessor reads **directly from `datasets/`** — nothing is re-extracted or moved. Only the selected subset is copied → `data/subset/{train,val,test}/{images,labels}/`.

Skip: `truncated_datasets/` (kept for future use), `archive.zip` and `IDD_FGVD.tar.gz` (excluded from demo).

Native resolution 1920×1080 — keep as-is, YOLO resizes internally. Images with no label file are excluded from training.

### 1.2 Class IDs — names are cosmetic (NOT blocking)

DriveIndia ships **no `data.yaml`**, but training, PCU weighting and density all key on the **integer class ID**, so name verification no longer blocks training:

- Labels are `class_id cx cy w h` — YOLO trains on the numbers; the `names:` list only labels charts/reports.
- PCU lookup is ID-based (`configs/pcu_weights.yaml`: 2/3/10/5/7).
- `configs/data.yaml` is generated with `preprocessor.py config` (default `--names pcu`: plain `id<N>` for every class, real names only for the 5 PCU classes). `--names ids` = all plain IDs, `--names paper` = hypothesized names (reference only).
- **PCU IDs spot-checked** via rendered atlases (`data/verify/pcul_*.jpg`, `atlas_*.jpg`): ID2=car, ID3=motorcycle, ID10=auto-rickshaw, ID5=bus, ID7=truck ✓ (a few noisy boxes, classes correct).
- **Tail IDs 14–27**: hypothesized names known-wrong and irrelevant — density ignores every class except the 5 PCU IDs. Anomaly IDs 24–27 kept as-is (no remapping). Final names for all 28 classes to be supplied later — switching names never requires retraining.
- Missing IDs 11, 13, 17 simply stay empty slots (nc = 28 = max id + 1).

### 1.3 Stratified Subset Reduction (`src/data/preprocessor.py`)

Full dataset (30K imgs) is unnecessary for the demo. Agreed targets, kept within the predefined splits:

- **train: 2,500–3,000** · **val: 300–500** · **test: 500–1,000**

Selection algorithm:

1. Exclude unlabeled images (train has none; val misses 89, test misses 102).
2. Cap over-represented classes (car, motorcycle dominate).
3. Rare-class-first greedy fill: prioritize images containing the rarest class IDs (12, 14, 15, 18, 20–27 …), then fill remaining quota.
4. Stride sampling within video sequences (take every Nth frame) to avoid near-duplicate frames from the F_/R_ dashcam feeds.
5. Copy selected files → `data/subset/{train,val,test}/{images,labels}/`.

Keep **all 24 classes** (class count barely affects training time). No class remapping — PCU lookup filters to the 5 target classes at inference.

Expected accuracy: with 2.5–3K train imgs, mAP for the 5 PCU classes should be good (they are the high-frequency ones); rare classes may be weak — acceptable for a demo.

### 1.4 Config Generation

Generate `configs/data.yaml` from the subset (`preprocessor.py config`, default `--names pcu`):

```yaml
path: <abs path to data/subset>
train: train/images
val: val/images
test: test/images
nc: 28
names:
  0: id0
  1: id1
  2: car          # the 5 PCU classes get real names
  3: motorcycle
  4: id4
  5: bus
  6: id6
  7: truck
  8: id8
  9: id9
  10: auto-rickshaw
  11: id11        # ...everything else stays id<N> until final names arrive
```

---

## Phase 2: Model Training

### 2.1 Detector (`src/models/detector.py`, entry: `src/main.py`)

- **Environment: conda env `ATDE`** (Miniconda) — no `.venv` anywhere. Setup once:
  ```powershell
  conda activate ATDE
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # RTX 5060 = Blackwell, needs CUDA 13.0 wheels
  python -m pip install -r requirements.txt
  ```
- Model: `yolo26n` (nano), transfer learning from pretrained `weights/yolo26n.pt`
- Data: `configs/data.yaml` → `data/subset/`
- Run: `python src/main.py train` (options: `--epochs --imgsz --batch --device --patience`)
- Hyperparameters (demo baseline):

| Param | Value |
|---|---|
| epochs | 50 (early stop, patience 15) |
| imgsz | 640 |
| batch | 16 |
| device | 0 (RTX 5060, 8 GB) |

- Expected time: ~5–15 min on 2.5–3K train imgs (vs ~1–2 hrs on full 30K)
- Ultralytics built-in augmentation (mosaic, hsv, flip) is sufficient — no custom augmentor
- Output: `runs/detect/train*/weights/best.pt` → copy to `weights/`

### 2.2 Training config

Driven by CLI args / YAML in `configs/` — no separate trainer module.

---

## Phase 3: PCU-Weighted Density Estimation

### 3.1 PCU Weight Definitions (`src/models/pcu.py`, `configs/pcu_weights.yaml`)

IRC-based PCU values (configurable for sensitivity analysis):

| Class | PCU |
|---|---|
| Car | 1.0 |
| Motorcycle | 0.5 |
| Auto-rickshaw | 0.75 |
| Bus | 3.0 |
| Truck | 3.5 |

PCU applies only to these 5 classes; other detections are ignored at density time.

### 3.2 Virtual Spatial Zones (`src/density/zones.py`)

- Divide road image into N width-wise vertical strips/zones (configurable: 3, 5, 7)
- Perspective handling: zones wider at bottom, narrower at top
- Assign each detected vehicle to its zone by bounding-box center

### 3.3 Density Estimation (`src/density/estimator.py`)

- Per-vehicle PCU weight = `PCU[class]`
- Zone PCU density = sum(PCU in zone) / zone area
- Overall PCU density = sum(all PCU) / total road area
- Temporal smoothing for video sequences

### 3.4 Occlusion Correction (`src/occlusion.py`)

- Correct under-count from partially occluded/overlapping vehicles (crowded zones)

---

## Phase 4: Evaluation

### 4.1 Metrics (`src/evaluation/metrics.py`)

- Detection mAP (standard YOLO metrics) on `data/subset/test`
- PCU density estimation error vs ground truth
- Zone assignment accuracy

### 4.2 Baselines (`src/baselines.py`)

- Raw vehicle count density (no PCU) vs PCU-weighted density
- Fixed lane-based zones vs virtual spatial zones

### 4.3 Visualization (`src/evaluation/visualize.py`)

- Detections with PCU annotations
- Zone-wise density heatmap overlay
- Comparative plots (PCU density vs raw count)

---

## Phase 5: Inference / Demo

### 5.1 Entry (`src/main.py`)

- Image / video inference → annotated output + zone-wise density JSON

### 5.2 Demo app (`src/app.py`)

- Simple interactive demo (upload image/video → density result)

---

## Configuration Files

| File | Purpose |
|---|---|
| `configs/data.yaml` | Dataset paths, class names, splits (generated in 1.4) |
| `configs/pcu_weights.yaml` | PCU values per vehicle class |

---

## Class Mapping

Hypothesized DriveIndia ID → name (from paper Table II + val frequency match — **must pass visual verification, §1.2, before training**):

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | pedestrian | 12 | construction vehicle |
| 1 | bicycle | 13 | animal |
| 2 | **car** | 14 | (verify — freq 21) |
| 3 | **motorcycle** | 15 | (verify — freq 7) |
| 4 | route board | 16 | (missing in val) |
| 5 | **bus** | 17 | (missing in val) |
| 6 | commercial vehicle | 18 | (verify — freq 3) |
| 7 | **truck** | 19 | (missing in val) |
| 8 | traffic sign | 20 | (verify — freq 16) |
| 9 | traffic light | 21 | (verify — freq 5) |
| 10 | **auto-rickshaw** | 22 | (verify — freq 5) |
| 11 | ambulance (missing in val) | 23 | (verify — freq 70) |
| | | 24–27 | **anomaly** — present in val (77/8/7/3), must resolve |

**Bold** = the 5 PCU target classes. Anomaly: IDs 24–27 appear in val but IDs 11/13/16/17/19 never do.

PCU target classes (post-verification): car 1.0 · motorcycle 0.5 · auto-rickshaw 0.75 · bus 3.0 · truck 3.5

---

## Data Notes

- DriveIndia claims 66,986 imgs / 24 classes in the paper; delivered zip has 30,000 (partial release)
- Splits: train 8,333 + 8,333 + 8,334 = 25,000 · val 2,500 (2,411 labeled) · test 2,500 (2,398 labeled)
- Prefixes: `F_` = front camera, `R_` = rear camera
- Val class frequencies: 0:3594, 1:38, 2:3748, 3:3153, 4:89, 5:335, 6:443, 7:505, 8:313, 9:69, 10:956, 12:5, 14:21, 15:7, 18:3, 20:16, 21:5, 22:5, 23:70, 24:77, 25:8, 26:7, 27:3
