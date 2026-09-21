# Implementation Plan — PCU-Weighted Spatial Traffic Density Estimation (Finalized)

## Dataset Decision

**Primary dataset: DriveIndia**
- Best fit ignoring size constraints — real Indian roads, 3000km of diverse traffic conditions, 24 classes including auto-rickshaw, already in YOLO format.
- Richer and more generalizable than the CCTV dataset (archive.zip), which is smaller and narrower in scene diversity.
- IDD_FGVD is not a good fit regardless of size — its 210 fine-grained classes are far more granular than needed (only 5 broad categories are required), and it requires manual bounding-box-to-YOLO conversion for little added benefit. Use it only in small amounts later if extra validation diversity is needed.

**Approach: Narrow DriveIndia down, don't use it in full**
1. **Remap classes** — collapse DriveIndia's 24 classes into the 5 needed categories: car, motorcycle, auto-rickshaw, bus, truck. Drop irrelevant classes.
2. **Stratified sampling** — don't randomly sample. Sample per-class so rare classes like bus and truck aren't underrepresented compared to cars and motorcycles.
3. **Prioritize congested scenes** — if location/scene metadata is available, favor denser junction-like traffic over open highway stretches, since that matches the occlusion-heavy use case this project targets.
4. **Target size** — aim for roughly 2,000–3,000 training images, 300–500 validation, 500–1,000 test, staying within the original prototype budget.
5. **Optional supplement** — once narrowed, add a small slice of the CCTV dataset (archive.zip) for extra domain diversity, since it's already YOLO-ready and uses a similar class set.

## Detection & Tracking

- Use **YOLO26n** (pretrained, fine-tuned on the narrowed dataset) as the vehicle detector — it's the current lightweight, real-time-capable Ultralytics model, NMS-free and fast on CPU, which fits a prototype well.
- Use **ByteTrack** for tracking. ByteTrack comes built into Ultralytics with no extra setup, and BoT-SORT's appearance-matching advantage isn't needed for a single-camera prototype.

## PCU Weighting

Assign a Passenger Car Unit value to each detected vehicle class instead of counting all vehicles equally:
- Motorcycle → 0.5
- Car → 1.0
- Auto-rickshaw → 1.0
- Bus → 3.0
- Truck → 3.0

These are provisional starting values and should be backed by a traffic-engineering reference before being used as final results.

## Virtual Spatial Zones

- Divide the camera's road view into virtual width-wise zones (not physical lanes), since Chennai roads often lack lane discipline.
- Each tracked vehicle is assigned to a zone based on its position, and its PCU value is added to that zone's total.
- **Known limitation to state upfront in the paper:** zone "area" here is measured in pixels, not real-world distance, so it's only accurate for elevated/top-down camera angles. For oblique angles this is an approximation; a full camera-calibration/homography correction is left as future work rather than solved in the prototype.

## Occlusion Correction

Two concrete, implementable techniques instead of a vague "confidence-aware counting" claim:
1. **Overlap correction** — when two same-class bounding boxes overlap heavily, apply a small correction multiplier to compensate for likely undercounted, clustered vehicles (especially common with tightly packed motorcycles).
2. **Track-gap filling** — if a tracked vehicle briefly disappears for a couple of frames and then reappears nearby, treat it as a temporary occlusion rather than counting it as a new or lost vehicle.

## Baselines & Ablation

Three versions are implemented and compared:
1. Raw vehicle count (no weighting, no zones)
2. PCU-weighted count only (no spatial zones)
3. Full proposed method — PCU weighting + spatial zones + occlusion correction

This shows, step by step, how much each added component improves accuracy over the simplest baseline — this is the core evidence for the paper's novelty claim.

## Evaluation

- **Detection quality** — precision, recall, mAP@50.
- **Density accuracy** — compare predicted PCU density against manually verified ground truth per zone, using MAE/RMSE. Ground truth should be built using CVAT to hand-annotate PCU-per-zone on a held-out test set, rather than just raw manual counts, since that output is directly usable for the accuracy tables.
- **Tracking (secondary)** — ID consistency and switch rate, only for the video-based experiments; not a primary focus since tracking isn't the core research contribution.

## Demo / Visualization

- Skip building a full FastAPI + React web app for the prototype stage — unnecessary overhead.
- Use **Streamlit or Gradio** instead: a single Python file that can show live video with bounding boxes, zone overlays, and PCU density numbers, shareable as a link. Fast to build, sufficient for demonstrating the system and for a paper's supplementary material.

## Chennai Validation

- Datasets used for training (DriveIndia, CCTV) represent Indian traffic broadly, but are not Chennai-specific.
- Collect a small, separate Chennai traffic test set (a handful of locations, mixed conditions) purely for **external validation** — to check whether the trained system generalizes to Chennai-style traffic, not as a training source.

## Priority Principle

Working system + clear, testable research contribution + measurable comparative results — not a huge dataset, maximum detector accuracy, or a production-grade deployment. Increase dataset size or model complexity only if the initial results turn out inadequate.


## Tech Stack

### Core ML / Computer Vision
- **YOLO26n** — vehicle detection and classification (Ultralytics)
- **ByteTrack** — multi-object tracking (built into Ultralytics, no extra setup)
- **PyTorch** — underlying framework for model training/inference
- **OpenCV** — video/image processing, drawing bounding boxes, zone overlays
- **NumPy** — numerical calculations (IoU, PCU math, zone assignment)
- **Pandas** — dataset handling, class remapping, evaluation tables

### Dataset Preparation
- **PyYAML** — reading class-mapping and data config files
- **CVAT** — annotation/verification, and building ground-truth PCU-per-zone labels for evaluation

### Evaluation / Visualization
- **Matplotlib** — density graphs, ablation comparison plots, per-zone charts
- **Scikit-learn** — MAE/RMSE and other evaluation metrics
