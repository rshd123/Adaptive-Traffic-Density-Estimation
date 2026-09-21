# Implementation Plan

## Pipeline Overview

```
Traffic Image/Video → YOLOv8 → Vehicle Detection + Classification → PCU Weighting → Virtual Spatial Zones → Zone-wise PCU Density → Overall Traffic Density
```

---

## Phase 1: Data Preparation

### 1.1 Dataset Extraction & Preprocessing (`src/data/preprocessor.py`)
- Extract DriveIndia.zip (11.2 GB) — primary dataset
- Extract archive.zip (Kaggle, 607 MB) — supplementary
- Extract IDD_FGVD.tar.gz (2.6 GB) — fine-grained augmentation
- Filter all datasets to 5 target classes: Car, Motorcycle, Auto-rickshaw, Bus, Truck
- Convert all annotations to统一 YOLO format: `class_id x_center y_center width height`
- Merge datasets into unified train/val/test splits

### 1.2 Data Augmentation (`src/data/augmentor.py`)
- Standard: flip, rotation, scale, mosaic, mixup
- Domain-specific: brightness variation (night), blur (rain/fog), occlusion simulation
- Class-balanced sampling to handle imbalance (motorcycles overrepresented)

### 1.3 Data Loading (`src/data/dataloader.py`)
- YOLO-compatible dataset loading
- Train/val/test split management
- Class mapping and statistics

---

## Phase 2: Model Training

### 2.1 YOLOv8 Detection (`src/models/detector.py`)
- YOLOv8 backbone (n/s/m/l variants for ablation)
- Transfer learning from COCO pretrained weights
- Fine-tune on filtered 5-class Indian traffic dataset
- Hyperparameter tuning (learning rate, augmentation, epochs)

### 2.2 Training Scripts (`src/scripts/train.py`)
- Single entry point for training with config-driven parameters
- Logging, checkpointing, early stopping
- TensorBoard/WandB integration

---

## Phase 3: PCU-Weighted Density Estimation

### 3.1 PCU Weight Definitions (`src/models/pcu.py`, `src/configs/pcu_weights.yaml`)
- Standard PCU values (IRC-based):
  - Car = 1.0
  - Motorcycle = 0.5
  - Auto-rickshaw = 0.75
  - Bus = 3.0
  - Truck = 3.5
- Configurable weights for sensitivity analysis

### 3.2 Virtual Spatial Zones (`src/density/zones.py`)
- Divide road image into N width-wise vertical strips/zones
- Zone boundaries configurable (e.g., 3, 5, 7 zones)
- Handle perspective: zones are wider at bottom, narrower at top
- Assign each detected vehicle to its spatial zone based on bounding box center

### 3.3 Density Estimation (`src/density/estimator.py`)
- Per-vehicle PCU weight = PCU[class]
- Zone PCU density = sum(PCU weights of vehicles in zone) / zone area
- Overall PCU density = sum(all PCU weights) / total road area
- Temporal smoothing for video sequences

### 3.4 Heatmap Visualization (`src/density/heatmap.py`)
- Zone-wise color-coded density heatmap overlay on input image
- Per-zone density bar charts
- Overall congestion level indicator

---

## Phase 4: Evaluation

### 4.1 Metrics (`src/evaluation/metrics.py`)
- PCU density estimation accuracy vs ground truth
- Vehicle detection mAP (standard YOLO metrics)
- Zone assignment accuracy
- Comparison: PCU-weighted density vs raw vehicle count density

### 4.2 Visualization (`src/evaluation/visualize.py`)
- Detection results with PCU annotations
- Zone-wise density heatmaps
- Comparative plots (PCU density vs raw count)
- Per-class detection analysis

---

## Phase 5: Inference

### 5.1 Prediction (`src/scripts/predict.py`)
- Single image inference
- Video stream inference
- Real-time zone-wise density output
- Output: annotated image + density JSON

### 5.2 Evaluation Pipeline (`src/scripts/evaluate.py`)
- Full evaluation on test set
- Generate comparison report
- Export results for paper figures

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/data.yaml` | Dataset paths, class names, splits |
| `configs/model.yaml` | YOLOv8 architecture variant |
| `configs/pcu_weights.yaml` | PCU values per vehicle class |

---

## Class Mapping

| # | Target Class | DriveIndia Label | Kaggle Label | FGVD Label | PCU |
|---|---|---|---|---|---|
| 0 | Car | car | Car | car | 1.0 |
| 1 | Motorcycle | motorcycle/bike | Two Wheeler | motorcycle | 0.5 |
| 2 | Auto-rickshaw | auto-rickshaw | Auto | autorickshaw | 0.75 |
| 3 | Bus | bus | Bus | bus | 3.0 |
| 4 | Truck | truck | Truck | truck | 3.5 |
