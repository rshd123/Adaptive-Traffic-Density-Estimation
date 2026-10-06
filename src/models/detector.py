"""YOLO detector - training entry for the DriveIndia subset.

torch / ultralytics are imported inside the functions so the rest of the
project (preprocessor, pcu) keeps working without the training deps.

Usage (ATDE conda env):
    python src/main.py train
    python src/main.py train --epochs 50 --batch 16 --device 0
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "configs" / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs" / "detect"


def resolve_model(model: str = "yolo26n.pt") -> str:
    """Prefer a non-empty local copy in weights/, else let ultralytics fetch."""
    local = WEIGHTS_DIR / Path(model).name
    if local.is_file() and local.stat().st_size > 0:
        return str(local)
    p = Path(model)
    if p.is_file() and p.stat().st_size > 0:
        return str(p)
    return model


def train(
    data: str | Path = DATA_YAML,
    model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    patience: int = 15,
    project: str | Path = RUNS_DIR,
    name: str = "train",
) -> Path:
    """Transfer-learn the detector on the DriveIndia subset.

    Returns the path of the copied best checkpoint (weights/*_best.pt).
    """
    from ultralytics import YOLO

    data = Path(data)
    if not data.is_file():
        raise SystemExit(
            f"missing dataset config: {data}\n"
            "run: python src/data/preprocessor.py build && python src/data/preprocessor.py config"
        )

    src = resolve_model(model)
    yolo = YOLO(src)

    results = yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished without best.pt in {results.save_dir}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / f"{Path(src).stem}_best.pt"
    dest.write_bytes(best.read_bytes())
    print(f"best checkpoint -> {dest}")
    return dest
