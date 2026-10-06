"""DriveIndia preprocessor.

Subcommands:
    stats   - per-split class/image statistics (also writes data/stats.json)
    verify  - render sampled images with hypothesized class names -> data/verify/
    build   - stratified subset selection -> data/subset/{train,val,test}/{images,labels}
    config  - generate configs/data.yaml from the built subset

Usage (ATDE env):
    python src/data/preprocessor.py stats
    python src/data/preprocessor.py verify --n 40
    python src/data/preprocessor.py build --train 3000 --val 400 --test 700
    python src/data/preprocessor.py config
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SUBSET = ROOT / "data" / "subset"
VERIFY_OUT = ROOT / "data" / "verify"
STATS_JSON = ROOT / "data" / "stats.json"
CONFIG_YAML = ROOT / "configs" / "data.yaml"

SPLITS = {
    "train": [DATASETS / "train1", DATASETS / "train2", DATASETS / "train3"],
    "val": [DATASETS / "val"],
    "test": [DATASETS / "test"],
}

# Hypothesized DriveIndia ID -> name (paper Table II + val frequency match).
# IDs 14-23 ordering unverified; 24-27 are an anomaly (present in val only).
# MUST be visually confirmed (verify subcommand) before training.
NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "route board",
    5: "bus",
    6: "commercial vehicle",
    7: "truck",
    8: "traffic sign",
    9: "traffic light",
    10: "auto-rickshaw",
    11: "ambulance",
    12: "construction vehicle",
    13: "animal",
    14: "speed bump?",
    15: "pothole?",
    16: "police?",
    17: "tractor?",
    18: "pushcart?",
    19: "unknown19?",
    20: "barrier?",
    21: "rumble strip?",
    22: "cone?",
    23: "zebra crossing?",
    24: "anomaly24?",
    25: "anomaly25?",
    26: "anomaly26?",
    27: "anomaly27?",
}

# IRC-based PCU weights for the 5 target classes (IDs per mapping above).
PCU = {2: 1.0, 3: 0.5, 10: 0.75, 5: 3.0, 7: 3.5}

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#a9a9a9", "#00ff00", "#ff00ff", "#00ffff", "#ff8c00",
    "#7b68ee", "#32cd32", "#dc143c",
]


# --------------------------------------------------------------------------- io

def _label_dir(img_dir: Path) -> Path:
    """images / images_2500 -> labels / labels_2500"""
    name = img_dir.name.replace("images", "labels", 1)
    return img_dir.parent / name


def _split_dirs(split: str) -> list[Path]:
    """Image dirs for a split (handles images vs images_2500 naming)."""
    out = []
    for root in SPLITS[split]:
        if not root.is_dir():
            raise SystemExit(f"missing split dir: {root}")
        out.extend(sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("images")))
    return out


def scan(split: str) -> list[dict]:
    """All labeled images of a split as sample dicts."""
    samples = []
    for img_dir in _split_dirs(split):
        lbl_dir = _label_dir(img_dir)
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.is_file():
                continue
            classes = []
            for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if parts:
                    try:
                        classes.append(int(float(parts[0])))
                    except ValueError:
                        pass
            seq, _, frame = img.stem.rpartition("_")
            samples.append({
                "split": split,
                "img": img,
                "lbl": lbl,
                "classes": classes,
                "seq": seq or img.stem,
                "frame": int(frame) if frame.isdigit() else 0,
            })
    return samples


def unlabeled_count(split: str) -> int:
    n = 0
    for img_dir in _split_dirs(split):
        lbl_dir = _label_dir(img_dir)
        for img in img_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and not (lbl_dir / (img.stem + ".txt")).is_file():
                n += 1
    return n


def name(cid: int) -> str:
    return NAMES.get(cid, f"id{cid}")


# ------------------------------------------------------------------- subcommands

def cmd_stats(_args) -> None:
    report = {}
    total_img = total_lbl = 0
    for split in SPLITS:
        samples = scan(split)
        inst = Counter(c for s in samples for c in s["classes"])
        per_img = Counter(c for s in samples for c in set(s["classes"]))
        unl = unlabeled_count(split)
        report[split] = {
            "labeled_images": len(samples),
            "unlabeled_images": unl,
            "instances": dict(sorted(inst.items())),
            "images_per_class": dict(sorted(per_img.items())),
        }
        total_img += len(samples) + unl
        total_lbl += len(samples)
        print(f"\n== {split} ==  labeled={len(samples)}  unlabeled={unl}")
        print(f"{'id':>3} {'name':<20} {'instances':>9} {'images':>7}")
        for cid in sorted(set(inst) | set(per_img)):
            print(f"{cid:>3} {name(cid):<20} {inst.get(cid, 0):>9} {per_img.get(cid, 0):>7}")
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ntotal images={total_img}  labeled={total_lbl}  -> {STATS_JSON}")


def cmd_verify(args) -> None:
    from PIL import Image, ImageDraw

    pool = [s for split in SPLITS for s in scan(split) if s["classes"]]
    freq = Counter(c for s in pool for c in set(s["classes"]))

    # guarantee coverage: round-robin over classes (rarest first),
    # max `per_class` images per class so no class is starved by the cap
    by_class = defaultdict(list)
    for s in sorted(pool, key=lambda s: (min(freq[c] for c in set(s["classes"])), s["img"].name)):
        for c in set(s["classes"]):
            by_class[c].append(s)

    per_class = max(1, args.n // max(1, len(freq)))
    picked, seen, cursor = [], set(), {c: 0 for c in freq}
    progress = True
    while len(picked) < args.n and progress:
        progress = False
        for cid in sorted(freq, key=lambda c: freq[c]):
            lst = by_class[cid]
            added = 0
            while added < per_class and cursor[cid] < len(lst) and len(picked) < args.n:
                s = lst[cursor[cid]]
                cursor[cid] += 1
                if s["img"] in seen:
                    continue
                seen.add(s["img"])
                picked.append(s)
                added += 1
                progress = True
            if len(picked) >= args.n:
                break

    if VERIFY_OUT.exists():
        shutil.rmtree(VERIFY_OUT)
    VERIFY_OUT.mkdir(parents=True)

    rng = random.Random(args.seed)
    rows = []
    for i, s in enumerate(sorted(picked, key=lambda s: (s["split"], s["img"].name)), 1):
        img = Image.open(s["img"]).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)
        for line in s["lbl"].read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
            color = PALETTE[cid % len(PALETTE)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            label = f"{cid} {name(cid)}"
            tb = draw.textbbox((x0, y0), label)
            draw.rectangle([tb[0], tb[1] - 4, tb[2] + 4, tb[3] + 4], fill=color)
            draw.text((x0 + 2, y0 - 2), label, fill="white")
        out = VERIFY_OUT / f"{i:02d}_{s['split']}_{s['img'].stem}.jpg"
        img.save(out, quality=88)
        rows.append((out.name, s["split"], sorted(set(s["classes"]))))

    html = ["<html><head><meta charset='utf-8'><title>class verification</title><style>",
            "body{font-family:sans-serif;background:#111;color:#eee}",
            "img{width:47%;margin:1%;border:1px solid #444}",
            "p{font-size:12px;margin:2px 0 12px}</style></head><body>",
            "<h2>DriveIndia class ID verification (hypothesized names)</h2>"]
    for fn, split, cids in rows:
        html.append(f"<img src='{fn}'>")
        html.append(f"<p><b>{fn}</b> [{split}] classes: {', '.join(str(c) for c in cids)}</p>")
    html.append("</body></html>")
    (VERIFY_OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"{len(rows)} images -> {VERIFY_OUT}\nopen {VERIFY_OUT / 'index.html'} and confirm names")


def cmd_build(args) -> None:
    targets = {"train": args.train, "val": args.val, "test": args.test}
    report = {}

    for split, quota in targets.items():
        pool = scan(split)
        if not pool:
            raise SystemExit(f"no labeled samples in {split}")
        freq = Counter(c for s in pool for c in set(s["classes"]))
        anomaly = sum(1 for s in pool for c in s["classes"] if c >= 24)

        # rarest-class-first: key = frequency of the rarest class in the image
        ordered = sorted(pool, key=lambda s: (min(freq[c] for c in set(s["classes"])), s["seq"], s["frame"]))

        accepted, last_frame = [], {}
        for gap in [args.gap, max(1, args.gap // 2), 1, 0]:
            for s in ordered:
                if len(accepted) >= quota:
                    break
                if s in accepted:
                    continue
                prev = last_frame.get(s["seq"])
                if prev is not None and s["frame"] - prev < gap:
                    continue
                accepted.append(s)
                last_frame[s["seq"]] = s["frame"]
            if len(accepted) >= quota:
                break
        accepted = accepted[:quota]

        out_root = SUBSET / split
        if out_root.exists():
            shutil.rmtree(out_root)
        (out_root / "images").mkdir(parents=True)
        (out_root / "labels").mkdir(parents=True)
        for s in accepted:
            shutil.copy2(s["img"], out_root / "images" / s["img"].name)
            shutil.copy2(s["lbl"], out_root / "labels" / s["lbl"].name)

        sub_inst = Counter(c for s in accepted for c in s["classes"])
        report[split] = {
            "selected": len(accepted),
            "quota": quota,
            "instances": dict(sorted(sub_inst.items())),
        }
        print(f"\n== {split} == {len(accepted)}/{quota} images copied")
        if anomaly:
            print(f"  note: {anomaly} label instances with class id >= 24 (anomaly) in full pool")
        print(f"  {'id':>3} {'name':<20} {'instances':>9}")
        for cid in sorted(set(sub_inst) | set(PCU)):
            print(f"  {cid:>3} {name(cid):<20} {sub_inst.get(cid, 0):>9}")

    (SUBSET / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsubset -> {SUBSET}\nreport -> {SUBSET / 'build_report.json'}")


def cmd_config(_args) -> None:
    missing = [s for s in ("train", "val", "test") if not (SUBSET / s / "images").is_dir()]
    if missing:
        raise SystemExit(f"build the subset first (missing: {', '.join(missing)})")

    max_id = -1
    for split in SPLITS:
        for s in scan(split):
            if s["classes"]:
                max_id = max(max_id, max(s["classes"]))
    nc = max_id + 1
    names = [name(i) for i in range(nc)]

    lines = [
        f"path: {SUBSET.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {nc}",
        "names:",
    ]
    for i, nm in enumerate(names):
        lines.append(f"  {i}: {nm}")
    CONFIG_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"nc={nc} -> {CONFIG_YAML}")
    if nc > 24:
        print("WARNING: ids >= 24 present (anomaly) - resolve via 'verify' before training")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="class/image statistics")

    v = sub.add_parser("verify", help="render images with hypothesized class names")
    v.add_argument("--n", type=int, default=40, help="number of images (default 40)")
    v.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("build", help="stratified subset -> data/subset")
    b.add_argument("--train", type=int, default=3000)
    b.add_argument("--val", type=int, default=400)
    b.add_argument("--test", type=int, default=700)
    b.add_argument("--gap", type=int, default=10,
                   help="min frame gap within a sequence to avoid near-duplicates (default 10)")

    sub.add_parser("config", help="generate configs/data.yaml from subset")

    args = ap.parse_args()
    {"stats": cmd_stats, "verify": cmd_verify, "build": cmd_build, "config": cmd_config}[args.cmd](args)


if __name__ == "__main__":
    main()
