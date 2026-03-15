# yolo/auto_label.py
# -*- coding: utf-8 -*-
"""
best.pt を使って unlabeled を自動ラベルするスクリプト

やること:
- yolo/dataset/images/unlabeled/ を読む
- best.pt で推論
- 高信頼画像は train にコピーし、labels/train に YOLO txt を保存
- 低信頼 or 未検出画像は auto_review にコピー
- 元の unlabeled は消さない（copy运用）

实行例:
    python yolo/auto_label.py
    python yolo/auto_label.py --conf 0.70 --avg-conf 0.80
    python yolo/auto_label.py --model D:\\vrc-auto-fish-stable\\yolo\\runs\\fish_detect\\weights\\best.pt
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_NAMES = ["fish", "bar", "track", "progress", "hook"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_preview(
    img,
    boxes,
    out_path: Path,
):
    preview = img.copy()
    colors = {
        0: (0, 255, 0),      # fish
        1: (255, 255, 255),  # bar
        2: (0, 180, 255),    # track
        3: (0, 220, 180),    # progress
        4: (0, 0, 255),      # hook
    }

    if boxes is not None:
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())

            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue

            x1, y1, x2, y2 = xyxy
            color = colors.get(cls_id, (128, 128, 128))
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview,
                f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_path), preview)


def main():
    parser = argparse.ArgumentParser(description="Auto label unlabeled images with best.pt")
    parser.add_argument("--model", type=str, default=str(Path(config.YOLO_MODEL)),
                        help="teacher model path (default: config.YOLO_MODEL)")
    parser.add_argument("--conf", type=float, default=0.72,
                        help="per-box confidence threshold")
    parser.add_argument("--avg-conf", type=float, default=0.80,
                        help="per-image average confidence threshold")
    parser.add_argument("--min-boxes", type=int, default=2,
                        help="minimum number of boxes required to auto-accept")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="predict image size")
    parser.add_argument("--max-det", type=int, default=20,
                        help="max detections per image")
    parser.add_argument("--require-fish", action="store_true",
                        help="accept only if fish class exists")
    parser.add_argument("--require-bar", action="store_true",
                        help="accept only if bar class exists")
    parser.add_argument("--require-track", action="store_true",
                        help="accept only if track class exists")
    parser.add_argument("--require-progress", action="store_true",
                        help="accept only if progress class exists")
    parser.add_argument("--require-hook", action="store_true",
                        help="accept only if hook class exists")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics が未インストールです")
        print("pip install ultralytics")
        return

    import torch

    base_dir = Path(config.BASE_DIR)
    dataset_dir = base_dir / "yolo" / "dataset"

    unlabeled_dir = dataset_dir / "images" / "unlabeled"
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"

    review_img_dir = dataset_dir / "images" / "auto_review"
    review_preview_dir = dataset_dir / "images" / "auto_review_preview"

    ensure_dir(train_img_dir)
    ensure_dir(train_lbl_dir)
    ensure_dir(review_img_dir)
    ensure_dir(review_preview_dir)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] model not found: {model_path}")
        return

    images = list_images(unlabeled_dir)
    if not images:
        print(f"[WARN] unlabeled に画像がありません: {unlabeled_dir}")
        return

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Auto labeling start")
    print("=" * 60)
    print(f"model      : {model_path}")
    print(f"unlabeled  : {unlabeled_dir}")
    print(f"train img  : {train_img_dir}")
    print(f"train lbl  : {train_lbl_dir}")
    print(f"review dir : {review_img_dir}")
    print(f"conf       : {args.conf}")
    print(f"avg-conf   : {args.avg_conf}")
    print(f"imgsz      : {args.imgsz}")
    print(f"device     : {'GPU' if device == 0 else 'CPU'}")
    print()

    accepted = 0
    reviewed = 0
    skipped = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]

        result = model.predict(
            source=img,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            max_det=args.max_det,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, None, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        lines = []
        confs = []
        found_classes = set()

        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())

            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue

            x1, y1, x2, y2 = xyxy
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            lines.append(yolo_line(cls_id, x1, y1, x2, y2, w, h))
            confs.append(conf)
            found_classes.add(cls_id)

        if len(lines) < args.min_boxes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        avg_conf = sum(confs) / max(1, len(confs))
        if avg_conf < args.avg_conf:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        if args.require_fish and 0 not in found_classes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        if args.require_bar and 1 not in found_classes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        if args.require_track and 2 not in found_classes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        if args.require_progress and 3 not in found_classes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        if args.require_hook and 4 not in found_classes:
            shutil.copy2(img_path, review_img_dir / img_path.name)
            save_preview(img, boxes, review_preview_dir / img_path.name)
            reviewed += 1
            continue

        out_img = train_img_dir / img_path.name
        out_lbl = train_lbl_dir / f"{img_path.stem}.txt"

        shutil.copy2(img_path, out_img)
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        accepted += 1

    print()
    print("=" * 60)
    print("Auto labeling finished")
    print("=" * 60)
    print(f"accepted: {accepted}")
    print(f"review  : {reviewed}")
    print(f"skipped : {skipped}")
    print()
    print("次のおすすめ:")
    print("1) auto_review_preview を見て怪しい画像だけ確認")
    print("2) 必要なら label.py --relabel で修正")
    print("3) train.py で再学習")


if __name__ == "__main__":
    main()