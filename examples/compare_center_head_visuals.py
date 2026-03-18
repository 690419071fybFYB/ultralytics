#!/usr/bin/env python3
"""Compare baseline RT-DETR vs center-head RT-DETR and export error/recovery visualizations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import RTDETR


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=str,
        default="/home/fyb/datasets/DIOR_COCO_ULTRA/dior_coco_ultralytics.yaml",
        help="Dataset yaml path.",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default="/home/fyb/mydir/ultralytics/runs/detect/runs/train/rtdetr_dior5/weights/best.pt",
        help="Baseline model weights.",
    )
    p.add_argument(
        "--center",
        type=str,
        default="/home/fyb/mydir/ultralytics/runs/detect/runs/train/phase1_center_lam035_seed0_e40/weights/best.pt",
        help="Center-head model weights.",
    )
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"), help="Dataset split.")
    p.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for GT-pred matching.")
    p.add_argument(
        "--fp-correct-iou",
        type=float,
        default=0.5,
        help="IoU threshold to treat a baseline FP region as corrected by a center-model TP.",
    )
    p.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    p.add_argument("--device", type=str, default="0", help='Inference device, e.g. "0" or "cpu".')
    p.add_argument("--max-vis-errors", type=int, default=80, help="Max baseline error images to visualize.")
    p.add_argument("--max-vis-recovers", type=int, default=80, help="Max center recover images to visualize.")
    p.add_argument("--max-vis-quad", type=int, default=80, help="Max 2x3 FP-correction visuals.")
    p.add_argument(
        "--outdir",
        type=str,
        default="/home/fyb/mydir/ultralytics/runs/analysis/center_head_case_study",
        help="Output directory.",
    )
    return p.parse_args()


def load_data_cfg(data_yaml: Path, split: str) -> tuple[list[Path], Path, dict[int, str]]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"]).expanduser().resolve()
    split_rel = Path(cfg[split])
    img_dir = (root / split_rel).resolve()

    parts = list(split_rel.parts)
    parts = ["labels" if p == "images" else p for p in parts]
    label_dir = (root / Path(*parts)).resolve()

    names_raw = cfg.get("names", {})
    if isinstance(names_raw, list):
        names = {i: n for i, n in enumerate(names_raw)}
    else:
        names = {int(k): str(v) for k, v in names_raw.items()}

    images = sorted([p.resolve() for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    return images, label_dir, names


def yolo_xywhn_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = box.tolist()
    x1 = (x - bw / 2.0) * w
    y1 = (y - bh / 2.0) * h
    x2 = (x + bw / 2.0) * w
    y2 = (y + bh / 2.0) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def read_gt_boxes(img_path: Path, label_dir: Path, img_shape: tuple[int, int]) -> list[dict]:
    h, w = img_shape[:2]
    rel = img_path.stem + ".txt"
    label_path = label_dir / rel
    out = []
    if not label_path.exists():
        return out

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xywhn = np.array([float(v) for v in parts[1:]], dtype=np.float32)
        xyxy = yolo_xywhn_to_xyxy(xywhn, w, h)
        out.append({"cls": cls, "xyxy": xyxy})
    return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def match_preds_to_gts(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
    matched_pred, matched_gt = set(), set()
    pairs = []
    order = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)
    for pi in order:
        p = preds[pi]
        best_gi, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            if p["cls"] != g["cls"]:
                continue
            iou = iou_xyxy(p["xyxy"], g["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_pred.add(pi)
            matched_gt.add(best_gi)
            pairs.append((pi, best_gi, best_iou))
    return matched_pred, matched_gt, pairs


def split_unmatched_preds(
    preds: list[dict], gts: list[dict], matched_pred: set[int], iou_thr: float
) -> tuple[list[int], list[tuple[int, int, float]]]:
    pure_fp = []
    wrong_cls = []
    for pi, p in enumerate(preds):
        if pi in matched_pred:
            continue
        if not gts:
            pure_fp.append(pi)
            continue
        ious = [iou_xyxy(p["xyxy"], g["xyxy"]) for g in gts]
        gi = int(np.argmax(ious))
        biou = float(ious[gi])
        if biou >= iou_thr and p["cls"] != gts[gi]["cls"]:
            wrong_cls.append((pi, gi, biou))
        else:
            pure_fp.append(pi)
    return pure_fp, wrong_cls


def infer_all(model_path: Path, images_dir: Path, imgsz: int, conf: float, nms_iou: float, batch: int, device: str) -> dict[Path, list[dict]]:
    model = RTDETR(str(model_path))
    preds = {}
    for r in model.predict(
        source=str(images_dir),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=nms_iou,
        batch=batch,
        device=device,
        verbose=False,
    ):
        path = Path(r.path).resolve()
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cfs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                boxes.append({"cls": int(cls[i]), "conf": float(cfs[i]), "xyxy": xyxy[i].astype(np.float32)})
        preds[path] = boxes
    return preds


def put_box(img: np.ndarray, box: np.ndarray, color: tuple[int, int, int], text: str, thick: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv2.putText(
        img,
        text,
        (x1, max(10, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def find_fp_corrected_pairs(
    b_preds: list[dict], baseline_fp_idx: set[int], c_preds: list[dict], c_tp_idx: set[int], iou_thr: float
) -> list[tuple[int, int, float]]:
    """Match baseline FP boxes to center TP boxes by IoU to identify corrected FP regions."""
    pairs = []
    used_center = set()
    for bi in sorted(list(baseline_fp_idx)):
        best_c, best_iou = -1, 0.0
        b_box = b_preds[bi]["xyxy"]
        for cj in c_tp_idx:
            if cj in used_center:
                continue
            iou = iou_xyxy(b_box, c_preds[cj]["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_c = cj
        if best_c >= 0 and best_iou >= iou_thr:
            used_center.add(best_c)
            pairs.append((bi, best_c, best_iou))
    return pairs


def _draw_panel(img: np.ndarray, preds: list[dict], names: dict[int, str], color: tuple[int, int, int], title: str) -> np.ndarray:
    panel = img.copy()
    for p in preds:
        # No confidence text by request.
        put_box(panel, p["xyxy"], color, f"{names.get(p['cls'], p['cls'])}", 2)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return panel


def make_grid_fp_correct_vis(
    img_path: Path,
    gts: list[dict],
    b_preds: list[dict],
    c_preds: list[dict],
    baseline_fp_idx: set[int],
    center_fp_idx: set[int],
    corrected_pairs: list[tuple[int, int, float]],
    names: dict[int, str],
    out_path: Path,
) -> None:
    """Render a 2x3 panel:
    1 baseline predictions, 2 center predictions, 3 GT,
    4 baseline FP, 5 center FP, 6 baseline FP corrected by center.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return

    # Row 1
    p1 = _draw_panel(img, b_preds, names, (0, 0, 255), "1 Baseline Predictions")
    p2 = _draw_panel(img, c_preds, names, (255, 0, 0), "2 Center-Head Predictions")
    p3 = _draw_panel(img, gts, names, (0, 220, 0), "3 GT")

    # Row 2
    p4_preds = [b_preds[i] for i in sorted(list(baseline_fp_idx))]
    p4 = _draw_panel(img, p4_preds, names, (0, 165, 255), "4 Baseline FP")
    p5_preds = [c_preds[i] for i in sorted(list(center_fp_idx))]
    p5 = _draw_panel(img, p5_preds, names, (180, 0, 180), "5 Center-Head FP")

    # Baseline FP regions corrected by center: show corrected center boxes.
    corrected_center_idx = sorted({cj for _, cj, _ in corrected_pairs})
    p6_preds = [c_preds[j] for j in corrected_center_idx]
    p6 = _draw_panel(img, p6_preds, names, (255, 255, 0), "6 Baseline FP Corrected by Center")

    # 2x3 layout.
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    canvas = np.vstack([row1, row2])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def make_baseline_error_vis(
    img_path: Path,
    gts: list[dict],
    b_preds: list[dict],
    names: dict[int, str],
    pure_fp: list[int],
    wrong_cls: list[tuple[int, int, float]],
    fn: set[int],
    out_path: Path,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    canvas = img.copy()
    for gi, g in enumerate(gts):
        tag = "GT-MISS" if gi in fn else "GT"
        put_box(canvas, g["xyxy"], (0, 220, 0), f"{tag}:{names.get(g['cls'], g['cls'])}", 2)
    wrong_set = {pi for pi, _, _ in wrong_cls}
    for pi, p in enumerate(b_preds):
        if pi in wrong_set:
            col = (0, 165, 255)  # orange
            prefix = "B-WRONG"
        elif pi in pure_fp:
            col = (0, 0, 255)  # red
            prefix = "B-FP"
        else:
            col = (0, 0, 200)
            prefix = "B-TP"
        put_box(canvas, p["xyxy"], col, f"{prefix}:{names.get(p['cls'], p['cls'])} {p['conf']:.2f}", 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def make_recover_vis(
    img_path: Path,
    gts: list[dict],
    b_preds: list[dict],
    c_preds: list[dict],
    names: dict[int, str],
    recovered_gt: set[int],
    out_path: Path,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    left = img.copy()
    right = img.copy()

    for gi, g in enumerate(gts):
        base_color = (0, 220, 0)
        if gi in recovered_gt:
            put_box(left, g["xyxy"], (0, 255, 255), f"REC:{names.get(g['cls'], g['cls'])}", 3)
            put_box(right, g["xyxy"], (0, 255, 255), f"REC:{names.get(g['cls'], g['cls'])}", 3)
        else:
            put_box(left, g["xyxy"], base_color, f"GT:{names.get(g['cls'], g['cls'])}", 2)
            put_box(right, g["xyxy"], base_color, f"GT:{names.get(g['cls'], g['cls'])}", 2)

    for p in b_preds:
        put_box(left, p["xyxy"], (0, 0, 255), f"B:{names.get(p['cls'], p['cls'])} {p['conf']:.2f}", 2)
    for p in c_preds:
        put_box(right, p["xyxy"], (255, 0, 0), f"C:{names.get(p['cls'], p['cls'])} {p['conf']:.2f}", 2)

    cv2.putText(left, "Baseline", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "Center Head", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    canvas = np.hstack([left, right])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    baseline_w = Path(args.baseline).resolve()
    center_w = Path(args.center).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    images, label_dir, names = load_data_cfg(data_yaml, args.split)
    if not images:
        raise RuntimeError(f"No images found for split={args.split} in {data_yaml}")
    img_dir = images[0].parent

    print(f"[INFO] Split={args.split}, images={len(images)}, img_dir={img_dir}")
    print(f"[INFO] Baseline: {baseline_w}")
    print(f"[INFO] Center:   {center_w}")

    print("[INFO] Running baseline inference...")
    baseline_preds = infer_all(baseline_w, img_dir, args.imgsz, args.conf, args.nms_iou, args.batch, args.device)
    print("[INFO] Running center inference...")
    center_preds = infer_all(center_w, img_dir, args.imgsz, args.conf, args.nms_iou, args.batch, args.device)

    baseline_errors = []
    center_recovers = []

    vis_err_dir = outdir / "baseline_errors_vis"
    vis_rec_dir = outdir / "center_recovers_vis"
    vis_quad_dir = outdir / "grid_2x3_fp_correction_vis"
    rel_rows = []
    fp_correct_rows = []

    for ip in images:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gts = read_gt_boxes(ip, label_dir, img.shape)
        b_preds = baseline_preds.get(ip, [])
        c_preds = center_preds.get(ip, [])

        b_matched_pred, b_matched_gt, _ = match_preds_to_gts(b_preds, gts, args.match_iou)
        c_matched_pred, c_matched_gt, _ = match_preds_to_gts(c_preds, gts, args.match_iou)

        b_fn = set(range(len(gts))) - b_matched_gt
        c_fn = set(range(len(gts))) - c_matched_gt
        b_pure_fp, b_wrong = split_unmatched_preds(b_preds, gts, b_matched_pred, args.match_iou)
        c_pure_fp, c_wrong = split_unmatched_preds(c_preds, gts, c_matched_pred, args.match_iou)
        b_fp_set = set(b_pure_fp) | {pi for pi, _, _ in b_wrong}
        c_fp_set = set(c_pure_fp) | {pi for pi, _, _ in c_wrong}
        corrected_pairs = find_fp_corrected_pairs(
            b_preds=b_preds,
            baseline_fp_idx=b_fp_set,
            c_preds=c_preds,
            c_tp_idx=c_matched_pred,
            iou_thr=args.fp_correct_iou,
        )

        if b_fn or b_wrong or b_pure_fp:
            score = 2.0 * len(b_fn) + 1.5 * len(b_wrong) + 0.5 * len(b_pure_fp)
            baseline_errors.append(
                {
                    "image": str(ip),
                    "gt_count": len(gts),
                    "baseline_fn": len(b_fn),
                    "baseline_wrong_cls": len(b_wrong),
                    "baseline_fp": len(b_pure_fp),
                    "error_score": f"{score:.2f}",
                }
            )

        recovered = b_fn & c_matched_gt
        if recovered:
            center_recovers.append(
                {
                    "image": str(ip),
                    "gt_count": len(gts),
                    "recovered_gt_count": len(recovered),
                    "baseline_fn": len(b_fn),
                    "center_fn": len(c_fn),
                }
            )

        if corrected_pairs:
            fp_correct_rows.append(
                {
                    "image": str(ip),
                    "baseline_fp_count": len(b_fp_set),
                    "corrected_fp_count": len(corrected_pairs),
                    "baseline_fn": len(b_fn),
                    "center_fn": len(c_fn),
                }
            )

        rel_rows.append(
            {
                "image": str(ip),
                "gt_count": len(gts),
                "baseline_fn": len(b_fn),
                "center_fn": len(c_fn),
                "baseline_matched": len(b_matched_gt),
                "center_matched": len(c_matched_gt),
                "recovered": len(recovered),
            }
        )

    baseline_errors_sorted = sorted(baseline_errors, key=lambda x: float(x["error_score"]), reverse=True)
    center_recovers_sorted = sorted(center_recovers, key=lambda x: int(x["recovered_gt_count"]), reverse=True)
    fp_correct_sorted = sorted(fp_correct_rows, key=lambda x: int(x["corrected_fp_count"]), reverse=True)

    write_csv(outdir / "all_image_comparison.csv", rel_rows)
    write_csv(outdir / "baseline_errors_all.csv", baseline_errors_sorted)
    write_csv(outdir / "center_recovers_all.csv", center_recovers_sorted)
    write_csv(outdir / "fp_corrected_all.csv", fp_correct_sorted)

    print(f"[INFO] Baseline error images: {len(baseline_errors_sorted)}")
    print(f"[INFO] Center recovered images: {len(center_recovers_sorted)}")
    print(f"[INFO] Baseline FP corrected images: {len(fp_correct_sorted)}")

    err_top = baseline_errors_sorted[: args.max_vis_errors]
    rec_top = center_recovers_sorted[: args.max_vis_recovers]
    quad_top = fp_correct_sorted[: args.max_vis_quad]

    print(f"[INFO] Rendering baseline error visuals: {len(err_top)}")
    for i, row in enumerate(err_top, 1):
        ip = Path(row["image"])
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gts = read_gt_boxes(ip, label_dir, img.shape)
        b_preds = baseline_preds.get(ip, [])
        b_matched_pred, b_matched_gt, _ = match_preds_to_gts(b_preds, gts, args.match_iou)
        b_fn = set(range(len(gts))) - b_matched_gt
        b_pure_fp, b_wrong = split_unmatched_preds(b_preds, gts, b_matched_pred, args.match_iou)
        out = vis_err_dir / f"{i:03d}_{ip.stem}.jpg"
        make_baseline_error_vis(ip, gts, b_preds, names, b_pure_fp, b_wrong, b_fn, out)

    print(f"[INFO] Rendering center recover visuals: {len(rec_top)}")
    for i, row in enumerate(rec_top, 1):
        ip = Path(row["image"])
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gts = read_gt_boxes(ip, label_dir, img.shape)
        b_preds = baseline_preds.get(ip, [])
        c_preds = center_preds.get(ip, [])
        _, b_matched_gt, _ = match_preds_to_gts(b_preds, gts, args.match_iou)
        _, c_matched_gt, _ = match_preds_to_gts(c_preds, gts, args.match_iou)
        recovered = (set(range(len(gts))) - b_matched_gt) & c_matched_gt
        out = vis_rec_dir / f"{i:03d}_{ip.stem}.jpg"
        make_recover_vis(ip, gts, b_preds, c_preds, names, recovered, out)

    print(f"[INFO] Rendering 2x3 FP-correction visuals: {len(quad_top)}")
    for i, row in enumerate(quad_top, 1):
        ip = Path(row["image"])
        b_preds = baseline_preds.get(ip, [])
        c_preds = center_preds.get(ip, [])
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gts = read_gt_boxes(ip, label_dir, img.shape)
        b_matched_pred, _, _ = match_preds_to_gts(b_preds, gts, args.match_iou)
        c_matched_pred, _, _ = match_preds_to_gts(c_preds, gts, args.match_iou)
        b_pure_fp, b_wrong = split_unmatched_preds(b_preds, gts, b_matched_pred, args.match_iou)
        c_pure_fp, c_wrong = split_unmatched_preds(c_preds, gts, c_matched_pred, args.match_iou)
        b_fp_set = set(b_pure_fp) | {pi for pi, _, _ in b_wrong}
        c_fp_set = set(c_pure_fp) | {pi for pi, _, _ in c_wrong}
        corrected_pairs = find_fp_corrected_pairs(
            b_preds=b_preds,
            baseline_fp_idx=b_fp_set,
            c_preds=c_preds,
            c_tp_idx=c_matched_pred,
            iou_thr=args.fp_correct_iou,
        )
        out = vis_quad_dir / f"{i:03d}_{ip.stem}.jpg"
        make_grid_fp_correct_vis(ip, gts, b_preds, c_preds, b_fp_set, c_fp_set, corrected_pairs, names, out)

    summary = outdir / "README.md"
    summary.write_text(
        "\n".join(
            [
                "# Baseline vs Center Head Case Study",
                "",
                f"- split: `{args.split}`",
                f"- baseline: `{baseline_w}`",
                f"- center: `{center_w}`",
                f"- match_iou: `{args.match_iou}`",
                f"- conf: `{args.conf}`",
                "",
                f"- baseline error image count: **{len(baseline_errors_sorted)}**",
                f"- center recovered image count: **{len(center_recovers_sorted)}**",
                f"- baseline FP corrected image count: **{len(fp_correct_sorted)}**",
                "",
                "## Files",
                f"- all image stats: `{(outdir / 'all_image_comparison.csv').as_posix()}`",
                f"- baseline errors list: `{(outdir / 'baseline_errors_all.csv').as_posix()}`",
                f"- center recovers list: `{(outdir / 'center_recovers_all.csv').as_posix()}`",
                f"- fp corrected list: `{(outdir / 'fp_corrected_all.csv').as_posix()}`",
                f"- baseline error visuals: `{vis_err_dir.as_posix()}`",
                f"- center recover visuals: `{vis_rec_dir.as_posix()}`",
                f"- 2x3 fp-correction visuals: `{vis_quad_dir.as_posix()}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[DONE] Output directory: {outdir}")


if __name__ == "__main__":
    main()
