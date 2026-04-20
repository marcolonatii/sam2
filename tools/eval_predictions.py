"""
Evaluate predicted masks against GT masks (only annotated frames are used).

Computes per-video and overall:
    - J  (Jaccard / IoU)
    - F  (boundary F-measure)
    - Dice
    - J&F (mean of J and F)

Only frames that have a corresponding GT mask are evaluated.

Usage:
    python tools/eval_predictions.py \
        --pred_dir  /home/marcol01/sam2/masks_sam2_dino_fusion \
        --gt_dir    /Experiments/marcol01/masks \
        --output_dir /home/marcol01/sam2/eval_dino_fusion
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _binary(arr: np.ndarray) -> np.ndarray:
    """Convert any mask image to a boolean array."""
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """Jaccard index (IoU) between two boolean arrays."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter) / float(denom)


def _boundary(mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    """Thin boundary via morphological erosion difference."""
    from scipy.ndimage import binary_erosion
    struct = np.ones((dilation, dilation), dtype=bool)
    eroded = binary_erosion(mask, structure=struct)
    return np.logical_xor(mask, eroded)


def f_boundary(pred: np.ndarray, gt: np.ndarray, dilation: int = 3) -> float:
    """Boundary F-measure."""
    pred_b = _boundary(pred, dilation)
    gt_b   = _boundary(gt,   dilation)

    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------

def evaluate_video(pred_dir: str, gt_dir: str, video: str):
    """Evaluate one video. Only GT-annotated frames are used."""
    gt_video_dir   = os.path.join(gt_dir,   video)
    pred_video_dir = os.path.join(pred_dir, video)

    # All GT frames for this video (skip non-png / hidden files)
    gt_frames = sorted(
        f for f in os.listdir(gt_video_dir)
        if f.lower().endswith(".png") and not f.startswith(".")
    )

    n_gt_frames  = len(gt_frames)
    n_pred_total = len([
        f for f in os.listdir(pred_video_dir)
        if f.lower().endswith(".png")
    ]) if os.path.isdir(pred_video_dir) else 0

    per_frame   = []
    js, fs, ds = [], [], []
    n_detected  = 0

    for fname in gt_frames:
        gt_path   = os.path.join(gt_video_dir,   fname)
        pred_path = os.path.join(pred_video_dir, fname)

        if not os.path.exists(pred_path):
            # Prediction missing for this GT frame — skip
            continue

        gt_arr   = _binary(np.array(Image.open(gt_path).convert("L")))
        pred_arr = _binary(np.array(Image.open(pred_path).convert("L")))

        # Resize pred to GT size if needed
        if gt_arr.shape != pred_arr.shape:
            pred_img = Image.open(pred_path).convert("L").resize(
                (gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST
            )
            pred_arr = _binary(np.array(pred_img))

        j = jaccard(pred_arr, gt_arr)
        f = f_boundary(pred_arr, gt_arr)
        d = dice(pred_arr, gt_arr)

        js.append(j); fs.append(f); ds.append(d)
        n_detected += 1

        per_frame.append({
            "frame":  fname,
            "J":      round(j, 6),
            "F":      round(f, 6),
            "Dice":   round(d, 6),
            "J&F":    round((j + f) / 2, 6),
        })

    n_evaluated = len(js)
    mean_j  = float(np.mean(js))  if js else 0.0
    mean_f  = float(np.mean(fs))  if fs else 0.0
    mean_d  = float(np.mean(ds))  if ds else 0.0
    mean_jf = (mean_j + mean_f) / 2

    return {
        "video":       video,
        "n_frames":    n_pred_total,
        "n_gt_frames": n_gt_frames,
        "n_evaluated": n_evaluated,
        "n_detected":  n_detected,
        "mean_J":      mean_j,
        "mean_F":      mean_f,
        "mean_Dice":   mean_d,
        "mean_J&F":    mean_jf,
        "per_frame":   per_frame,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against GT masks")
    parser.add_argument("--pred_dir",   required=True,
                        help="Root folder with per-video prediction subdirs")
    parser.add_argument("--gt_dir",     required=True,
                        help="Root folder with per-video GT mask subdirs")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save results.json and per_video_metrics.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover videos: folders present in both pred and gt
    pred_videos = {
        v for v in os.listdir(args.pred_dir)
        if os.path.isdir(os.path.join(args.pred_dir, v))
    }
    gt_videos = {
        v for v in os.listdir(args.gt_dir)
        if os.path.isdir(os.path.join(args.gt_dir, v))
    }
    videos = sorted(pred_videos & gt_videos)

    if not videos:
        print("No matching video folders found between pred_dir and gt_dir.")
        sys.exit(1)

    print(f"Evaluating {len(videos)} videos...")

    per_video_results = []
    all_j, all_f, all_d = [], [], []

    for video in videos:
        result = evaluate_video(args.pred_dir, args.gt_dir, video)
        per_video_results.append(result)
        if result["n_evaluated"] > 0:
            all_j.append(result["mean_J"])
            all_f.append(result["mean_F"])
            all_d.append(result["mean_Dice"])
        print(
            f"  {video:<30}  J={result['mean_J']:.4f}  "
            f"F={result['mean_F']:.4f}  Dice={result['mean_Dice']:.4f}  "
            f"J&F={result['mean_J&F']:.4f}  "
            f"({result['n_evaluated']}/{result['n_gt_frames']} GT frames evaluated)"
        )

    overall_j  = float(np.mean(all_j))  if all_j else 0.0
    overall_f  = float(np.mean(all_f))  if all_f else 0.0
    overall_d  = float(np.mean(all_d))  if all_d else 0.0
    overall_jf = (overall_j + overall_f) / 2

    print(f"\nOverall ({len(videos)} videos):")
    print(f"  J={overall_j:.4f}  F={overall_f:.4f}  Dice={overall_d:.4f}  J&F={overall_jf:.4f}")

    # --- JSON ---
    output = {
        "config": {
            "pred_dir": args.pred_dir,
            "gt_dir":   args.gt_dir,
        },
        "overall": {
            "J":        overall_j,
            "F":        overall_f,
            "J&F":      overall_jf,
            "Dice":     overall_d,
            "n_videos": len(videos),
        },
        "per_video": per_video_results,
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- CSV ---
    csv_path = os.path.join(args.output_dir, "per_video_metrics.csv")
    csv_cols  = ["video", "n_frames", "n_gt_frames", "n_evaluated",
                 "n_detected", "mean_J", "mean_F", "mean_Dice", "mean_J&F"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in per_video_results:
            writer.writerow({k: r[k] for k in csv_cols})
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
