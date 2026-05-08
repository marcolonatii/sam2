"""
Visualize DINOv3 patch features as videos.

For each video (subfolder) in --frames_dir, extracts DINOv3 patch tokens,
reduces them to 3 components via PCA, and saves a side-by-side video
(original | DINO features) as an MP4.

Usage:
    python tools/visualize_dino_features.py \
        --frames_dir /experiments/marcol01/frames \
        --output_dir /home/marcol01/sam2/outputs/dino_feature_videos

Options:
    --frames_dir   Directory containing per-video subfolders of JPEG frames.
    --output_dir   Where to save the output MP4 files.
    --dino_input_size  Size to resize frames before DINO (default: 518, divisible by 14 for DINOv2; use 1024 for dinov3-vitl16).
    --fps          Output video FPS (default: 10).
    --device       Torch device (default: cuda if available, else cpu).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
PATCH_SIZE = 16


def load_frames(video_dir):
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in exts
    )
    frames = []
    for f in files:
        img = Image.open(os.path.join(video_dir, f)).convert("RGB")
        frames.append((f, img))
    return frames


def extract_dino_features(frames, processor, model, dino_input_size, device):
    """Returns list of patch feature arrays, each shape [H_p, W_p, embed_dim]."""
    # Infer actual patch grid from model config (processor may resize to its own native size)
    native_img_size = model.config.image_size  # e.g. 224
    native_patch_size = model.config.patch_size  # e.g. 16
    n_patches_side = native_img_size // native_patch_size  # e.g. 14
    n_patches = n_patches_side ** 2

    all_features = []

    for _, img in frames:
        # Let the processor handle resizing to model's native resolution
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # last_hidden_state: [1, 1+n_registers+n_patches, embed_dim]
        tokens = outputs.last_hidden_state[0, -n_patches:, :]  # [N_patches, embed_dim]
        feat = tokens.float().cpu().numpy()  # [N_patches, embed_dim]
        feat = feat.reshape(n_patches_side, n_patches_side, -1)  # [H_p, W_p, embed_dim]
        all_features.append(feat)

    return all_features, n_patches_side


def features_to_rgb(all_features, n_patches_side):
    """Use PCA across all frames to map embed_dim → 3 channels, then scale to [0,255]."""
    embed_dim = all_features[0].shape[-1]
    # Stack all frames: [T * N_patches, embed_dim]
    stacked = np.stack(all_features, axis=0)  # [T, H_p, W_p, embed_dim]
    T = len(all_features)
    flat = stacked.reshape(T * n_patches_side * n_patches_side, embed_dim)

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(flat)  # [T*N_patches, 3]

    # Normalize to [0, 1] per component
    for i in range(3):
        pmin, pmax = pca_features[:, i].min(), pca_features[:, i].max()
        pca_features[:, i] = (pca_features[:, i] - pmin) / (pmax - pmin + 1e-8)

    pca_features = pca_features.reshape(T, n_patches_side, n_patches_side, 3)
    return pca_features  # values in [0, 1]


def make_video(video_name, frames, pca_features, n_patches_side, dino_input_size, fps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}_dino.mp4")

    # Determine output frame size from first original frame
    w0, h0 = frames[0][1].size
    out_h = max(h0, dino_input_size)
    dino_w = dino_input_size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w0 + dino_w, out_h))

    for i, (_, img) in enumerate(frames):
        # Left: original frame
        orig = np.array(img.resize((w0, h0)))
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        if orig_bgr.shape[0] != out_h:
            pad = np.zeros((out_h - orig_bgr.shape[0], w0, 3), dtype=np.uint8)
            orig_bgr = np.vstack([orig_bgr, pad])

        # Right: DINO PCA features upsampled to dino_input_size
        feat_rgb = (pca_features[i] * 255).astype(np.uint8)  # [H_p, W_p, 3]
        feat_bgr = cv2.cvtColor(feat_rgb, cv2.COLOR_RGB2BGR)
        feat_bgr = cv2.resize(feat_bgr, (dino_w, dino_input_size), interpolation=cv2.INTER_NEAREST)
        if feat_bgr.shape[0] != out_h:
            pad = np.zeros((out_h - feat_bgr.shape[0], dino_w, 3), dtype=np.uint8)
            feat_bgr = np.vstack([feat_bgr, pad])

        combined = np.hstack([orig_bgr, feat_bgr])
        writer.write(combined)

    writer.release()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features as videos")
    parser.add_argument("--frames_dir", type=str, default="/experiments/marcol01/frames")
    parser.add_argument("--output_dir", type=str, default="/home/marcol01/sam2/outputs/dino_feature_videos")
    parser.add_argument("--dino_input_size", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    assert args.dino_input_size % PATCH_SIZE == 0, \
        f"dino_input_size must be divisible by {PATCH_SIZE}"

    print(f"Loading DINOv3 model: {_DINOV3_MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(_DINOV3_MODEL_ID)
    model = AutoModel.from_pretrained(_DINOV3_MODEL_ID).to(args.device).eval()

    video_dirs = sorted(
        d for d in os.listdir(args.frames_dir)
        if os.path.isdir(os.path.join(args.frames_dir, d))
    )
    print(f"Found {len(video_dirs)} videos in {args.frames_dir}")

    for video_name in video_dirs:
        video_dir = os.path.join(args.frames_dir, video_name)
        print(f"\nProcessing: {video_name}")

        frames = load_frames(video_dir)
        if not frames:
            print("  No frames found, skipping.")
            continue

        all_features, n_patches_side = extract_dino_features(
            frames, processor, model, args.dino_input_size, args.device
        )
        pca_features = features_to_rgb(all_features, n_patches_side)
        make_video(video_name, frames, pca_features, n_patches_side,
                   args.dino_input_size, args.fps, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
