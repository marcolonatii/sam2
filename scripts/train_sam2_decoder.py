"""
Fine-tune the SAM2 mask decoder (and optionally the prompt encoder)
on top of a frozen SAM2 backbone — no DINO fusion.

Training path (propagation path, no-memory):
    frozen SAM2 backbone → _prepare_backbone_features
    → no-memory pix_feat (raw vision features, finest level)
    → sam_mask_decoder (with a GT-sampled point prompt)   ← weights trained here

Prompt: one positive click sampled uniformly from the GT mask, per image.

Datasets:
    - MoCA video frames    : --moca_frames / --moca_masks
    - COD10K static images : --cod10k_root
    - CAMO  static images  : --camo_root

By default only sam_mask_decoder is trained.
Pass --train_prompt_encoder to also train sam_prompt_encoder.
Everything else (backbone, memory modules) is frozen.

Usage:
    # run from /home/marcol01/sam2/
    python scripts/train_sam2_decoder.py \\
        --config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --checkpoint ./sam2.1_hiera_large.pt \\
        --datasets moca \\
        --output_dir ./checkpoints_sam2_decoder_finetune
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# SAM2 image preprocessing constants
# ---------------------------------------------------------------------------

SAM2_IMAGE_SIZE = 1024
SAM2_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
SAM2_PIXEL_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_image(img_path: str, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (t - SAM2_PIXEL_MEAN) / SAM2_PIXEL_STD


def _load_mask(mask_path: str, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    m = Image.open(mask_path).convert("L")
    m = m.resize((size, size), Image.NEAREST)
    t = torch.from_numpy(np.array(m)).float() / 255.0
    return (t > 0.5).float().unsqueeze(0)  # [1, H, W] binary float


# ---------------------------------------------------------------------------
# Datasets  (identical to train_dino_fusion.py)
# ---------------------------------------------------------------------------

class MoCADataset(Dataset):
    """Per-frame dataset from MoCA; only annotated frames are used."""

    def __init__(self, frames_root: str, masks_root: str):
        super().__init__()
        self.samples = []
        for seq in sorted(os.listdir(masks_root)):
            mask_dir  = os.path.join(masks_root,  seq)
            frame_dir = os.path.join(frames_root, seq)
            if not os.path.isdir(mask_dir) or not os.path.isdir(frame_dir):
                continue
            for mask_file in sorted(os.listdir(mask_dir)):
                mask_path  = os.path.join(mask_dir, mask_file)
                stem       = os.path.splitext(mask_file)[0]
                frame_path = os.path.join(frame_dir, stem + ".jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(frame_dir, stem + ".png")
                if os.path.exists(frame_path):
                    self.samples.append((frame_path, mask_path))
        print(f"[MoCA]   {len(self.samples)} image-mask pairs from {frames_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path = self.samples[idx]
        return _load_image(frame_path), _load_mask(mask_path)


class COD10KDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        img_dir = os.path.join(root, "Train", "Image")
        gt_dir  = os.path.join(root, "Train", "GT_Object")
        self.samples = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".jpg"):
                continue
            stem    = os.path.splitext(fn)[0]
            gt_path = os.path.join(gt_dir, stem + ".png")
            if os.path.exists(gt_path):
                self.samples.append((os.path.join(img_dir, fn), gt_path))
        print(f"[COD10K] {len(self.samples)} image-mask pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return _load_image(img_path), _load_mask(mask_path)


class CAMODataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        img_dir = os.path.join(root, "Images", "Train")
        gt_dir  = os.path.join(root, "GT")
        self.samples = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".jpg"):
                continue
            stem    = os.path.splitext(fn)[0]
            gt_path = os.path.join(gt_dir, stem + ".png")
            if os.path.exists(gt_path):
                self.samples.append((os.path.join(img_dir, fn), gt_path))
        print(f"[CAMO]   {len(self.samples)} image-mask pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return _load_image(img_path), _load_mask(mask_path)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred   = torch.sigmoid(pred_logits).flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    union  = pred.sum(1) + target.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor):
    return F.binary_cross_entropy_with_logits(pred_logits, target) + dice_loss(pred_logits, target)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args):
    from sam2.build_sam import build_sam2
    return build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device="cpu",   # moved to GPU after freezing
        mode="train",
        apply_postprocessing=False,
    )


def freeze_and_collect_trainable(model, train_prompt_encoder: bool = False):
    """Freeze everything; unfreeze sam_mask_decoder (and optionally sam_prompt_encoder)."""
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    trainable_names  = []

    keywords = ["sam_mask_decoder"]
    if train_prompt_encoder:
        keywords.append("sam_prompt_encoder")

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True
            trainable_params.append(param)
            trainable_names.append(name)

    print(f"\nTrainable parameters ({len(trainable_names)}):")
    total_trainable = 0
    for name in trainable_names:
        p = dict(model.named_parameters())[name]
        total_trainable += p.numel()
        print(f"  {name}: {list(p.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total trainable: {total_trainable:,} / {total:,} ({total_trainable / total:.4%})\n")
    return trainable_params


# ---------------------------------------------------------------------------
# Forward pass  (propagation path, no-memory, no DINO)
# ---------------------------------------------------------------------------

def forward_single_frame(model, images, masks):
    """Backbone (frozen) → no-memory pix_feat → SAM decoder (trainable).

    Args:
        model:  SAM2 model.
        images: [B, 3, H, W] SAM2-normalized tensor (H=W=1024).
        masks:  [B, 1, H, W] binary float GT masks (H=W=1024).

    Returns:
        high_res_masks: [B, 1, 1024, 1024] mask logits.
    """
    from sam2.modeling.sam2_utils import get_next_point

    B = images.shape[0]

    # 1. Frozen backbone
    with torch.no_grad():
        backbone_out_raw = model.forward_image(images)
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out_raw)

    C = model.hidden_dim
    H, W = feat_sizes[-1]

    # 2. No-memory pix_feat: raw top-level features [B, C, H, W]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

    # 3. High-res features for the decoder
    high_res_features = [
        x.permute(1, 2, 0).view(B, x.shape[2], *s)
        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
    ]

    # 4. One GT positive click per image as the prompt
    gt_bool = masks.bool()
    points, labels = get_next_point(gt_masks=gt_bool, pred_masks=None, method="uniform")
    point_inputs = {
        "point_coords": points.to(images.device),
        "point_labels": labels.to(images.device),
    }

    # 5. SAM decoder
    _, _, _, _, high_res_masks, _, _ = model._forward_sam_heads(
        backbone_features=pix_feat,
        point_inputs=point_inputs,
        mask_inputs=None,
        high_res_features=high_res_features,
        multimask_output=False,
    )
    return high_res_masks


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, args):
    model.train()
    # Keep frozen submodules in eval mode
    model.image_encoder.eval()
    model.memory_attention.eval()
    model.memory_encoder.eval()
    if not args.train_prompt_encoder:
        model.sam_prompt_encoder.eval()

    epoch_loss = 0.0
    n_batches  = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_logits = forward_single_frame(model, images, masks)
            loss = combined_loss(pred_logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

        if batch_idx % args.log_every == 0:
            print(f"  [Epoch {epoch+1}] Batch {batch_idx}/{len(dataloader)}  Loss: {loss.item():.4f}")

    return epoch_loss / max(n_batches, 1)


def validate_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_logits = forward_single_frame(model, images, masks)
                loss = combined_loss(pred_logits, masks)
            epoch_loss += loss.item()
            n_batches  += 1

    return epoch_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 mask decoder (no DINO)")
    parser.add_argument("--config",       type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint",   type=str,
                        default="./sam2.1_hiera_large.pt")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--save_every",   type=int,   default=5)
    parser.add_argument("--val_split",    type=float, default=0.15)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--train_prompt_encoder", action="store_true",
                        help="Also fine-tune sam_prompt_encoder (default: frozen).")
    parser.add_argument("--output_dir",   type=str,
                        default="/home/marcol01/sam2/checkpoints_sam2_decoder_finetune")
    # Dataset selection
    parser.add_argument("--datasets", nargs="+",
                        choices=["moca", "cod10k", "camo"],
                        default=["moca"])
    parser.add_argument("--moca_frames", type=str, default="/Experiments/marcol01/frames_train")
    parser.add_argument("--moca_masks",  type=str, default="/Experiments/marcol01/masks_train")
    parser.add_argument("--cod10k_root", type=str, default="/Experiments/marcol01/COD10K-v3")
    parser.add_argument("--camo_root",   type=str, default="/Experiments/marcol01/CAMO")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building SAM2 model (no DINO fusion)...")
    model            = build_model(args)
    trainable_params = freeze_and_collect_trainable(model, args.train_prompt_encoder)
    model            = model.to(device)

    print(f"\nLoading datasets (selected: {', '.join(args.datasets)})...")
    datasets = []
    if "moca" in args.datasets:
        if os.path.isdir(args.moca_frames) and os.path.isdir(args.moca_masks):
            datasets.append(MoCADataset(args.moca_frames, args.moca_masks))
        else:
            print(f"[MoCA]   skipped (paths not found)")
    else:
        print(f"[MoCA]   skipped (not selected)")

    if "cod10k" in args.datasets:
        if os.path.isdir(args.cod10k_root):
            datasets.append(COD10KDataset(args.cod10k_root))
        else:
            print(f"[COD10K] skipped (path not found: {args.cod10k_root})")
    else:
        print(f"[COD10K] skipped (not selected)")

    if "camo" in args.datasets:
        if os.path.isdir(args.camo_root):
            datasets.append(CAMODataset(args.camo_root))
        else:
            print(f"[CAMO]   skipped (path not found: {args.camo_root})")
    else:
        print(f"[CAMO]   skipped (not selected)")

    if not datasets:
        print("ERROR: no datasets found.")
        sys.exit(1)

    combined_ds = ConcatDataset(datasets)
    total   = len(combined_ds)
    n_val   = max(1, int(total * args.val_split))
    n_train = total - n_val
    train_ds, val_ds = random_split(
        combined_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nCombined: {total} samples  →  train: {n_train}  val: {n_val}")

    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  LR: {args.lr}  Weight decay: {args.weight_decay}  Batch size: {args.batch_size}")
    print(f"  Val split: {args.val_split}  Early stopping patience: {args.patience}")
    print(f"  Train prompt encoder: {args.train_prompt_encoder}")
    print(f"  Output dir: {args.output_dir}\n")

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_history     = []
    val_history       = []

    for epoch in range(args.epochs):
        t0       = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, epoch, args)
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step()
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"Train: {avg_loss:.4f}  Val: {val_loss:.4f} — "
            f"LR: {lr_now:.6f} — Time: {elapsed:.1f}s"
        )

        train_history.append(avg_loss)
        val_history.append(val_loss)
        epochs_x = range(1, len(train_history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, train_history, marker="o", markersize=3, linewidth=1.5, label="Train")
        ax.plot(epochs_x, val_history,   marker="s", markersize=3, linewidth=1.5, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("SAM2 Decoder Fine-tuning Loss")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=120)
        plt.close(fig)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"sam2_decoder_epoch{epoch+1:03d}.pt")
            torch.save({
                "epoch":             epoch + 1,
                "sam_mask_decoder":  model.sam_mask_decoder.state_dict(),
                "sam_prompt_encoder": model.sam_prompt_encoder.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "scheduler":         scheduler.state_dict(),
                "train_loss":        avg_loss,
                "val_loss":          val_loss,
                "config":            args.config,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(args.output_dir, "sam2_decoder_best.pt")
            torch.save({
                "epoch":             epoch + 1,
                "sam_mask_decoder":  model.sam_mask_decoder.state_dict(),
                "sam_prompt_encoder": model.sam_prompt_encoder.state_dict(),
                "train_loss":        avg_loss,
                "val_loss":          val_loss,
                "config":            args.config,
            }, best_path)
            print(f"  New best model (val={val_loss:.4f}) saved: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No val improvement for {epochs_no_improve}/{args.patience} epochs")
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
