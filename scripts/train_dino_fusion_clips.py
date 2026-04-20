"""
Train the DINOv3 fusion components on short MoCA clips (with memory propagation).

Training exposes the DINO cross-attention fuser to memory-conditioned features,
matching the inference-time distribution for frames 2+ in a video.

Clip structure (default clip_len=5):
  Frame 0 → GT mask prompt → memory encoded
            (DINO skipped: model takes mask-as-output path when
             use_mask_input_as_output_without_sam=True)
  Frame 1 → memory-conditioned pix_feat → DINO fusion → loss
  Frame 2 → memory-conditioned pix_feat → DINO fusion → loss
  ...
  Frame T-1 → memory-conditioned pix_feat → DINO fusion → loss

Loss is averaged over all T frames (frame-0 loss is ~0 since prediction ≈ GT).

Only dino_encoder.proj and cross_attn_fuser are trained (+ optionally
sam_mask_decoder when --decoder_lr is set).

Usage (from /home/marcol01/sam2/):
    python scripts/train_dino_fusion_clips.py \\
        --config   configs/sam2.1/sam2.1_hiera_l.yaml \\
        --checkpoint ./sam2.1_hiera_large.pt \\
        --init_weights ./checkpoints_dino_fusion_pretrain_simple/dino_fusion_best.pt \\
        --moca_frames /Experiments/marcol01/frames_train \\
        --moca_masks  /Experiments/marcol01/masks_train \\
        --output_dir  ./checkpoints_dino_fusion_clips
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
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Constants (identical to train_dino_fusion.py)
# ---------------------------------------------------------------------------
SAM2_IMAGE_SIZE = 1024
SAM2_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
SAM2_PIXEL_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(
        (SAM2_IMAGE_SIZE, SAM2_IMAGE_SIZE), Image.BILINEAR
    )
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (t - SAM2_PIXEL_MEAN) / SAM2_PIXEL_STD


def _load_mask(path: str) -> torch.Tensor:
    m = Image.open(path).convert("L").resize(
        (SAM2_IMAGE_SIZE, SAM2_IMAGE_SIZE), Image.NEAREST
    )
    t = torch.from_numpy(np.array(m)).float() / 255.0
    return (t > 0.5).float().unsqueeze(0)  # [1, H, W] binary float


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MoCAClipDataset(Dataset):
    """Sliding-window clips of `clip_len` consecutive annotated MoCA frames.

    Only annotated frames (those with a GT mask file) are used.  Clips are
    sampled with a sliding window of step `clip_stride` over the sorted list
    of annotated frames within each sequence.

    Returns:
        images : [T, 3, H, W]  SAM2-normalised float32
        masks  : [T, 1, H, W]  binary float32
    """

    def __init__(
        self,
        frames_root: str,
        masks_root:  str,
        clip_len:    int = 5,
        clip_stride: int = 1,
    ):
        super().__init__()
        self.clips: list[list[tuple[str, str]]] = []

        for seq in sorted(os.listdir(masks_root)):
            # Support both flat layout  (masks_root/<seq>/<frame>.png)
            # and MoCA-Mask-Pseudo layout (masks_root/<seq>/GT/<frame>.png)
            candidate_gt = os.path.join(masks_root, seq, "GT")
            if os.path.isdir(candidate_gt):
                mask_dir = candidate_gt
            else:
                mask_dir = os.path.join(masks_root, seq)
            frame_dir = os.path.join(frames_root, seq)
            if not os.path.isdir(mask_dir) or not os.path.isdir(frame_dir):
                continue

            pairs: list[tuple[str, str]] = []
            for mf in sorted(os.listdir(mask_dir)):
                stem = os.path.splitext(mf)[0]
                fp   = os.path.join(frame_dir, stem + ".jpg")
                if not os.path.exists(fp):
                    fp = os.path.join(frame_dir, stem + ".png")
                if os.path.exists(fp):
                    pairs.append((fp, os.path.join(mask_dir, mf)))

            for i in range(0, len(pairs) - clip_len + 1, clip_stride):
                self.clips.append(pairs[i : i + clip_len])

        n_seq = len({os.path.dirname(c[0][0]) for c in self.clips}) if self.clips else 0
        print(
            f"[MoCAClips] clip_len={clip_len}, stride={clip_stride}: "
            f"{len(self.clips)} clips from {n_seq} sequences  ({frames_root})"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip   = self.clips[idx]
        images = torch.stack([_load_image(fp) for fp, _  in clip])  # [T, 3, H, W]
        masks  = torch.stack([_load_mask(mp)  for _,  mp in clip])  # [T, 1, H, W]
        return images, masks


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred   = torch.sigmoid(pred_logits).flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    union  = pred.sum(1) + target.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_logits, target) + \
           dice_loss(pred_logits, target)


# ---------------------------------------------------------------------------
# Model builder / freeze helper  (identical to train_dino_fusion.py)
# ---------------------------------------------------------------------------

def build_model(args):
    from sam2.build_sam import build_sam2_with_dino_fusion
    return build_sam2_with_dino_fusion(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device="cpu",
        mode="train",
        dino_model_name=args.dino_model,
        dino_out_dim=256,
        dino_freeze_backbone=True,
        dino_input_size=SAM2_IMAGE_SIZE,
        cross_attn_num_heads=8,
        strict_checkpoint_loading=False,
    )


def freeze_and_collect_trainable(model, fusion_lr: float, decoder_lr=None):
    for p in model.parameters():
        p.requires_grad = False

    def _collect(keywords):
        params, names = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in keywords):
                p.requires_grad = True
                params.append(p)
                names.append(name)
        return params, names

    fusion_params,  fusion_names  = _collect(["dino_encoder.proj", "cross_attn_fuser"])
    decoder_params, decoder_names = _collect(["sam_mask_decoder"]) if decoder_lr else ([], [])

    total       = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nFusion trainable ({len(fusion_names)} tensors):")
    for n in fusion_names:
        p = dict(model.named_parameters())[n]
        print(f"  {n}: {list(p.shape)}")
    if decoder_params:
        print(f"\nDecoder trainable ({len(decoder_names)} tensors):")
        for n in decoder_names:
            p = dict(model.named_parameters())[n]
            print(f"  {n}: {list(p.shape)}")
    print(f"Total trainable: {n_trainable:,} / {total:,} ({n_trainable / total:.4%})\n")

    groups = [{"params": fusion_params, "lr": fusion_lr}]
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr})
    return groups


# ---------------------------------------------------------------------------
# Clip-level forward pass
# ---------------------------------------------------------------------------

def forward_clip(
    model,
    images: torch.Tensor,   # [B, T, 3, H, W]
    masks:  torch.Tensor,   # [B, T, 1, H, W]
    device: torch.device,
) -> torch.Tensor:
    """Run B clips of T frames through SAM2 with online memory propagation.

    Frame 0 of each clip:
      - GT mask supplied as mask_input.
      - If use_mask_input_as_output_without_sam=True (default for large models),
        the prediction is the GT mask itself → loss ≈ 0, memory is encoded.
      - DINO fusion is skipped on this frame (mask-as-output bypasses it).

    Frames 1..T-1:
      - Memory from frame 0+ conditions the backbone features.
      - DINO cross-attention runs on memory-conditioned features.
      - Loss computed against GT masks.

    Returns:
        Scalar loss averaged over T frames (and over the batch via combined_loss).
    """
    B, T = images.shape[:2]

    # Fresh memory bank for this clip
    output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    total_loss: torch.Tensor | None = None

    for t in range(T):
        frame   = images[:, t]   # [B, 3, H, W]
        gt_mask = masks[:, t]    # [B, 1, H, W]

        # --- Backbone (frozen, no gradients needed) ---
        with torch.no_grad():
            backbone_out = model.forward_image(frame)
            _, vision_feats, vision_pos_embeds, feat_sizes = \
                model._prepare_backbone_features(backbone_out)

        is_init = (t == 0)

        # --- SAM2 track step:
        #   • _prepare_memory_conditioned_features (memory transformer, frozen)
        #   • DINO cross-attention fusion           (trainable)
        #   • _forward_sam_heads                   (decoder, frozen or trainable)
        #   • _encode_memory_in_output             (memory encoder, frozen)
        current_out = model.track_step(
            frame_idx=t,
            is_init_cond_frame=is_init,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=gt_mask if is_init else None,
            output_dict=output_dict,
            num_frames=T,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
            image=frame,   # activates DINO fusion inside _track_step
        )

        # --- Store so subsequent frames can read this frame's memory ---
        if is_init:
            output_dict["cond_frame_outputs"][t] = current_out
        else:
            output_dict["non_cond_frame_outputs"][t] = current_out

        # --- Loss: skip frame 0 (prediction ≈ GT due to mask-as-output) ---
        if not is_init:
            loss_t = combined_loss(current_out["pred_masks_high_res"], gt_mask)
            total_loss = loss_t if total_loss is None else total_loss + loss_t

    return total_loss / (T - 1)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, args):
    model.train()
    # Keep frozen modules in eval mode so BN/Dropout stats are stable
    model.image_encoder.eval()
    model.memory_attention.eval()
    model.memory_encoder.eval()
    if args.decoder_lr is None:
        model.sam_mask_decoder.eval()
    model.sam_prompt_encoder.eval()
    if model.dino_encoder is not None:
        model.dino_encoder.backbone.eval()

    epoch_loss, n = 0.0, 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = forward_clip(model, images, masks, device)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n += 1
        if batch_idx % args.log_every == 0:
            print(
                f"  [Epoch {epoch+1}] Batch {batch_idx}/{len(dataloader)}  "
                f"Loss: {loss.item():.4f}"
            )

    return epoch_loss / max(n, 1)


def validate_epoch(model, dataloader, device):
    model.eval()
    epoch_loss, n = 0.0, 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = forward_clip(model, images, masks, device)
            epoch_loss += loss.item()
            n += 1
    return epoch_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Fusion parameter monitor
# ---------------------------------------------------------------------------

def _print_fusion_params(model):
    fuser = model.cross_attn_fuser
    blocks = fuser.blocks if hasattr(fuser, "blocks") else [fuser]
    for i, block in enumerate(blocks):
        alpha = block.alpha.item()
        gate_bias = block.gate[-1].bias
        gate_open = torch.sigmoid(gate_bias).mean().item()
        print(
            f"  [fusion block {i}] alpha={alpha:.4f}  "
            f"gate_bias_mean={gate_bias.mean().item():.4f}  "
            f"gate_open={gate_open:.4f}"
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _make_ckpt(model, epoch, tr_loss, val_loss, args, include_optim=False,
               optimizer=None, scheduler=None):
    ckpt = {
        "epoch":             epoch,
        "dino_encoder_proj": model.dino_encoder.proj.state_dict(),
        "cross_attn_fuser":  model.cross_attn_fuser.state_dict(),
        "train_loss":        tr_loss,
        "val_loss":          val_loss,
        "config":            args.config,
    }
    if args.decoder_lr is not None:
        ckpt["sam_mask_decoder"] = model.sam_mask_decoder.state_dict()
    if include_optim and optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
        ckpt["scheduler"] = scheduler.state_dict()
    return ckpt


def _save_loss_curve(train_hist, val_hist, output_dir):
    xs = range(1, len(train_hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, train_hist, marker="o", markersize=3, linewidth=1.5, label="Train")
    ax.plot(xs, val_hist,   marker="s", markersize=3, linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (avg over clip frames)")
    ax.set_title("DINOv3 Fusion Clip Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 fusion for SAM2 on short MoCA clips"
    )
    parser.add_argument("--config",      type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint",  type=str,
                        default="./sam2.1_hiera_large.pt")
    parser.add_argument("--dino_model",  type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--clip_len",    type=int,   default=5,
                        help="Frames per clip (1 prompt + clip_len-1 propagation)")
    parser.add_argument("--clip_stride", type=int,   default=1,
                        help="Sliding-window stride over annotated frames "
                             "(1=fully overlapping, clip_len=non-overlapping)")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-5,
                        help="LR for dino_encoder.proj and cross_attn_fuser")
    parser.add_argument("--decoder_lr",   type=float, default=None,
                        help="If set, also fine-tune sam_mask_decoder at this LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=2,
                        help="Clips per batch (≤2 recommended for 5-frame clips on 24GB GPU)")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--save_every",   type=int,   default=5)
    parser.add_argument("--val_split",    type=float, default=0.15)
    parser.add_argument("--init_weights", type=str,   default=None,
                        help="Fusion checkpoint to warm-start from "
                             "(optimizer/scheduler NOT restored)")
    parser.add_argument("--output_dir",   type=str,
                        default="./checkpoints_dino_fusion_clips")
    parser.add_argument("--moca_frames",  type=str,
                        default="/Experiments/marcol01/frames_train")
    parser.add_argument("--moca_masks",   type=str,
                        default="/Experiments/marcol01/masks_train")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    print("Building SAM2 + DINOv3 fusion model...")
    model        = build_model(args)
    param_groups = freeze_and_collect_trainable(model, args.lr, args.decoder_lr)
    model        = model.to(device)

    if args.init_weights is not None:
        if not os.path.isfile(args.init_weights):
            print(f"ERROR: init_weights not found: {args.init_weights}")
            sys.exit(1)
        print(f"Loading initial weights from: {args.init_weights}")
        ckpt = torch.load(args.init_weights, map_location=device, weights_only=True)
        model.dino_encoder.proj.load_state_dict(ckpt["dino_encoder_proj"])
        model.cross_attn_fuser.load_state_dict(ckpt["cross_attn_fuser"])
        if args.decoder_lr is not None and "sam_mask_decoder" in ckpt:
            model.sam_mask_decoder.load_state_dict(ckpt["sam_mask_decoder"])
            print("  Loaded sam_mask_decoder weights.")
        print("  Done. Optimizer/scheduler start fresh.\n")

    # ---- Dataset ----
    print(f"Loading MoCA clips (clip_len={args.clip_len}, stride={args.clip_stride})...")
    if not os.path.isdir(args.moca_frames) or not os.path.isdir(args.moca_masks):
        print(f"ERROR: MoCA paths not found.\n"
              f"  frames: {args.moca_frames}\n  masks:  {args.moca_masks}")
        sys.exit(1)

    full_ds = MoCAClipDataset(
        args.moca_frames, args.moca_masks,
        clip_len=args.clip_len, clip_stride=args.clip_stride,
    )
    if len(full_ds) == 0:
        print("ERROR: no clips found. "
              "Check that clip_len <= number of annotated frames per sequence.")
        sys.exit(1)

    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Clips: {len(full_ds)} total → train: {n_train}, val: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nStarting clip training for {args.epochs} epochs")
    print(
        f"  LR_fusion={args.lr}  LR_decoder={args.decoder_lr}  "
        f"wd={args.weight_decay}  batch={args.batch_size}  "
        f"clip_len={args.clip_len}"
    )
    print(f"  val_split={args.val_split}")
    print(f"  output: {args.output_dir}\n")

    best_val   = float("inf")
    train_hist: list[float] = []
    val_hist:   list[float] = []

    for epoch in range(args.epochs):
        t0       = time.time()
        tr_loss  = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"Train: {tr_loss:.4f}  Val: {val_loss:.4f} — "
            f"LR: {lr_now:.2e} — Time: {elapsed:.1f}s"
        )
        _print_fusion_params(model)

        train_hist.append(tr_loss)
        val_hist.append(val_loss)
        _save_loss_curve(train_hist, val_hist, args.output_dir)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            path = os.path.join(args.output_dir, f"dino_fusion_epoch{epoch+1:03d}.pt")
            torch.save(
                _make_ckpt(model, epoch + 1, tr_loss, val_loss, args,
                           include_optim=True, optimizer=optimizer,
                           scheduler=scheduler),
                path,
            )
            print(f"  Saved: {path}")

        if val_loss < best_val:
            best_val  = val_loss
            best_path = os.path.join(args.output_dir, "dino_fusion_best.pt")
            torch.save(
                _make_ckpt(model, epoch + 1, tr_loss, val_loss, args),
                best_path,
            )
            print(f"  New best (val={val_loss:.4f}) → {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
