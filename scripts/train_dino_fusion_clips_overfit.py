"""
Overfit sanity-check for the DINOv3 fusion clip training.

Identical to train_dino_fusion_clips.py but limits the dataset to --max_clips
clips and trains on the same clips for both train and val.  A well-functioning
training pipeline should drive loss close to 0 on this tiny fixed set.

Typical usage (from /home/marcol01/sam2/):
    python scripts/train_dino_fusion_clips_overfit.py \\
        --config   configs/sam2.1/sam2.1_hiera_l.yaml \\
        --checkpoint ./sam2.1_hiera_large.pt \\
        --init_weights ./checkpoints_dino_fusion_pretrain_simple/dino_fusion_best.pt \\
        --moca_frames /Experiments/marcol01/frames_train \\
        --moca_masks  /Experiments/marcol01/MoCA-Mask-Pseudo/MoCA-Video-Train \\
        --output_dir  ./checkpoints_dino_fusion_overfit \\
        --max_clips 4 \\
        --epochs 100 \\
        --lr 1e-4
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
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Constants
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
    def __init__(
        self,
        frames_root: str,
        masks_root:  str,
        clip_len:    int = 5,
        clip_stride: int = 1,
        max_clips:   int | None = None,
    ):
        super().__init__()
        self.clips: list[list[tuple[str, str]]] = []

        for seq in sorted(os.listdir(masks_root)):
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
                if not mf.lower().endswith(".png"):
                    continue
                stem = os.path.splitext(mf)[0]
                fp   = os.path.join(frame_dir, stem + ".jpg")
                if not os.path.exists(fp):
                    fp = os.path.join(frame_dir, stem + ".png")
                if os.path.exists(fp):
                    pairs.append((fp, os.path.join(mask_dir, mf)))

            for i in range(0, len(pairs) - clip_len + 1, clip_stride):
                self.clips.append(pairs[i : i + clip_len])
                if max_clips is not None and len(self.clips) >= max_clips:
                    break
            if max_clips is not None and len(self.clips) >= max_clips:
                break

        n_seq = len({os.path.dirname(c[0][0]) for c in self.clips}) if self.clips else 0
        print(
            f"[MoCAClips] clip_len={clip_len}, stride={clip_stride}, "
            f"max_clips={max_clips}: "
            f"{len(self.clips)} clips from {n_seq} sequences  ({frames_root})"
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip   = self.clips[idx]
        images = torch.stack([_load_image(fp) for fp, _  in clip])
        masks  = torch.stack([_load_mask(mp)  for _,  mp in clip])
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
# Model builder / freeze helper
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
# DINO ablation
# ---------------------------------------------------------------------------

def apply_dino_ablation(dino_feats: torch.Tensor, mode: str) -> torch.Tensor:
    """Ablate DINO features before fusion.

    normal  — pass through unchanged
    zero    — replace with zeros (removes all DINO signal)
    shuffle — shuffle token order (destroys spatial correspondence but
              preserves feature statistics; stronger control than zero)
    """
    if mode == "normal":
        return dino_feats
    if mode == "zero":
        return torch.zeros_like(dino_feats)
    if mode == "shuffle":
        B, N, C = dino_feats.shape
        if B > 1:
            # Shuffle samples across batch so each sample gets another
            # sample's DINO tokens (different semantic content, same stats)
            idx = torch.randperm(B, device=dino_feats.device)
            return dino_feats[idx]
        else:
            # B==1: shuffle token (spatial) order within the single sample
            idx = torch.randperm(N, device=dino_feats.device)
            return dino_feats[:, idx, :]
    raise ValueError(f"Unknown dino_mode: {mode!r}  (choose: normal, zero, shuffle)")


# ---------------------------------------------------------------------------
# Clip-level forward pass
# ---------------------------------------------------------------------------

def forward_clip(model, images, masks, device, dino_mode: str = "normal"):
    B, T = images.shape[:2]
    output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    total_loss = None

    for t in range(T):
        frame   = images[:, t]
        gt_mask = masks[:, t]

        with torch.no_grad():
            backbone_out = model.forward_image(frame)
            _, vision_feats, vision_pos_embeds, feat_sizes = \
                model._prepare_backbone_features(backbone_out)

        is_init = (t == 0)

        # --- DINO ablation: pre-hook intercepts dino_feats before the fuser ---
        if dino_mode != "normal":
            def _pre_hook(module, args):
                # args = (pix_feat, dino_feats)  or positional kwargs — handle both
                if len(args) >= 2:
                    return (args[0], apply_dino_ablation(args[1], dino_mode)) + args[2:]
            _hook = model.cross_attn_fuser.register_forward_pre_hook(_pre_hook)

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
            image=frame,
        )

        if dino_mode != "normal":
            _hook.remove()

        if is_init:
            output_dict["cond_frame_outputs"][t] = current_out
        else:
            output_dict["non_cond_frame_outputs"][t] = current_out

        # Skip frame 0 (prediction ≈ GT due to mask-as-output)
        if not is_init:
            loss_t = combined_loss(current_out["pred_masks_high_res"], gt_mask)
            total_loss = loss_t if total_loss is None else total_loss + loss_t

    return total_loss / (T - 1)


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, dataloader, optimizer, scaler, device, epoch, args, train=True):
    if train:
        model.train()
        model.image_encoder.eval()
        model.memory_attention.eval()
        model.memory_encoder.eval()
        if args.decoder_lr is None:
            model.sam_mask_decoder.eval()
        model.sam_prompt_encoder.eval()
        if model.dino_encoder is not None:
            model.dino_encoder.backbone.eval()
    else:
        model.eval()

    epoch_loss, n = 0.0, 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks  = masks.to(device)

            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = forward_clip(model, images, masks, device, dino_mode=args.dino_mode)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += loss.item()
            n += 1
            print(
                f"  [{'Train' if train else 'Val'} Epoch {epoch+1}] "
                f"Batch {batch_idx}/{len(dataloader)}  Loss: {loss.item():.4f}"
            )

    return epoch_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Fusion parameter monitor
# ---------------------------------------------------------------------------

def _print_fusion_params(model):
    """Print alpha and gate openness for each AttentionFuserBlock."""
    fuser = model.cross_attn_fuser
    blocks = fuser.blocks if hasattr(fuser, "blocks") else [fuser]
    for i, block in enumerate(blocks):
        alpha = block.alpha.item()
        # gate[-1] is the output Linear; its bias drives the sigmoid gate activation
        gate_bias = block.gate[-1].bias
        gate_open = torch.sigmoid(gate_bias).mean().item()
        print(
            f"  [fusion block {i}] alpha={alpha:.4f}  "
            f"gate_bias_mean={gate_bias.mean().item():.4f}  "
            f"gate_open={gate_open:.4f}"
        )


# ---------------------------------------------------------------------------
# Checkpoint / plot helpers
# ---------------------------------------------------------------------------

def _make_ckpt(model, epoch, tr_loss, val_loss, args, include_optim=False,
               optimizer=None):
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
    return ckpt


def _save_loss_curve(train_hist, val_hist, output_dir):
    xs = range(1, len(train_hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, train_hist, marker="o", markersize=3, linewidth=1.5, label="Train")
    ax.plot(xs, val_hist,   marker="s", markersize=3, linewidth=1.5, label="Val (same clips)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Overfit test — {len(train_hist)} epochs")
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
        description="Overfit sanity-check for DINOv3 fusion clip training"
    )
    parser.add_argument("--config",      type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint",  type=str,
                        default="./sam2.1_hiera_large.pt")
    parser.add_argument("--dino_model",  type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--clip_len",    type=int,   default=5)
    parser.add_argument("--clip_stride", type=int,   default=1)
    parser.add_argument("--max_clips",   type=int,   default=4,
                        help="Number of clips to use (same set for train and val)")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Higher LR than full training to converge faster")
    parser.add_argument("--decoder_lr",  type=float, default=None)
    parser.add_argument("--weight_decay",type=float, default=0.0,
                        help="Set to 0 for overfit test")
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--save_every",  type=int,   default=25)
    parser.add_argument("--init_weights",type=str,   default=None)
    parser.add_argument("--output_dir",  type=str,
                        default="./checkpoints_dino_fusion_overfit")
    parser.add_argument("--moca_frames", type=str,
                        default="/Experiments/marcol01/frames_train")
    parser.add_argument("--moca_masks",  type=str,
                        default="/Experiments/marcol01/MoCA-Mask-Pseudo/MoCA-Video-Train")
    parser.add_argument("--dino_mode",   type=str, default="normal",
                        choices=["normal", "zero", "shuffle"],
                        help="DINO ablation mode (normal/zero/shuffle)")
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
        print("  Done.\n")

    # ---- Dataset: same N clips for both train and val ----
    print(f"Loading {args.max_clips} clips for overfit test...")
    if not os.path.isdir(args.moca_frames) or not os.path.isdir(args.moca_masks):
        print(f"ERROR: MoCA paths not found.\n"
              f"  frames: {args.moca_frames}\n  masks:  {args.moca_masks}")
        sys.exit(1)

    dataset = MoCAClipDataset(
        args.moca_frames, args.moca_masks,
        clip_len=args.clip_len, clip_stride=args.clip_stride,
        max_clips=args.max_clips,
    )
    if len(dataset) == 0:
        print("ERROR: no clips found.")
        sys.exit(1)

    print(f"Overfitting on {len(dataset)} clip(s) (train == val)\n")

    # Both loaders iterate the same clips
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Starting overfit test: {args.epochs} epochs, "
          f"lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
          f"clips={len(dataset)}")
    print(f"  output: {args.output_dir}\n")

    best_loss   = float("inf")
    train_hist: list[float] = []
    val_hist:   list[float] = []

    for epoch in range(args.epochs):
        t0      = time.time()
        tr_loss = run_epoch(model, train_loader, optimizer, scaler,
                            device, epoch, args, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, scaler,
                             device, epoch, args, train=False)
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
            path = os.path.join(args.output_dir, f"overfit_epoch{epoch+1:03d}.pt")
            torch.save(
                _make_ckpt(model, epoch + 1, tr_loss, val_loss, args,
                           include_optim=True, optimizer=optimizer),
                path,
            )
            print(f"  Saved: {path}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                _make_ckpt(model, epoch + 1, tr_loss, val_loss, args),
                os.path.join(args.output_dir, "overfit_best.pt"),
            )
            print(f"  New best loss: {best_loss:.4f}")

    print(f"\nOverfit test complete. Best loss: {best_loss:.4f}")
    if best_loss > 0.5:
        print("  WARNING: loss did not drop significantly — check training pipeline.")
    else:
        print("  OK: loss dropped, training pipeline appears functional.")


if __name__ == "__main__":
    main()
