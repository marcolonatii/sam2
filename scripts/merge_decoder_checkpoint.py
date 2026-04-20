"""
Merge a finetuned decoder checkpoint into the full SAM2 base checkpoint,
producing a new .pt file that build_sam2_video_predictor can load directly.

Usage:
    python scripts/merge_decoder_checkpoint.py \
        --base_ckpt ./sam2.1_hiera_large.pt \
        --decoder_ckpt ./checkpoints_sam2_decoder_finetune/sam2_decoder_best.pt \
        --output ./checkpoints_sam2_decoder_finetune/sam2_decoder_merged.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to the original SAM2 checkpoint")
    parser.add_argument("--decoder_ckpt", type=str, required=True,
                        help="Path to the finetuned decoder checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the merged checkpoint")
    args = parser.parse_args()

    print(f"Loading base checkpoint: {args.base_ckpt}")
    base = torch.load(args.base_ckpt, map_location="cpu")
    # SAM2 checkpoints may be a flat state dict or wrapped under 'model'
    if isinstance(base, dict) and "model" in base:
        state_dict = base["model"]
        wrap_key = True
    else:
        state_dict = base
        wrap_key = False

    print(f"Loading decoder checkpoint: {args.decoder_ckpt}")
    decoder_ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    decoder_sd = decoder_ckpt["sam_mask_decoder"]
    prompt_sd = decoder_ckpt.get("sam_prompt_encoder", None)

    # Patch decoder weights
    patched = 0
    for k, v in decoder_sd.items():
        full_key = f"sam_mask_decoder.{k}"
        if full_key in state_dict:
            state_dict[full_key] = v
            patched += 1
        else:
            print(f"  [WARN] key not found in base: {full_key}")
    print(f"Patched {patched} decoder parameters")

    # Optionally patch prompt encoder weights
    if prompt_sd is not None:
        patched_pe = 0
        for k, v in prompt_sd.items():
            full_key = f"sam_prompt_encoder.{k}"
            if full_key in state_dict:
                state_dict[full_key] = v
                patched_pe += 1
        print(f"Patched {patched_pe} prompt encoder parameters")

    merged = {"model": state_dict} if wrap_key else state_dict
    torch.save(merged, args.output)
    print(f"Saved merged checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
