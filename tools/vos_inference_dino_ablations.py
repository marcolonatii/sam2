# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VOS inference with SAM2 + DINOv3 fusion — ablation modes.

Identical to vos_inference_dino_fusion.py but adds --dino_mode to ablate
how much the DINO fusion branch contributes at inference time:

  real    — normal: use real DINOv3 features (default)
  zero    — keep fusion branch active, zero-out DINO features before cross-attn
  shuffle — keep fusion branch active, shuffle DINO features across batch dim
             NOTE: during video inference batch_size=1 per frame, so shuffle is
             effectively a no-op unless you raise the batch size.  The code
             falls back gracefully (does nothing when B=1).
  off     — bypass DINO branch entirely (pass image=None into track_step)

Usage:
  python tools/vos_inference_dino_ablations.py \\
      --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
      --sam2_checkpoint ./sam2.1_hiera_large.pt \\
      --dino_fusion_checkpoint ./checkpoints_dino_fusion_finetune_simple/dino_fusion_best.pt \\
      --base_video_dir /Experiments/marcol01/frames \\
      --input_mask_dir /Experiments/marcol01/masks \\
      --output_mask_dir /home/marcol01/sam2/sam2_predictions_dino_ablation_<mode> \\
      --dino_mode real   # or zero / shuffle / off
"""

import argparse
import os
import sys
import types
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_with_dino_fusion
from sam2.utils.misc import fill_holes_in_mask_scores


DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


# ---------------------------------------------------------------------------
# Mask I/O helpers
# ---------------------------------------------------------------------------

def load_ann_png(path):
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    return {object_id: (mask == object_id) for object_id in object_ids}


def put_per_obj_mask(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(per_obj_mask)[::-1]:
        object_mask = per_obj_mask[object_id].reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0
    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir, video_name, frame_name, per_obj_output_mask,
    height, width, per_obj_png_file, output_palette,
):
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name), exist_ok=True
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


# ---------------------------------------------------------------------------
# Monkey-patch: activate DINO fusion at inference time
# ---------------------------------------------------------------------------

def _patch_predictor_for_dino_fusion(predictor, dino_mode: str):
    """Monkey-patch SAM2VideoPredictor._run_single_frame_inference.

    dino_mode:
      real    — pass real image into track_step (DINO branch active, real feats)
      zero    — pass real image into track_step (DINO branch active, feats zeroed
                 inside the model via model.dino_mode)
      shuffle — pass real image into track_step (DINO branch active, feats shuffled
                 inside the model via model.dino_mode)
      off     — pass image=None into track_step (DINO branch bypassed entirely)
    """

    def _run_single_frame_inference_with_dino(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # For 'off' mode bypass DINO entirely by not passing the image.
        image_for_track = None if dino_mode == "off" else image

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            image=image_for_track,
        )

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    predictor._run_single_frame_inference = types.MethodType(
        _run_single_frame_inference_with_dino, predictor
    )


# ---------------------------------------------------------------------------
# VOS inference  (identical to vos_inference_dino_fusion.py)
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    if not use_all_masks:
        input_frame_inds = [0]
    else:
        if not per_obj_png_file:
            input_frame_inds = [
                idx
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, f"{name}.png")
                )
            ]
        else:
            input_frame_inds = [
                idx
                for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                )
            ]
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        per_obj_input_mask, input_palette = load_masks_from_dir(
            input_mask_dir=input_mask_dir,
            video_name=video_name,
            frame_name=frame_names[input_frame_idx],
            per_obj_png_file=per_obj_png_file,
        )
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )

    if not object_ids_set:
        raise RuntimeError(f"In {video_name=}, got no object ids.")

    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VOS inference with SAM2 + DINOv3 fusion — ablation modes"
    )
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--dino_fusion_checkpoint",
        type=str,
        required=True,
        help="DINO fusion checkpoint (from train_dino_fusion.py)",
    )
    parser.add_argument("--base_video_dir", type=str, required=True)
    parser.add_argument("--input_mask_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--video_list_file", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--use_all_masks", action="store_true")
    parser.add_argument("--per_obj_png_file", action="store_true")
    parser.add_argument("--apply_postprocessing", action="store_true")
    # DINO architecture options (must match training)
    parser.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument("--dino_input_size", type=int, default=1024)
    parser.add_argument("--cross_attn_num_heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    # ---- ablation flag ----
    parser.add_argument(
        "--dino_mode",
        type=str,
        default="real",
        choices=["real", "zero", "shuffle", "off"],
        help=(
            "real    — use real DINOv3 features (default)\n"
            "zero    — zero out DINO features before cross-attn\n"
            "shuffle — shuffle DINO features across batch dim (no-op for B=1 per frame)\n"
            "off     — bypass DINO branch entirely (image=None into track_step)"
        ),
    )
    args = parser.parse_args()

    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]

    print(f"Building SAM2 + DINOv3 fusion model (dino_mode={args.dino_mode}) ...")
    predictor = build_sam2_with_dino_fusion(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        mode="eval",
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=args.apply_postprocessing,
        dino_model_name=args.dino_model,
        dino_out_dim=256,
        dino_freeze_backbone=True,
        dino_input_size=args.dino_input_size,
        cross_attn_num_heads=args.cross_attn_num_heads,
        use_video_predictor=True,
        strict_checkpoint_loading=False,
    )

    print(f"Loading DINO fusion checkpoint from {args.dino_fusion_checkpoint} ...")
    fusion_ckpt = torch.load(
        args.dino_fusion_checkpoint, map_location=args.device, weights_only=True
    )
    predictor.dino_encoder.proj.load_state_dict(fusion_ckpt["dino_encoder_proj"])
    predictor.cross_attn_fuser.load_state_dict(fusion_ckpt["cross_attn_fuser"])
    print("DINO fusion weights loaded.")

    # Propagate dino_mode onto the predictor (which IS the SAM2Base model) so
    # the feature modification (zero / shuffle) is applied inside track_step.
    predictor.dino_mode = args.dino_mode

    # Monkey-patch to control whether image is forwarded at all (off vs. others).
    _patch_predictor_for_dino_fusion(predictor, dino_mode=args.dino_mode)

    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f if v.strip()]
    else:
        video_names = sorted(
            p for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )
    print(f"Running VOS prediction on {len(video_names)} videos")

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - {video_name}")
        try:
            vos_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
            )
        except Exception as e:
            print(f"  Error on {video_name}: {e}")
            torch.cuda.empty_cache()

    print(
        f"\nCompleted VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
