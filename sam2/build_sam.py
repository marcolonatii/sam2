# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def build_sam2_with_dino_fusion(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    # DINO fusion options
    dino_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    dino_out_dim=256,
    dino_freeze_backbone=True,
    dino_input_size=1024,
    cross_attn_num_heads=8,
    use_video_predictor=False,
    strict_checkpoint_loading=True,
):
    """Build a SAM2 model with DINOv3 cross-attention fusion modules attached.

    Loads the SAM2 checkpoint with strict=False (the DINO modules are new),
    then attaches DinoEncoder and CrossAttentionFuser to the model.

    Args:
        config_file: SAM2 Hydra config file (e.g. "configs/sam2/sam2_hiera_l.yaml").
        ckpt_path: Path to SAM2 checkpoint. None = no checkpoint loaded.
        device: Target device.
        mode: "eval" or "train".
        hydra_overrides_extra: Additional Hydra overrides.
        apply_postprocessing: Whether to apply SAM2 postprocessing overrides.
        dino_model_name: HuggingFace model ID for DINOv3.
        dino_out_dim: Projection output dim (must match SAM2 hidden_dim=256).
        dino_freeze_backbone: Whether to freeze DINOv3 backbone weights.
        dino_input_size: Spatial size for DINOv3 input (must be divisible by 16).
            Default 1024 → 64×64 patch grid matching SAM2's 64×64 feature map
            with no resize needed for SAM2's native 1024×1024 input.
        cross_attn_num_heads: Number of heads in cross-attention fuser.
        use_video_predictor: If True, build as SAM2VideoPredictor; else SAM2Base.
        strict_checkpoint_loading: If False, ignore missing/unexpected keys (e.g.
            when loading a base SAM2 checkpoint into a DINO-augmented model).
    """
    from sam2.modeling.dino_encoder import DinoEncoder
    from sam2.modeling.cross_attention_fuser import CrossAttentionFuser

    if use_video_predictor:
        model = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=None,  # load manually below with strict=False
            device="cpu",
            mode=mode,
            hydra_overrides_extra=hydra_overrides_extra,
            apply_postprocessing=apply_postprocessing,
        )
    else:
        model = build_sam2(
            config_file=config_file,
            ckpt_path=None,
            device="cpu",
            mode=mode,
            hydra_overrides_extra=hydra_overrides_extra,
            apply_postprocessing=apply_postprocessing,
        )

    # Load SAM2 checkpoint with strict=False so new DINO keys are silently missing
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if strict_checkpoint_loading:
            if missing:
                logging.error(f"Missing keys: {missing}")
                raise RuntimeError("Strict checkpoint loading failed — missing keys")
            if unexpected:
                logging.error(f"Unexpected keys: {unexpected}")
                raise RuntimeError("Strict checkpoint loading failed — unexpected keys")
        else:
            if missing:
                logging.info(f"Checkpoint missing keys (expected for DINO fusion init): {missing}")
            if unexpected:
                logging.warning(f"Checkpoint unexpected keys: {unexpected}")

    # Attach DINO fusion modules
    model.dino_encoder = DinoEncoder(
        model_name=dino_model_name,
        out_dim=dino_out_dim,
        freeze_backbone=dino_freeze_backbone,
        dino_input_size=dino_input_size,
    )
    model.cross_attn_fuser = CrossAttentionFuser(
        embed_dim=dino_out_dim,
        num_heads=cross_attn_num_heads,
    )

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return model


def build_sam2_with_dino_fusion_hf(model_id, **kwargs):
    """Like build_sam2_with_dino_fusion but downloads checkpoint from HuggingFace."""
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_with_dino_fusion(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
