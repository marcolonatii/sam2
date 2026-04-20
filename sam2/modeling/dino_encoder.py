# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DINOv3 encoder wrapper for SAM2.

Extracts patch-level features from an input image and projects them to SAM2's
embedding dimension (256), producing a spatial grid that matches SAM2's 64×64
feature map.

SAM2 uses 1024×1024 images with backbone stride 16 → 64×64 feature map.
DINOv3 (ViT-L/16) uses patch size 16 → 1024/16 = 64 patch grid exactly.
No resize is needed when SAM2's native 1024×1024 input is used.

SAM2 normalizes with ImageNet stats; DINOv3 normalization is read from
AutoImageProcessor and applied after undoing SAM2 normalization.

Output is consumed by CrossAttentionFuser before the SAM mask decoder.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# SAM2 normalization constants (ImageNet stats)
_SAM2_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SAM2_PIXEL_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoEncoder(nn.Module):
    """Wraps DINOv3 (HuggingFace) and projects its patch tokens to ``out_dim``.

    Accepts images normalized with SAM2's convention (ImageNet stats), converts
    them back to [0, 1] range, and re-normalizes with DINOv3's expected stats
    (read from AutoImageProcessor).

    Args:
        model_name: HuggingFace identifier for the DINOv3 model.
        out_dim: Output channel dimension. Must match SAM2's hidden_dim (256).
        freeze_backbone: If True, DINOv3 weights are frozen during training.
        dino_input_size: Spatial size (square) to resize images to before
            passing to DINOv3.  Must be divisible by the patch size (16).
            Default 1024 → 64×64 patch grid, matching SAM2's 64×64 feature map
            with no resize needed for SAM2's native 1024×1024 input.
        sam_pixel_mean: SAM2 pixel mean used to invert SAM normalization.
        sam_pixel_std: SAM2 pixel std used to invert SAM normalization.
    """

    def __init__(
        self,
        model_name: str = _DINOV3_MODEL_ID,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        dino_input_size: int = 1024,
        sam_pixel_mean: Optional[torch.Tensor] = None,
        sam_pixel_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        patch_size = 16
        assert dino_input_size % patch_size == 0, (
            f"dino_input_size={dino_input_size} must be divisible by DINOv3 patch size {patch_size}"
        )

        self.dino_input_size = dino_input_size
        self.n_patches = (dino_input_size // patch_size) ** 2

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dino_embed_dim = self.backbone.config.hidden_size  # 1024 for ViT-L
        self.proj = nn.Linear(dino_embed_dim, out_dim)

        if sam_pixel_mean is None:
            sam_pixel_mean = _SAM2_PIXEL_MEAN.clone()
        if sam_pixel_std is None:
            sam_pixel_std = _SAM2_PIXEL_STD.clone()

        dino_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        dino_std  = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        self.register_buffer("sam_pixel_mean", sam_pixel_mean)
        self.register_buffer("sam_pixel_std",  sam_pixel_std)
        self.register_buffer("dino_pixel_mean", dino_mean)
        self.register_buffer("dino_pixel_std",  dino_std)

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from SAM2 normalization to DINOv3 normalization."""
        x = x * self.sam_pixel_std + self.sam_pixel_mean  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        x = (x - self.dino_pixel_mean) / self.dino_pixel_std
        return x

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Extract and project DINOv3 patch features.

        Args:
            img: SAM2-normalized image tensor of shape ``[B, 3, H, W]``.
                Resized internally to ``dino_input_size × dino_input_size``
                if needed (no-op for SAM2's native 1024×1024 input).

        Returns:
            Projected patch features of shape ``[B, N_patches, out_dim]``,
            where ``N_patches = (dino_input_size // 16) ** 2``.
            For the default 1024 input: ``N_patches = 64 * 64 = 4096``.
        """
        if img.shape[-2] != self.dino_input_size or img.shape[-1] != self.dino_input_size:
            img = F.interpolate(
                img,
                size=(self.dino_input_size, self.dino_input_size),
                mode="bilinear",
                align_corners=False,
            )

        img = self._renormalize(img)

        frozen = not next(iter(self.backbone.parameters())).requires_grad
        if frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=img)
        else:
            outputs = self.backbone(pixel_values=img)

        # last_hidden_state: [B, 1_cls + N_registers + N_patches, dino_embed_dim]
        # Patch tokens are always the last n_patches tokens (CLS and any register tokens come first)
        patch_tokens = outputs.last_hidden_state[:, -self.n_patches:, :]  # [B, N_patches, dino_embed_dim]
        return self.proj(patch_tokens)  # [B, N_patches, out_dim]
