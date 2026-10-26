"""Canvas visual encoder extracting spatial patch tokens from rendered drawing surfaces."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanvasEncoder(nn.Module):
    """Lightweight convolutional patch encoder converting rendered canvas into visual tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 256,
        canvas_res: int = 128,
        patch_grid_res: int = 8,
    ):
        super().__init__()
        self.canvas_res = canvas_res
        self.patch_grid_res = patch_grid_res
        self.d_model = d_model
        self.num_patches = patch_grid_res * patch_grid_res  # e.g. 64 patches

        # Progressive downsampling: 128 -> 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(16, 128)

        self.conv4 = nn.Conv2d(128, d_model, kernel_size=4, stride=2, padding=1)
        self.gn4 = nn.GroupNorm(16, d_model)

        self.act = nn.GELU()

        # Learned 2D spatial positional embeddings for canvas patches
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        # Learnable global summary token [CANVAS_CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        """Encodes (B, 3, H, W) canvas into (B, N_patches + 1, d_model) visual context tokens."""
        B, C, H, W = canvas.shape
        if H != self.canvas_res or W != self.canvas_res:
            canvas = F.interpolate(canvas, size=(self.canvas_res, self.canvas_res), mode="bilinear", align_corners=False)

        x = self.act(self.gn1(self.conv1(canvas)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.act(self.gn3(self.conv3(x)))
        x = self.act(self.gn4(self.conv4(x)))  # (B, d_model, 8, 8)

        # Flatten spatial grid: (B, d_model, 8, 8) -> (B, 64, d_model)
        B, D, gh, gw = x.shape
        patches = x.flatten(2).permute(0, 2, 1)

        # Add 2D spatial positional embeddings
        if patches.shape[1] == self.num_patches:
            patches = patches + self.pos_emb
        else:
            # Interpolate pos_emb if grid differs
            p_emb = F.interpolate(
                self.pos_emb.permute(0, 2, 1).view(1, D, self.patch_grid_res, self.patch_grid_res),
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).permute(0, 2, 1)
            patches = patches + p_emb

        # Prepend [CANVAS_CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls_tokens, patches], dim=1)  # (B, 65, d_model)
        return out
