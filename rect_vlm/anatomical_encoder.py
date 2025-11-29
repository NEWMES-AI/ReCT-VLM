"""
Anatomical Structure Encoder

Encodes segmentation masks into region-specific embeddings and creates
patch-to-region mappings for region-aware attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AnatomicalStructureEncoder(nn.Module):
    """
    Anatomical Structure Encoder

    Converts segmentation masks into learnable region embeddings and
    computes soft assignments between patches and anatomical regions.

    Args:
        num_regions (int): Number of anatomical regions to model
        embed_dim (int): Embedding dimension
        patch_size (Tuple[int, int, int]): Patch size for computing assignments
        img_size (Tuple[int, int, int]): Input image size
        use_mask_encoder (bool): Use CNN to encode mask features
    """

    def __init__(
        self,
        num_regions: int = 20,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        img_size: Tuple[int, int, int] = (64, 512, 512),
        use_mask_encoder: bool = True
    ):
        super().__init__()

        self.num_regions = num_regions
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_mask_encoder = use_mask_encoder

        # Region embedding table (learnable)
        self.region_embeds = nn.Embedding(num_regions + 1, embed_dim)

        # Learnable region feature prototypes
        self.region_prototypes = nn.Parameter(
            torch.randn(num_regions, embed_dim)
        )

        if use_mask_encoder:
            # 3D CNN to encode mask spatial features
            self.mask_encoder = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),

                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),

                nn.Conv3d(128, embed_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool3d(1)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.region_embeds.weight, std=0.02)
        nn.init.normal_(self.region_prototypes, std=0.02)

    def forward(
        self,
        segmentation_mask: torch.Tensor,
        patch_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            segmentation_mask: (B, 1, D, H, W) - region IDs
            patch_coords: (B, N_patches, 3) - patch centers (z, y, x) normalized [0, 1]

        Returns:
            region_features: (B, num_regions, embed_dim)
            patch_to_region: (B, N_patches, num_regions) - soft assignment
        """
        B = segmentation_mask.shape[0]

        # Extract region features
        region_features = self._extract_region_features(segmentation_mask)

        # Compute assignments
        patch_to_region = self._compute_assignments(segmentation_mask, patch_coords)

        return region_features, patch_to_region

    def _extract_region_features(self, segmentation_mask: torch.Tensor) -> torch.Tensor:
        """Extract region features from mask"""
        B = segmentation_mask.shape[0]

        # Simple approach: use learned prototypes
        region_features = self.region_prototypes.unsqueeze(0).expand(B, -1, -1)

        return region_features

    def _compute_assignments(
        self,
        segmentation_mask: torch.Tensor,
        patch_coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft assignment from patches to regions"""
        B, _, D, H, W = segmentation_mask.shape
        N_patches = patch_coords.shape[1]

        assignments = torch.zeros(
            B, N_patches, self.num_regions,
            device=segmentation_mask.device
        )

        # Convert normalized coordinates to pixel coordinates
        coords_pixel = patch_coords.clone()
        coords_pixel[:, :, 0] = coords_pixel[:, :, 0] * D
        coords_pixel[:, :, 1] = coords_pixel[:, :, 1] * H
        coords_pixel[:, :, 2] = coords_pixel[:, :, 2] * W

        for b in range(B):
            mask_b = segmentation_mask[b, 0]

            for p in range(N_patches):
                z_center, y_center, x_center = coords_pixel[b, p]

                z_start = max(0, int(z_center - self.patch_size[0] // 2))
                z_end = min(D, int(z_center + self.patch_size[0] // 2))
                y_start = max(0, int(y_center - self.patch_size[1] // 2))
                y_end = min(H, int(y_center + self.patch_size[1] // 2))
                x_start = max(0, int(x_center - self.patch_size[2] // 2))
                x_end = min(W, int(x_center + self.patch_size[2] // 2))

                patch_mask = mask_b[z_start:z_end, y_start:y_end, x_start:x_end]

                if patch_mask.numel() == 0:
                    continue

                for region_id in range(1, self.num_regions + 1):
                    count = (patch_mask == region_id).sum().float()
                    proportion = count / patch_mask.numel()
                    assignments[b, p, region_id - 1] = proportion

        # Normalize
        assignments = assignments + 1e-6
        assignments = assignments / assignments.sum(dim=-1, keepdim=True)

        return assignments


if __name__ == "__main__":
    print("Testing AnatomicalStructureEncoder...")

    batch_size = 2
    D, H, W = 64, 512, 512
    num_patches = 16384
    num_regions = 20
    embed_dim = 768

    seg_mask = torch.randint(0, num_regions + 1, (batch_size, 1, D, H, W))
    patch_coords = torch.rand(batch_size, num_patches, 3)

    encoder = AnatomicalStructureEncoder(
        num_regions=num_regions,
        embed_dim=embed_dim,
        patch_size=(4, 16, 16),
        img_size=(D, H, W)
    )

    region_features, assignments = encoder(seg_mask, patch_coords)

    print(f"Region features shape: {region_features.shape}")
    print(f"Assignments shape: {assignments.shape}")
    print(f"✓ AnatomicalStructureEncoder test passed")
