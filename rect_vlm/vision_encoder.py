"""
Complete 3D Vision Encoder

Integrates all components:
- 3D Patch Embedding
- Anatomical Structure Encoder
- Stacked Transformer Blocks
- Multi-scale Feature Aggregation
- Projection Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .patch_embedding import ThreeDPatchEmbedding
from .anatomical_encoder import AnatomicalStructureEncoder
from .transformer_block import ThreeDTransformerBlock


class ThreeDVisionEncoder(nn.Module):
    """
    3D CT-Specialized Vision Encoder

    Complete encoder that produces multi-granular representations:
    - Global features: Volume-level representation
    - Local features: Patch-level representations
    - Region features: Anatomical region-specific representations

    Args:
        in_channels (int): Input channels (1 for CT)
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        num_regions (int): Number of anatomical regions
        patch_size (Tuple[int, int, int]): Patch size
        img_size (Tuple[int, int, int]): Input image size
        mlp_ratio (float): MLP hidden dim ratio
        dropout (float): Dropout rate
        attn_dropout (float): Attention dropout
        slice_distance_weight (float): Slice distance bias weight
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_regions: int = 20,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        img_size: Tuple[int, int, int] = (64, 512, 512),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        slice_distance_weight: float = 0.5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_regions = num_regions
        self.patch_size = patch_size
        self.img_size = img_size

        # 1. Patch Embedding
        self.patch_embed = ThreeDPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )

        num_patches = self.patch_embed.num_patches

        # 2. Anatomical Structure Encoder
        self.anatomical_encoder = AnatomicalStructureEncoder(
            num_regions=num_regions,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            ThreeDTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_regions=num_regions,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                slice_distance_weight=slice_distance_weight
            ) for _ in range(depth)
        ])

        # 4. Layer Norm (final)
        self.norm = nn.LayerNorm(embed_dim)

        # 5. Feature Aggregation
        # Global pooling (for volume-level features)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Region pooling layers
        self.region_pool = RegionPooling(embed_dim)

        # 6. Projection Heads
        # Global head: For volume-level alignment with text
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Local head: For patch-level features
        self.local_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Region head: For anatomical region features
        self.region_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection heads"""
        for m in [self.global_proj, self.local_proj, self.region_proj]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        ct_volume: torch.Tensor,
        segmentation_mask: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            ct_volume: (B, 1, D, H, W) - CT volume
            segmentation_mask: (B, 1, D, H, W) - Segmentation mask
            return_intermediate: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - global_features: (B, embed_dim) - Volume-level features
            - local_features: (B, N_patches, embed_dim) - Patch features
            - region_features: (B, N_regions, embed_dim) - Region features
            - cls_token: (B, embed_dim) - CLS token (global representation)
            - patch_features_raw: (B, N_patches, embed_dim) - Pre-projection
        """
        B = ct_volume.shape[0]

        # 1. Patch Embedding
        x, patch_positions = self.patch_embed(ct_volume)
        # x: (B, N_patches + 1, embed_dim) with CLS token
        # patch_positions: (B, N_patches, 3)

        # Separate CLS token and patch tokens
        cls_token = x[:, 0]  # (B, embed_dim)
        patch_tokens = x[:, 1:]  # (B, N_patches, embed_dim)

        # 2. Encode Anatomical Structure
        region_embeds, patch_to_region = self.anatomical_encoder(
            segmentation_mask, patch_positions
        )
        # region_embeds: (B, N_regions, embed_dim)
        # patch_to_region: (B, N_patches, N_regions)

        # 3. Transformer Blocks
        intermediate_features = []

        for blk in self.blocks:
            patch_tokens = blk(
                patch_tokens,
                region_embeds,
                patch_to_region,
                patch_positions
            )

            if return_intermediate:
                intermediate_features.append(patch_tokens)

        # 4. Final Layer Norm
        patch_tokens = self.norm(patch_tokens)

        # 5. Feature Aggregation
        # Global features (average pooling over patches)
        global_features = self.global_pool(
            patch_tokens.transpose(1, 2)
        ).squeeze(-1)  # (B, embed_dim)

        # Region features (weighted pooling by patch-to-region assignments)
        region_features = self.region_pool(
            patch_tokens, patch_to_region, region_embeds
        )  # (B, N_regions, embed_dim)

        # 6. Projection to shared embedding space
        global_proj = self.global_proj(global_features)  # (B, embed_dim)
        local_proj = self.local_proj(patch_tokens)  # (B, N_patches, embed_dim)
        region_proj = self.region_proj(region_features)  # (B, N_regions, embed_dim)

        # Prepare output dictionary
        outputs = {
            'global_features': global_proj,
            'local_features': local_proj,
            'region_features': region_proj,
            'cls_token': cls_token,
            'patch_features_raw': patch_tokens,
            'patch_positions': patch_positions,
            'patch_to_region': patch_to_region
        }

        if return_intermediate:
            outputs['intermediate_features'] = intermediate_features

        return outputs


class RegionPooling(nn.Module):
    """
    Region Pooling Layer

    Pools patch features into region features using patch-to-region assignments.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_to_region: torch.Tensor,
        region_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool patches into regions

        Args:
            patch_features: (B, N_patches, embed_dim)
            patch_to_region: (B, N_patches, N_regions) - soft assignments
            region_embeds: (B, N_regions, embed_dim) - region prototypes

        Returns:
            region_features: (B, N_regions, embed_dim)
        """
        # Weighted pooling: region_feat = sum(patch_feat * assignment)
        # patch_to_region: (B, N_patches, N_regions)
        # patch_features: (B, N_patches, embed_dim)

        # Transpose and matrix multiply
        region_features = torch.bmm(
            patch_to_region.transpose(1, 2),  # (B, N_regions, N_patches)
            patch_features  # (B, N_patches, embed_dim)
        )  # (B, N_regions, embed_dim)

        # Add region prototype embeddings
        region_features = region_features + region_embeds

        return region_features


if __name__ == "__main__":
    print("Testing ThreeDVisionEncoder...")

    batch_size = 2
    D, H, W = 64, 512, 512
    in_channels = 1
    embed_dim = 768
    depth = 6  # Smaller for testing
    num_heads = 12
    num_regions = 20

    # Create model
    model = ThreeDVisionEncoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_regions=num_regions,
        patch_size=(4, 16, 16),
        img_size=(D, H, W)
    )

    # Create dummy inputs
    ct_volume = torch.randn(batch_size, in_channels, D, H, W)
    seg_mask = torch.randint(0, num_regions + 1, (batch_size, 1, D, H, W))

    # Forward pass
    print("Running forward pass...")
    outputs = model(ct_volume, seg_mask)

    # Print output shapes
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Check dimensions
    assert outputs['global_features'].shape == (batch_size, embed_dim)
    assert outputs['local_features'].shape[0] == batch_size
    assert outputs['local_features'].shape[2] == embed_dim
    assert outputs['region_features'].shape == (batch_size, num_regions, embed_dim)

    print(f"\n✓ ThreeDVisionEncoder test passed")

    # Test with intermediate features
    print("\nTesting with intermediate features...")
    outputs = model(ct_volume, seg_mask, return_intermediate=True)
    print(f"Number of intermediate layers: {len(outputs['intermediate_features'])}")
    print(f"✓ Intermediate features test passed")
