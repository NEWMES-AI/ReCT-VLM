"""
3D Transformer Block

Complete transformer block combining:
- Slice-aware self-attention
- Region-aware cross-attention
- Feed-forward network
- Layer normalization and residual connections
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention import SliceAwareAttention, RegionAwareAttention, HybridAttention


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network)

    Standard FFN with GELU activation and dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ThreeDTransformerBlock(nn.Module):
    """
    3D Transformer Block

    Combines slice-aware attention, region-aware attention, and FFN
    with pre-normalization and residual connections.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_regions (int): Number of anatomical regions
        mlp_ratio (float): MLP hidden dim ratio
        dropout (float): Dropout rate
        attn_dropout (float): Attention dropout rate
        slice_distance_weight (float): Slice distance bias weight
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_regions: int = 20,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        slice_distance_weight: float = 0.5
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Slice-aware self-attention
        self.slice_attn = SliceAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            slice_distance_weight=slice_distance_weight
        )

        # Region-aware cross-attention
        self.region_attn = RegionAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        region_features: torch.Tensor,
        patch_to_region: torch.Tensor,
        patch_positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (B, N_patches, embed_dim)
            region_features: Region embeddings (B, N_regions, embed_dim)
            patch_to_region: Assignment matrix (B, N_patches, N_regions)
            patch_positions: Patch positions (B, N_patches, 3)
            attn_mask: Optional attention mask

        Returns:
            Output features (B, N_patches, embed_dim)
        """
        # Slice-aware self-attention with pre-norm and residual
        x = x + self.slice_attn(
            self.norm1(x),
            patch_positions,
            attn_mask
        )

        # Region-aware cross-attention with pre-norm and residual
        x = x + self.region_attn(
            self.norm2(x),
            region_features,
            patch_to_region
        )

        # FFN with pre-norm and residual
        x = x + self.mlp(self.norm3(x))

        return x


class SimpleTransformerBlock(nn.Module):
    """
    Simplified Transformer Block (only self-attention)

    For baseline comparison without region-aware attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        slice_distance_weight: float = 0.5
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = SliceAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            slice_distance_weight=slice_distance_weight
        )

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass (simplified)"""
        x = x + self.attn(self.norm1(x), patch_positions, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    print("Testing ThreeDTransformerBlock...")

    batch_size = 2
    num_patches = 256
    num_regions = 20
    embed_dim = 768
    num_heads = 12

    # Create dummy inputs
    x = torch.randn(batch_size, num_patches, embed_dim)
    region_features = torch.randn(batch_size, num_regions, embed_dim)
    patch_to_region = torch.softmax(
        torch.randn(batch_size, num_patches, num_regions), dim=-1
    )
    patch_positions = torch.rand(batch_size, num_patches, 3)

    # Test full transformer block
    block = ThreeDTransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_regions=num_regions
    )

    output = block(x, region_features, patch_to_region, patch_positions)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ ThreeDTransformerBlock test passed")

    # Test simplified block
    print("\nTesting SimpleTransformerBlock...")

    simple_block = SimpleTransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads
    )

    output = simple_block(x, patch_positions)

    print(f"Output shape: {output.shape}")
    print(f"✓ SimpleTransformerBlock test passed")
