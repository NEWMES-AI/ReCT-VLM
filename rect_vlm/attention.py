"""
Attention Mechanisms for 3D Vision Encoder

Implements:
1. Slice-Aware Multi-Head Attention: Considers z-axis (slice) proximity
2. Region-Aware Attention: Attends to anatomical regions based on segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SliceAwareAttention(nn.Module):
    """
    Slice-Aware Multi-Head Attention

    Standard multi-head attention with additional bias based on slice distance.
    Patches in the same or adjacent slices have stronger attention connections.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        slice_distance_weight (float): Weight for slice distance bias
        max_slice_distance (int): Maximum slice distance to consider
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        slice_distance_weight: float = 0.5,
        max_slice_distance: int = 32
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.slice_distance_weight = slice_distance_weight
        self.max_slice_distance = max_slice_distance

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable slice distance bias
        # Different heads can learn different distance preferences
        self.slice_bias = nn.Parameter(
            torch.zeros(num_heads, max_slice_distance + 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)

        # Initialize slice bias to be small
        nn.init.trunc_normal_(self.slice_bias, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (B, N, embed_dim)
            patch_positions: Patch positions (B, N, 3) - (z, y, x) coordinates
            attn_mask: Optional attention mask (B, N, N) or (1, N, N)

        Returns:
            Output features (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Add slice distance bias
        if patch_positions is not None:
            slice_bias = self._compute_slice_bias(patch_positions)  # (B, num_heads, N, N)
            attn = attn + slice_bias

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax and dropout
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _compute_slice_bias(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute slice distance bias for attention

        Patches in same/adjacent slices get positive bias (stronger attention),
        patches far apart get negative bias (weaker attention).

        Args:
            patch_positions: (B, N, 3) - (z, y, x) coordinates normalized to [0, 1]

        Returns:
            Slice bias (B, num_heads, N, N)
        """
        B, N, _ = patch_positions.shape

        # Extract z coordinates (slice indices)
        z_coords = patch_positions[:, :, 0]  # (B, N)

        # Compute pairwise slice distances
        z_dist = torch.abs(
            z_coords.unsqueeze(2) - z_coords.unsqueeze(1)
        )  # (B, N, N)

        # Convert normalized distances back to slice indices
        # Assuming z_coords are normalized [0, 1], scale by approximate number of slices
        # This is approximate; for exact indexing, pass actual slice indices
        z_dist_scaled = (z_dist * self.max_slice_distance).long()
        z_dist_clamped = torch.clamp(z_dist_scaled, max=self.max_slice_distance)

        # Get learnable bias for each distance
        # slice_bias: (num_heads, max_slice_distance + 1)
        # z_dist_clamped: (B, N, N)
        slice_bias = self.slice_bias[:, z_dist_clamped]  # (num_heads, B, N, N)
        slice_bias = slice_bias.permute(1, 0, 2, 3)  # (B, num_heads, N, N)

        # Apply weight
        slice_bias = slice_bias * self.slice_distance_weight

        return slice_bias


class RegionAwareAttention(nn.Module):
    """
    Region-Aware Cross-Attention

    Enables patches to attend to their corresponding anatomical regions.
    Uses cross-attention where queries come from patches and keys/values
    from region embeddings.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from patches)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)

        # Key and Value projections (from regions)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=True)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.kv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.q.bias, 0)
        nn.init.constant_(self.kv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(
        self,
        patch_features: torch.Tensor,
        region_features: torch.Tensor,
        patch_to_region: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            patch_features: Patch features (B, N_patches, embed_dim)
            region_features: Region features (B, N_regions, embed_dim)
            patch_to_region: Soft assignment matrix (B, N_patches, N_regions)
                             Values in [0, 1], sum to 1 across regions for each patch

        Returns:
            Output features (B, N_patches, embed_dim)
        """
        B, N_patches, C = patch_features.shape
        _, N_regions, _ = region_features.shape

        # Query from patches
        q = self.q(patch_features).reshape(
            B, N_patches, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # (B, num_heads, N_patches, head_dim)

        # Key and Value from regions
        kv = self.kv(region_features).reshape(
            B, N_regions, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N_regions, head_dim)
        k, v = kv[0], kv[1]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_patches, N_regions)

        # Weight attention by patch-to-region assignment
        # Expand patch_to_region for multi-head
        assignment = patch_to_region.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )  # (B, num_heads, N_patches, N_regions)

        # Modulate attention by assignment (multiplicative gating)
        attn = attn * assignment

        # Add small constant to avoid numerical issues
        attn = attn + 1e-8

        # Softmax over regions
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_patches, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HybridAttention(nn.Module):
    """
    Hybrid Attention combining Slice-Aware and Region-Aware mechanisms

    First applies slice-aware self-attention among patches,
    then applies region-aware cross-attention to anatomical regions.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        slice_distance_weight (float): Weight for slice distance bias
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        slice_distance_weight: float = 0.5
    ):
        super().__init__()

        # Slice-aware self-attention
        self.slice_attn = SliceAwareAttention(
            embed_dim, num_heads, dropout, slice_distance_weight
        )

        # Region-aware cross-attention
        self.region_attn = RegionAwareAttention(
            embed_dim, num_heads, dropout
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: torch.Tensor,
        region_features: torch.Tensor,
        patch_to_region: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (B, N_patches, embed_dim)
            patch_positions: Patch positions (B, N_patches, 3)
            region_features: Region features (B, N_regions, embed_dim)
            patch_to_region: Assignment matrix (B, N_patches, N_regions)
            attn_mask: Optional attention mask

        Returns:
            Output features (B, N_patches, embed_dim)
        """
        # Slice-aware self-attention with residual
        x = x + self.slice_attn(self.norm1(x), patch_positions, attn_mask)

        # Region-aware cross-attention with residual
        x = x + self.region_attn(self.norm2(x), region_features, patch_to_region)

        return x


if __name__ == "__main__":
    # Test Slice-Aware Attention
    print("Testing SliceAwareAttention...")

    batch_size = 2
    num_patches = 256
    embed_dim = 768
    num_heads = 12

    slice_attn = SliceAwareAttention(embed_dim, num_heads)

    x = torch.randn(batch_size, num_patches, embed_dim)
    positions = torch.rand(batch_size, num_patches, 3)  # Random positions

    out = slice_attn(x, positions)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✓ SliceAwareAttention test passed")

    # Test Region-Aware Attention
    print("\nTesting RegionAwareAttention...")

    num_regions = 20
    region_attn = RegionAwareAttention(embed_dim, num_heads)

    region_features = torch.randn(batch_size, num_regions, embed_dim)
    patch_to_region = torch.softmax(torch.randn(batch_size, num_patches, num_regions), dim=-1)

    out = region_attn(x, region_features, patch_to_region)

    print(f"Patch features shape: {x.shape}")
    print(f"Region features shape: {region_features.shape}")
    print(f"Assignment matrix shape: {patch_to_region.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✓ RegionAwareAttention test passed")

    # Test Hybrid Attention
    print("\nTesting HybridAttention...")

    hybrid_attn = HybridAttention(embed_dim, num_heads)

    out = hybrid_attn(x, positions, region_features, patch_to_region)

    print(f"Output shape: {out.shape}")
    print(f"✓ HybridAttention test passed")
