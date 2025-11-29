"""
3D Patch Embedding Module

Converts 3D CT volumes to sequence of patch embeddings while preserving
spatial structure across depth (z-axis) dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class ThreeDPatchEmbedding(nn.Module):
    """
    3D Patch Embedding for CT Volumes

    Extracts 3D patches from CT volumes and projects them to embedding space.
    Unlike 2D patch embedding, this preserves slice continuity and z-axis information.

    Args:
        in_channels (int): Number of input channels (1 for CT)
        embed_dim (int): Embedding dimension
        patch_size (Tuple[int, int, int]): Patch size (depth, height, width)
        img_size (Tuple[int, int, int]): Input image size (depth, height, width)
        use_3d_conv (bool): Use 3D convolution for patch extraction
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        img_size: Tuple[int, int, int] = (64, 512, 512),
        use_3d_conv: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_3d_conv = use_3d_conv

        # Calculate number of patches
        self.num_patches_d = img_size[0] // patch_size[0]
        self.num_patches_h = img_size[1] // patch_size[1]
        self.num_patches_w = img_size[2] // patch_size[2]
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w

        # Patch extraction
        if use_3d_conv:
            # Use 3D convolution for patch extraction (preserves spatial context)
            self.proj = nn.Conv3d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            # Alternative: 2D convolution per slice + aggregation
            self.proj_2d = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:]
            )
            self.depth_aggregation = nn.Conv1d(
                patch_size[0],
                1,
                kernel_size=1
            )

        # 3D Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # CLS token (optional, for global representation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal distribution"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize projection weights
        if self.use_3d_conv:
            nn.init.trunc_normal_(self.proj.weight, std=0.02)
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input CT volume (B, C, D, H, W)

        Returns:
            embeddings: Patch embeddings (B, N_patches + 1, embed_dim) with CLS token
            patch_positions: 3D coordinates of patch centers (B, N_patches, 3)
        """
        B, C, D, H, W = x.shape

        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input size ({D}, {H}, {W}) doesn't match model img_size {self.img_size}"

        if self.use_3d_conv:
            # Extract patches using 3D convolution
            x = self.proj(x)  # (B, embed_dim, D', H', W')
            # Flatten spatial dimensions
            x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        else:
            # Extract patches slice by slice
            patches = []
            for d in range(0, D, self.patch_size[0]):
                slice_group = x[:, :, d:d+self.patch_size[0]]  # (B, C, patch_d, H, W)
                # Pool across depth dimension
                slice_pooled = slice_group.mean(dim=2)  # (B, C, H, W)
                # Extract 2D patches
                patch_2d = self.proj_2d(slice_pooled)  # (B, embed_dim, H', W')
                patches.append(patch_2d)

            # Stack and flatten
            x = torch.stack(patches, dim=2)  # (B, embed_dim, D', H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Get patch positions (centers in 3D space)
        patch_positions = self._get_patch_positions(B, x.device)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_patches + 1, embed_dim)

        return x, patch_positions

    def _get_patch_positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Compute 3D positions of patch centers

        Returns:
            positions: (B, N_patches, 3) - (z, y, x) coordinates normalized to [0, 1]
        """
        # Create mesh grid of patch indices
        d_indices = torch.arange(self.num_patches_d, device=device)
        h_indices = torch.arange(self.num_patches_h, device=device)
        w_indices = torch.arange(self.num_patches_w, device=device)

        # Create 3D grid
        grid_d, grid_h, grid_w = torch.meshgrid(d_indices, h_indices, w_indices, indexing='ij')

        # Flatten
        grid_d = grid_d.reshape(-1)
        grid_h = grid_h.reshape(-1)
        grid_w = grid_w.reshape(-1)

        # Compute patch centers (in original image coordinates)
        center_d = (grid_d + 0.5) * self.patch_size[0]
        center_h = (grid_h + 0.5) * self.patch_size[1]
        center_w = (grid_w + 0.5) * self.patch_size[2]

        # Normalize to [0, 1]
        center_d = center_d / self.img_size[0]
        center_h = center_h / self.img_size[1]
        center_w = center_w / self.img_size[2]

        # Stack coordinates
        positions = torch.stack([center_d, center_h, center_w], dim=-1)  # (N_patches, 3)

        # Expand for batch
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N_patches, 3)

        return positions

    def interpolate_pos_encoding(self, x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
        """
        Interpolate positional encodings for different input sizes

        Useful for fine-tuning on different resolution images.

        Args:
            x: Input embeddings (B, N_patches, embed_dim)
            d, h, w: Target depth, height, width

        Returns:
            Interpolated positional embeddings
        """
        num_patches_d = d // self.patch_size[0]
        num_patches_h = h // self.patch_size[1]
        num_patches_w = w // self.patch_size[2]
        num_patches = num_patches_d * num_patches_h * num_patches_w

        if num_patches == self.num_patches:
            return self.pos_embed

        # Reshape positional embeddings
        pos_embed = self.pos_embed.reshape(
            1, self.num_patches_d, self.num_patches_h, self.num_patches_w, self.embed_dim
        ).permute(0, 4, 1, 2, 3)  # (1, embed_dim, D', H', W')

        # Interpolate to target size
        pos_embed = F.interpolate(
            pos_embed,
            size=(num_patches_d, num_patches_h, num_patches_w),
            mode='trilinear',
            align_corners=False
        )

        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).flatten(1, 3)

        return pos_embed


class SinusoidalPositionalEmbedding3D(nn.Module):
    """
    3D Sinusoidal Positional Embedding

    Alternative to learned positional embeddings. Uses sinusoidal functions
    to encode 3D positions, similar to Transformer positional encoding but
    extended to 3D.
    """

    def __init__(self, embed_dim: int, max_len: Tuple[int, int, int] = (16, 32, 32)):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Create sinusoidal encoding
        pe = self._create_encoding(max_len, embed_dim)
        self.register_buffer('pe', pe)

    def _create_encoding(
        self,
        max_len: Tuple[int, int, int],
        embed_dim: int
    ) -> torch.Tensor:
        """
        Create 3D sinusoidal positional encoding

        Returns:
            pe: (max_len[0] * max_len[1] * max_len[2], embed_dim)
        """
        # Divide embedding dimension among 3 dimensions
        dim_per_axis = embed_dim // 3

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, dim_per_axis, 2).float() *
            (-math.log(10000.0) / dim_per_axis)
        )

        # Create position tensors
        d_pos = torch.arange(max_len[0]).unsqueeze(1)
        h_pos = torch.arange(max_len[1]).unsqueeze(1)
        w_pos = torch.arange(max_len[2]).unsqueeze(1)

        # Compute sinusoidal embeddings for each dimension
        pe_d = torch.zeros(max_len[0], dim_per_axis)
        pe_h = torch.zeros(max_len[1], dim_per_axis)
        pe_w = torch.zeros(max_len[2], dim_per_axis)

        pe_d[:, 0::2] = torch.sin(d_pos * div_term)
        pe_d[:, 1::2] = torch.cos(d_pos * div_term)

        pe_h[:, 0::2] = torch.sin(h_pos * div_term)
        pe_h[:, 1::2] = torch.cos(h_pos * div_term)

        pe_w[:, 0::2] = torch.sin(w_pos * div_term)
        pe_w[:, 1::2] = torch.cos(w_pos * div_term)

        # Broadcast and combine
        # Shape: (D, H, W, dim_per_axis * 3)
        pe_d_exp = pe_d.unsqueeze(1).unsqueeze(2).expand(max_len[0], max_len[1], max_len[2], -1)
        pe_h_exp = pe_h.unsqueeze(0).unsqueeze(2).expand(max_len[0], max_len[1], max_len[2], -1)
        pe_w_exp = pe_w.unsqueeze(0).unsqueeze(1).expand(max_len[0], max_len[1], max_len[2], -1)

        pe = torch.cat([pe_d_exp, pe_h_exp, pe_w_exp], dim=-1)

        # Pad if embed_dim is not divisible by 3
        if embed_dim % 3 != 0:
            padding = embed_dim - pe.shape[-1]
            pe = F.pad(pe, (0, padding))

        # Flatten spatial dimensions
        pe = pe.reshape(-1, embed_dim)

        return pe

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings for given positions

        Args:
            positions: (B, N, 3) - 3D positions (d, h, w) in range [0, max_len-1]

        Returns:
            Positional embeddings (B, N, embed_dim)
        """
        B, N, _ = positions.shape

        # Convert to linear indices
        d = positions[:, :, 0].long()
        h = positions[:, :, 1].long()
        w = positions[:, :, 2].long()

        linear_idx = (
            d * self.max_len[1] * self.max_len[2] +
            h * self.max_len[2] +
            w
        )

        # Index positional encoding
        pe = self.pe[linear_idx]  # (B, N, embed_dim)

        return pe


if __name__ == "__main__":
    # Test 3D Patch Embedding
    print("Testing ThreeDPatchEmbedding...")

    # Create model
    model = ThreeDPatchEmbedding(
        in_channels=1,
        embed_dim=768,
        patch_size=(4, 16, 16),
        img_size=(64, 512, 512)
    )

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 512, 512)

    # Forward pass
    embeddings, positions = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Patch positions shape: {positions.shape}")
    print(f"Number of patches: {model.num_patches}")
    print(f"✓ ThreeDPatchEmbedding test passed")

    # Test sinusoidal encoding
    print("\nTesting SinusoidalPositionalEmbedding3D...")
    sin_pe = SinusoidalPositionalEmbedding3D(embed_dim=768, max_len=(16, 32, 32))
    pos = torch.randint(0, 16, (batch_size, 100, 3))
    pe = sin_pe(pos)
    print(f"Sinusoidal PE shape: {pe.shape}")
    print(f"✓ SinusoidalPositionalEmbedding3D test passed")
