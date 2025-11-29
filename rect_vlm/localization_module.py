"""
Lesion Localization Module (3-stage)

Three-stage approach for precise lesion segmentation:
1. Text Representation: Disease-specific text embeddings
2. Denoising: Feature compression with noise removal
3. Attention U-Net: High-resolution segmentation with text guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel


class TextEmbedder(nn.Module):
    """
    Stage 1: Text Representation Redefinition

    Converts disease text descriptions to rich semantic embeddings
    using medical domain language models.
    """

    # Disease-specific prompts with anatomical context
    DISEASE_PROMPTS = {
        "pericardial_effusion": (
            "Pericardial effusion: fluid accumulation in the pericardial cavity "
            "surrounding the heart, visible as low-density region around cardiac silhouette, "
            "measured in the pericardial space between visceral and parietal pericardium"
        ),
        "pleural_effusion": (
            "Pleural effusion: fluid accumulation in the pleural space between lung and chest wall, "
            "appearing as homogeneous opacity in dependent portions of thorax, "
            "often with meniscus sign and blunting of costophrenic angles"
        ),
        "consolidation": (
            "Consolidation: replacement of alveolar air with fluid, pus, blood, or cells, "
            "resulting in increased lung density on CT, air bronchograms may be visible, "
            "typically lobar or segmental distribution in lung parenchyma"
        ),
        "ground_glass_opacity": (
            "Ground-glass opacity: hazy increased lung attenuation on CT imaging "
            "with preserved bronchial and vascular markings, indicating partial alveolar filling, "
            "interstitial thickening, or partial lung collapse, distributed in lung parenchyma"
        ),
        "lung_nodule": (
            "Lung nodule: round or oval-shaped opacity in lung parenchyma, "
            "measuring up to 3 cm in diameter, well or poorly defined margins, "
            "solid, part-solid, or ground-glass attenuation, located within lung tissue"
        )
    }

    def __init__(
        self,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 768,
        freeze_encoder: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Load text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size

        # Projection to target dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Pre-compute disease embeddings
        self.disease_names = list(self.DISEASE_PROMPTS.keys())
        self.register_buffer('disease_embeds', torch.zeros(len(self.disease_names), embed_dim))
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Pre-compute text embeddings for all diseases."""
        print(f"Initializing {len(self.disease_names)} disease text embeddings...")

        with torch.no_grad():
            for idx, disease in enumerate(self.disease_names):
                prompt = self.DISEASE_PROMPTS[disease]
                embed = self._encode_text(prompt)
                self.disease_embeds[idx] = embed

        print("Disease embeddings initialized")

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding vector."""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.text_encoder.device)

        outputs = self.text_encoder(**tokens)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Project
        text_features = self.text_proj(text_features)

        return text_features.squeeze(0)

    def forward(self, disease_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get disease text embeddings.

        Args:
            disease_indices: Optional indices to select specific diseases (B,)
                           If None, returns all disease embeddings

        Returns:
            Text embeddings (B, embed_dim) or (num_diseases, embed_dim)
        """
        if disease_indices is None:
            return self.disease_embeds  # (5, embed_dim)
        else:
            return self.disease_embeds[disease_indices]  # (B, embed_dim)


class DenoisingTransformer(nn.Module):
    """
    Stage 2: 3D CT Representation Compression with Denoising

    Compresses local patch features while removing noise and preserving
    lesion information using transformer blocks with text guidance.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Denoising transformer blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoise and compress patch features with text guidance.

        Args:
            patch_features: Local patch features (B, N_patches, embed_dim)
            text_features: Disease text embeddings (B, embed_dim)

        Returns:
            Denoised features (B, N_patches, embed_dim)
        """
        x = patch_features

        # Apply denoising blocks
        for block in self.blocks:
            x = block(x, text_features)

        # Output projection
        x = self.output_proj(x)

        return x


class DenoisingBlock(nn.Module):
    """Transformer block with text-guided cross-attention for denoising."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Self-attention for patch features
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention to text features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch features (B, N_patches, embed_dim)
            text_features: Text embeddings (B, embed_dim)

        Returns:
            Denoised features (B, N_patches, embed_dim)
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to text
        x_norm = self.norm2(x)
        text_expanded = text_features.unsqueeze(1)  # (B, 1, embed_dim)
        attn_out, _ = self.cross_attn(x_norm, text_expanded, text_expanded)
        x = x + attn_out

        # FFN
        x = x + self.mlp(self.norm3(x))

        return x


class TextGuidedAttentionGate(nn.Module):
    """
    Attention gate with text guidance for U-Net skip connections.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, F_text: int = 768):
        """
        Args:
            F_g: Number of feature maps in gating signal (decoder)
            F_l: Number of feature maps in skip connection (encoder)
            F_int: Number of intermediate feature maps
            F_text: Dimension of text features
        """
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        # Text projection to spatial attention
        self.W_text = nn.Sequential(
            nn.Linear(F_text, F_int),
            nn.ReLU(inplace=True)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder (B, F_g, D, H, W)
            x: Skip connection from encoder (B, F_l, D, H, W)
            text_features: Text embeddings (B, F_text)

        Returns:
            Attention-gated features (B, F_l, D, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Text guidance
        text_proj = self.W_text(text_features)  # (B, F_int)
        text_proj = text_proj.view(text_proj.size(0), text_proj.size(1), 1, 1, 1)  # (B, F_int, 1, 1, 1)

        # Combine decoder, encoder, and text features
        psi = self.relu(g1 + x1 + text_proj)
        psi = self.psi(psi)  # (B, 1, D, H, W)

        return x * psi


class TextGuidedAttentionUNet(nn.Module):
    """
    Stage 3: Attention U-Net with Text Guidance

    Generates high-resolution segmentation masks using U-Net architecture
    with text-guided attention gates.
    """

    def __init__(
        self,
        in_channels: int = 768,
        base_channels: int = 64,
        text_dim: int = 768
    ):
        super().__init__()

        # Encoder path
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Text injection at bottleneck
        self.text_injector = nn.Sequential(
            nn.Linear(text_dim, base_channels * 16),
            nn.ReLU(inplace=True)
        )

        # Decoder path with attention gates
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = TextGuidedAttentionGate(base_channels * 8, base_channels * 8, base_channels * 4, text_dim)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = TextGuidedAttentionGate(base_channels * 4, base_channels * 4, base_channels * 2, text_dim)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = TextGuidedAttentionGate(base_channels * 2, base_channels * 2, base_channels, text_dim)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = TextGuidedAttentionGate(base_channels, base_channels, base_channels // 2, text_dim)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Final segmentation layer
        self.final = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """3D convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, in_channels, D, H, W)
            text_features: Disease text embeddings (B, text_dim)

        Returns:
            Segmentation mask (B, 1, D, H, W)
        """
        # Encoder path
        enc1 = self.enc1(x)  # (B, 64, D, H, W)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)  # (B, 128, D/2, H/2, W/2)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)  # (B, 256, D/4, H/4, W/4)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)  # (B, 512, D/8, H/8, W/8)
        x = self.pool4(enc4)

        # Bottleneck with text injection
        x = self.bottleneck(x)  # (B, 1024, D/16, H/16, W/16)

        # Inject text features
        text_proj = self.text_injector(text_features)  # (B, 1024)
        text_proj = text_proj.view(text_proj.size(0), text_proj.size(1), 1, 1, 1)
        x = x + text_proj

        # Decoder path with attention
        x = self.up4(x)  # (B, 512, D/8, H/8, W/8)
        enc4 = self.att4(x, enc4, text_features)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)  # (B, 256, D/4, H/4, W/4)
        enc3 = self.att3(x, enc3, text_features)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)  # (B, 128, D/2, H/2, W/2)
        enc2 = self.att2(x, enc2, text_features)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)  # (B, 64, D, H, W)
        enc1 = self.att1(x, enc1, text_features)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Final segmentation
        x = self.final(x)  # (B, 1, D, H, W)

        return x


class LesionLocalizationModule(nn.Module):
    """
    Complete 3-stage lesion localization module.

    Combines text embedding, denoising, and U-Net for precise lesion segmentation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_diseases: int = 5,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        denoising_layers: int = 4,
        unet_base_channels: int = 64
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        # Stage 1: Text embedder
        self.text_embedder = TextEmbedder(
            text_model_name=text_model_name,
            embed_dim=embed_dim
        )

        # Stage 2: Denoising transformer
        self.denoising = DenoisingTransformer(
            embed_dim=embed_dim,
            num_layers=denoising_layers
        )

        # Stage 3: Text-guided Attention U-Net (one per disease)
        self.unets = nn.ModuleList([
            TextGuidedAttentionUNet(
                in_channels=embed_dim,
                base_channels=unet_base_channels,
                text_dim=embed_dim
            )
            for _ in range(num_diseases)
        ])

        # Patch-to-3D reshaping parameters
        self.register_buffer('patch_size', torch.tensor([4, 16, 16]))  # (D, H, W)

    def _reshape_patches_to_3d(
        self,
        patch_features: torch.Tensor,
        original_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Reshape patch features back to 3D grid.

        Args:
            patch_features: (B, N_patches, embed_dim)
            original_shape: (D, H, W) original volume shape

        Returns:
            3D feature map (B, embed_dim, D', H', W')
        """
        B, N, C = patch_features.shape

        # Calculate grid dimensions
        D, H, W = original_shape
        D_p, H_p, W_p = self.patch_size.tolist()
        D_grid = D // D_p
        H_grid = H // H_p
        W_grid = W // W_p

        # Reshape
        x = patch_features.view(B, D_grid, H_grid, W_grid, C)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D', H', W')

        return x

    def forward(
        self,
        local_features: torch.Tensor,
        region_features: Optional[torch.Tensor] = None,
        disease_indices: Optional[List[int]] = None,
        original_shape: Tuple[int, int, int] = (64, 512, 512)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for lesion localization.

        Args:
            local_features: Local patch features (B, N_patches, embed_dim)
            region_features: Optional region features (B, N_regions, embed_dim)
            disease_indices: Which diseases to segment (default: all)
            original_shape: Original volume shape for reshaping

        Returns:
            Dictionary with segmentation masks for each disease
        """
        B = local_features.size(0)

        if disease_indices is None:
            disease_indices = list(range(self.num_diseases))

        # Stage 1: Get disease text embeddings
        text_embeds = self.text_embedder()  # (5, embed_dim)

        # Process each disease
        outputs = {}
        disease_names = list(self.text_embedder.disease_names)

        for disease_idx in disease_indices:
            disease_name = disease_names[disease_idx]
            text_embed = text_embeds[disease_idx].unsqueeze(0).expand(B, -1)  # (B, embed_dim)

            # Stage 2: Denoising
            denoised = self.denoising(local_features, text_embed)  # (B, N_patches, embed_dim)

            # Reshape to 3D grid
            features_3d = self._reshape_patches_to_3d(denoised, original_shape)  # (B, embed_dim, D', H', W')

            # Stage 3: U-Net segmentation
            seg_mask = self.unets[disease_idx](features_3d, text_embed)  # (B, 1, D', H', W')

            # Upsample to original resolution
            seg_mask = F.interpolate(
                seg_mask,
                size=original_shape,
                mode='trilinear',
                align_corners=False
            )  # (B, 1, D, H, W)

            outputs[disease_name] = seg_mask.squeeze(1)  # (B, D, H, W)

        return outputs


# Test code
if __name__ == "__main__":
    print("Testing Lesion Localization Module...")

    # Test parameters
    batch_size = 1
    embed_dim = 768
    num_patches = 16384
    original_shape = (64, 512, 512)

    # Create dummy data
    local_features = torch.randn(batch_size, num_patches, embed_dim)

    print("\n1. Testing TextEmbedder...")
    text_embedder = TextEmbedder(
        text_model_name="distilbert-base-uncased",  # Faster for testing
        embed_dim=embed_dim
    )
    text_embeds = text_embedder()
    print(f"   Text embeddings shape: {text_embeds.shape}")

    print("\n2. Testing DenoisingTransformer...")
    denoising = DenoisingTransformer(embed_dim=embed_dim, num_layers=2)
    text_embed = text_embeds[0].unsqueeze(0).expand(batch_size, -1)
    denoised = denoising(local_features, text_embed)
    print(f"   Denoised features shape: {denoised.shape}")

    print("\n3. Testing TextGuidedAttentionUNet...")
    # Create smaller 3D feature map for testing
    D, H, W = 16, 32, 32
    features_3d = torch.randn(batch_size, embed_dim, D, H, W)

    unet = TextGuidedAttentionUNet(
        in_channels=embed_dim,
        base_channels=32,  # Smaller for testing
        text_dim=embed_dim
    )
    seg_mask = unet(features_3d, text_embed)
    print(f"   Segmentation mask shape: {seg_mask.shape}")

    print("\n4. Testing Complete LesionLocalizationModule...")
    localization_module = LesionLocalizationModule(
        embed_dim=embed_dim,
        num_diseases=5,
        text_model_name="distilbert-base-uncased",
        denoising_layers=2,
        unet_base_channels=32
    )

    # Test with smaller shape for memory
    test_shape = (16, 64, 64)
    test_patches = (16 // 4) * (64 // 16) * (64 // 16)  # 64 patches
    test_features = torch.randn(batch_size, test_patches, embed_dim)

    outputs = localization_module(
        test_features,
        disease_indices=[0, 1],  # Test first two diseases
        original_shape=test_shape
    )

    print(f"   Number of outputs: {len(outputs)}")
    for disease, mask in outputs.items():
        print(f"   {disease}: {mask.shape}")

    print("\n✓ All tests passed!")
