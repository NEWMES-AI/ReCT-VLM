"""
Multi-task Dataset for ReCT-VLM Training

Handles loading and preprocessing of CT-RATE data for multi-task learning:
- Classification labels
- Lesion localization masks
- Report generation text

Author: ReCT-VLM Team
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path


class CTRATEMultiTaskDataset(Dataset):
    """
    Multi-task dataset for CT-RATE with classification, localization, and report generation.

    Args:
        data_dir: Root directory containing CT-RATE data
        labels_csv: Path to predicted_labels.csv
        reports_json: Path to reports JSON file (optional)
        split: 'train', 'valid', or 'test'
        max_slices: Maximum number of slices to sample per volume
        target_size: Target spatial size (H, W)
        use_augmentation: Whether to apply data augmentation
        localization_diseases: List of diseases for localization task
    """

    def __init__(
        self,
        data_dir: str,
        labels_csv: str,
        reports_json: Optional[str] = None,
        split: str = "train",
        max_slices: int = 64,
        target_size: Tuple[int, int] = (512, 512),
        use_augmentation: bool = True,
        localization_diseases: List[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_slices = max_slices
        self.target_size = target_size
        self.use_augmentation = use_augmentation and (split == "train")

        # Disease list for classification (18 diseases)
        self.disease_list = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
            'Pneumothorax', 'Lung nodule', 'Lung opacity', 'Enlarged lymph nodes',
            'Pleural effusion', 'Pericardial effusion'
        ]

        # Diseases with localization (5 diseases)
        if localization_diseases is None:
            self.localization_diseases = [
                'Lung nodule', 'Lung opacity', 'Enlarged lymph nodes',
                'Pleural effusion', 'Pericardial effusion'
            ]
        else:
            self.localization_diseases = localization_diseases

        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df['Split'] == split]

        # Load reports if available
        self.reports = None
        if reports_json and os.path.exists(reports_json):
            with open(reports_json, 'r') as f:
                self.reports = json.load(f)

        # Build dataset index
        self.samples = self._build_index()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _build_index(self) -> List[Dict[str, Any]]:
        """Build index of available samples."""
        samples = []

        for idx, row in self.labels_df.iterrows():
            volume_name = row['VolumeName']

            # Construct paths
            volume_path = self.data_dir / "volume" / f"{volume_name}.npz"
            mask_path = self.data_dir / "masks" / f"{volume_name}.nii.gz"

            # Check if volume exists
            if not volume_path.exists():
                volume_path = self.data_dir / "volume" / f"{volume_name}.nii.gz"
                if not volume_path.exists():
                    continue

            # Build labels vector (18 diseases)
            labels = []
            for disease in self.disease_list:
                if disease in row:
                    labels.append(float(row[disease]))
                else:
                    labels.append(0.0)

            # Check which diseases have localization masks
            has_localization = mask_path.exists()

            # Get report if available
            report = None
            if self.reports and volume_name in self.reports:
                report = self.reports[volume_name]

            sample = {
                'volume_name': volume_name,
                'volume_path': str(volume_path),
                'mask_path': str(mask_path) if has_localization else None,
                'labels': labels,
                'report': report,
            }

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, path: str) -> np.ndarray:
        """Load CT volume from NPZ or NIfTI file."""
        if path.endswith('.npz'):
            data = np.load(path)
            volume = data['volume'] if 'volume' in data else data['imgs']
        else:
            # Load NIfTI
            nii = nib.load(path)
            volume = nii.get_fdata()

        return volume.astype(np.float32)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load segmentation mask from NIfTI file."""
        if path is None or not os.path.exists(path):
            return None

        nii = nib.load(path)
        mask = nii.get_fdata()
        return mask.astype(np.uint8)

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess CT volume with windowing."""
        # CT windowing (lung window: center=40, width=400)
        window_center = 40
        window_width = 400

        lower = window_center - window_width // 2
        upper = window_center + window_width // 2

        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower) * 255.0

        return volume

    def _sample_slices(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample slices from volume."""
        D, H, W = volume.shape

        if D <= self.max_slices:
            # Use all slices
            sampled_volume = volume
            sampled_mask = mask if mask is not None else None
        else:
            # Sample uniformly
            indices = np.linspace(0, D - 1, self.max_slices).astype(int)
            sampled_volume = volume[indices]
            sampled_mask = mask[indices] if mask is not None else None

        return sampled_volume, sampled_mask

    def _resize_slices(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Resize slices to target size."""
        D, H, W = volume.shape

        # Convert to torch tensor
        volume_tensor = torch.from_numpy(volume).float()
        volume_tensor = volume_tensor.unsqueeze(1)  # (D, 1, H, W)

        # Resize
        volume_resized = F.interpolate(
            volume_tensor,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )  # (D, 1, H', W')

        volume_resized = volume_resized.squeeze(1)  # (D, H', W')

        # Resize mask if provided
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            mask_tensor = mask_tensor.unsqueeze(1)  # (D, 1, H, W)

            mask_resized = F.interpolate(
                mask_tensor,
                size=self.target_size,
                mode='nearest'
            )  # (D, 1, H', W')

            mask_resized = mask_resized.squeeze(1).long()  # (D, H', W')
        else:
            mask_resized = None

        return volume_resized, mask_resized

    def _augment(self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply data augmentation."""
        if not self.use_augmentation:
            return volume, mask

        # Random flip
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2])  # Flip width
            if mask is not None:
                mask = torch.flip(mask, dims=[2])

        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip height
            if mask is not None:
                mask = torch.flip(mask, dims=[1])

        # Random brightness/contrast (for volume only)
        if random.random() > 0.5:
            # Brightness
            brightness_factor = random.uniform(0.8, 1.2)
            volume = volume * brightness_factor
            volume = torch.clamp(volume, 0, 255)

        if random.random() > 0.5:
            # Contrast
            mean_val = volume.mean()
            contrast_factor = random.uniform(0.8, 1.2)
            volume = (volume - mean_val) * contrast_factor + mean_val
            volume = torch.clamp(volume, 0, 255)

        return volume, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]

        # Load volume
        volume = self._load_volume(sample['volume_path'])

        # Load mask if available
        mask = None
        if sample['mask_path'] is not None:
            mask = self._load_mask(sample['mask_path'])

        # Preprocess volume
        volume = self._preprocess_volume(volume)

        # Sample slices
        volume, mask = self._sample_slices(volume, mask)

        # Resize
        volume, mask = self._resize_slices(volume, mask)

        # Augmentation
        volume, mask = self._augment(volume, mask)

        # Normalize to [0, 1]
        volume = volume / 255.0

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume = volume.unsqueeze(0)

        # Prepare output
        output = {
            'volume': volume,  # (1, D, H, W)
            'labels': torch.tensor(sample['labels'], dtype=torch.float32),  # (18,)
            'volume_name': sample['volume_name'],
        }

        # Add mask if available
        if mask is not None:
            output['mask'] = mask  # (D, H, W)

        # Add report if available
        if sample['report'] is not None:
            output['report'] = sample['report']

        return output


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multi-task dataset.

    Handles variable-length masks and reports.
    """
    volumes = torch.stack([item['volume'] for item in batch])  # (B, 1, D, H, W)
    labels = torch.stack([item['labels'] for item in batch])  # (B, 18)
    volume_names = [item['volume_name'] for item in batch]

    output = {
        'volume': volumes,
        'labels': labels,
        'volume_name': volume_names,
    }

    # Masks (optional)
    if 'mask' in batch[0]:
        masks = [item['mask'] for item in batch if 'mask' in item]
        if masks:
            output['mask'] = torch.stack(masks)  # (B, D, H, W)

    # Reports (optional)
    if 'report' in batch[0]:
        reports = [item.get('report', '') for item in batch]
        output['report'] = reports

    return output


if __name__ == "__main__":
    # Test dataset
    dataset = CTRATEMultiTaskDataset(
        data_dir="/home/work/3D_CT_Foundation_Model/DATA/CT-RATE/lung_nodule_medsam2",
        labels_csv="/home/work/3D_CT_Foundation_Model/DATA/CT-RATE/train_predicted_labels.csv",
        split="train",
        max_slices=64,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Volume shape: {sample['volume'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    if 'mask' in sample:
        print(f"Mask shape: {sample['mask'].shape}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch volume shape: {batch['volume'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    if 'mask' in batch:
        print(f"Batch mask shape: {batch['mask'].shape}")
