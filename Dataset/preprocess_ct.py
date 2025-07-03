"""
This script is used to preprocess the CT dataset using GPUs. (CT-RATE)
- main function: 1. Read NIfTI files, metadata
                 2. Transform to HU(Hounsfield Unit)
                 3. Resize images to target spacing
                 4. Save preprocessed data
You can change the 2 things.
1. spacing, hu_min, hu_max, and train_range values in the ProcessingConfig class
2. preprocess_dataset_path, metadata_path
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple, Optional
from multiprocessing import Pool, set_start_method

# set start multi-processing method to 'spawn' (It is necessary when using CUDA)
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Define the target spacing values and other parameters
@dataclass
class ProcessingConfig:
    """Class to manage preprocessing settings"""
    target_x_spacing: float = 0.75
    target_y_spacing: float = 0.75
    target_z_spacing: float = 1.5
    hu_min: int = -1000
    hu_max: int = 1000
    train_range: Tuple[int, int] = (1, 5000)

def read_nii_files(directory: str) -> List[str]:
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files filtered between train1 and train5000.
    """
    nii_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                try:
                    train_num = int(file.split('_')[1])
                    if ProcessingConfig.train_range[0] <= train_num <= ProcessingConfig.train_range[1]:
                        nii_files.append(os.path.join(root, file))
                except (IndexError, ValueError):
                    continue
    return sorted(nii_files)

def read_nii_data(file_path: str) -> Optional[np.ndarray]:
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        return nii_img.get_fdata()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array: torch.Tensor, 
                current_spacing: Tuple[float, float, float],
                target_spacing: Tuple[float, float, float], 
                device: torch.device) -> np.ndarray:
    """
    Resize the array to match the target spacing.

    Args:
        array (torch.Tensor): Input 3D tensor
        current_spacing (tuple): Current voxel spacing (z, x, y)
        target_spacing (tuple): Target voxel spacing (z, x, y)
        device (torch.device): GPU device to use for computation
    
    Returns:
    np.ndarray: Resized array.
    """
    
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [curr / target for curr, target in zip(current_spacing, target_spacing)]
    new_shape = [int(orig * scale) for orig, scale in zip(original_shape, scaling_factors)]
    
    # Move tensor to specified GPU and resize
    array = array.to(device)
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized_array.cpu().numpy()

def process_file(args: Tuple[str, int, pd.DataFrame]) -> None:
    """
    Process a single NIfTI file.
    Args:
        args (tuple): (file path, GPU ID, metadata DataFrame)
    """
    
    file_path, gpu_id, df = args
    device = torch.device(f'cuda:{gpu_id}')
    
    # read file
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    # extract necessary information from metadata
    file_name = os.path.basename(file_path)

    row = df[df['VolumeName'] == file_name]

    # HU conversion parameters
    slope = float(row["RescaleSlope"])
    intercept = float(row["RescaleIntercept"])

    # Spacing information
    xy_spacing = float(row["XYSpacing"][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"])

    current_spacing = (z_spacing, xy_spacing, xy_spacing)
    target_spacing = (ProcessingConfig.target_z_spacing, 
                     ProcessingConfig.target_x_spacing, 
                     ProcessingConfig.target_y_spacing)

    # HU conversion and normalization
    img_data = slope * img_data + intercept  # Convert DICOM pixel value to HU. DICOM pixel value is not HU. (HU is corrected by rescale slope and intercept)
    img_data = np.clip(img_data, ProcessingConfig.hu_min, ProcessingConfig.hu_max)
    img_data = (img_data / 1000).astype(np.float32)  # Normalize to -1~1 range

    # Transpose and resize
    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    resized_array = resize_array(tensor, current_spacing, target_spacing, device)

    # save preprocessed data
    save_folder = "/media/compu/SSD NO1/CT-RATE/train_preprocessed/"
    train_num = file_name.split("_")[1]
    folder_path = os.path.join(save_folder, f"train_{train_num}", f"train_{train_num}{file_name.split('_')[2]}")
    os.makedirs(folder_path, exist_ok=True)
    
    save_path = os.path.join(folder_path, f"{file_name.split('.')[0]}.npz")
    np.savez(save_path, resized_array)

def process_gpu_batch(args: Tuple[List[str], int, pd.DataFrame]) -> None:
    """
    Process a batch of files on a specific GPU.
    Args:
        args (tuple): (file list, GPU ID, metadata DataFrame)
    """
    
    files, gpu_id, df = args
    for file_path in tqdm(files, desc=f"GPU {gpu_id}"):
        process_file((file_path, gpu_id, df))

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        exit()

    # Check if we have at least 2 GPUs
    if torch.cuda.device_count() < 4:
        print(f"Not enough GPUs. Required: 4, Available: {torch.cuda.device_count()}")

    # Check if we have at least 1 GPU
    """
     if torch.cuda.device_count() < 1:
         print(f"No GPU available. Required: 1, Available: {torch.cuda.device_count()}")
         exit()
    """

    # set data path: dataset path to preprocess, metadata path
    preprocess_dataset_path = '/data2/CT-RATE/reports/train/data_volumes/dataset/train'
    metadata_path = "/data2/CT-RATE/reports/train/train_metadata.csv"

    # load file list and metadata
    nii_files = read_nii_files(preprocess_dataset_path)
    df = pd.read_csv(metadata_path)

    """ Using GPUs (2 GPUs) """
    # Create process pool for 2 GPUs
    n_files = len(nii_files)
    split_idx = n_files // 2
    gpu_assignments = [
        (nii_files[:split_idx], 2, df),
        (nii_files[split_idx:], 3, df)
    ]

    # Run parallel processing
    with Pool(2) as pool:
        pool.map(process_gpu_batch, gpu_assignments)

    """ Using Single GPU (GPU 0) """
    # Process all files on GPU 0
    """
    print(f"Processing {len(nii_files)} files on GPU 0")
    process_gpu_batch((nii_files, 0, df))
    """