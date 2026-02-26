"""
Simplified PIV Dataset - no JSON manifests, no args objects.

Expects files named:
    {name}_img1.{ext}
    {name}_img2.{ext}  
    {name}_flow.flo
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

import flow_transforms as f_transforms
from utils import read_flow

# Alternative for relative imports if running as part of src package:
# from . import flow_transforms as f_transforms
# from .utils_plot import read_flow

class TransformSubset(Dataset):
    """Subset with its own transform (since Subset doesn't support per-split transforms)."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img1_path, img2_path, flow_path = self.dataset.samples[self.indices[idx]]
        
        img1 = Image.open(img1_path)#.convert('RGB')
        img2 = Image.open(img2_path)#.convert('RGB')
        flow = read_flow(flow_path)
        
        imgs = np.stack([img1, img2], axis=0, dtype = np.float32)
        flow = np.stack([flow[:,:,0],flow[:,:,1]], axis=0)
        
        # datatr = self.transform(*data)
        return imgs, flow

class PIVDataset(Dataset):
    """
    Simple PIV dataset compatible with f_transforms.
    
    Args:
        root: Root directory containing subdirectories (DNS_turbulence, JHTDB_channel, etc.)
        subsets: List of subdirectory names to include, or None for all
        ext: Image extension (default: 'tif')
        crop_size: (H, W) for cropping
        transform: f_transforms.Compose object (overrides default)
        is_train: Use random crops if True, center crops if False
    
    Expected structure:
        root/
        ├── DNS_turbulence/
        │   ├── DNS_turbulence_00500_img1.tif
        │   ├── DNS_turbulence_00500_img2.tif
        │   └── DNS_turbulence_00500_flow.flo
        ├── JHTDB_channel/
        │   └── ...
    """
    
    def __init__(
        self,
        root: str,
        subsets: list = None,
        ext: str = 'tif',
        crop_size: tuple = (256, 256),
        transform=None,
        is_train: bool = True,
    ):
        self.crop_size = crop_size
        self.transform = transform
        self.is_train = is_train
        self.samples = self._discover_files(root, subsets, ext)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root}")
        
        print(f"Found {len(self.samples)} samples")
    
    def _discover_files(self, root: str, subsets: list, ext: str) -> list:
        """Find all (img1, img2, flow) triplets across subdirectories."""
        root = Path(root)
        samples = []
        
        # Get subdirectories to scan
        if subsets is not None:
            dirs = [root / s for s in subsets]
        else:
            dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if not dirs:
            dirs = [root]
        
        for subdir in sorted(dirs):
            if not subdir.exists():
                print(f"Warning: {subdir} does not exist, skipping")
                continue
                
            for flo_path in sorted(subdir.glob('*_flow.flo')):
                # DNS_turbulence_00500_flow.flo -> DNS_turbulence_00500
                name = flo_path.stem.rsplit('_flow', 1)[0]
                img1 = subdir / f'{name}_img1.{ext}'
                img2 = subdir / f'{name}_img2.{ext}'
                
                if img1.exists() and img2.exists():
                    samples.append((str(img1), str(img2), str(flo_path)))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img1_path, img2_path, flow_path = self.samples[idx]
        
        # Load as PIL Image (RGB) - what f_transforms expects
        img1 = Image.open(img1_path)#.convert('RGB')
        img2 = Image.open(img2_path)#.convert('RGB')
        flow = read_flow(flow_path)
        
        # Format expected by f_transforms: [[img1, img2], [flow]]
        data = [[img1, img2], [flow]]
        
        # Apply transform
        if self.transform is not None:
            transformer = self.transform
        else:
            crop_type = 'rand' if self.is_train else 'center'
            transformer = f_transforms.Compose([
                f_transforms.Crop(self.crop_size, crop_type=crop_type),
                f_transforms.ModToTensor(),
            ])
        
        return tuple(transformer(*data))


# === Inference: just load image pairs, no flow needed ===

class PIVInference(Dataset):
    """For inference - just image pairs, no ground truth flow."""
    
    def __init__(self, root: str, subsets: list = None, ext: str = 'tif'):
        root = Path(root)
        self.samples = []
        
        # Get subdirectories to scan
        if subsets is not None:
            dirs = [root / s for s in subsets]
        else:
            dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # If root itself contains images (no subdirs), scan root directly
        if not dirs:
            dirs = [root]
        
        for subdir in sorted(dirs):
            if not subdir.exists():
                continue
            for img1 in sorted(subdir.glob(f'*_img1.{ext}')):
                name = img1.stem.rsplit('_img1', 1)[0]
                img2 = subdir / f'{name}_img2.{ext}'
                if img2.exists():
                    self.samples.append((str(img1), str(img2), name))
        
        if not self.samples:
            raise ValueError(f"No image pairs found in {root}")
        
        print(f"Found {len(self.samples)} image pairs for inference")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img1_path, img2_path, name = self.samples[idx]
        
        img1 = np.array(Image.open(img1_path).convert('RGB'), dtype=np.float32) / 255.0
        img2 = np.array(Image.open(img2_path).convert('RGB'), dtype=np.float32) / 255.0
        
        # Keep original size - adaptive resizing handled in inference function
        img1 = torch.from_numpy(img1.transpose(2, 0, 1))
        img2 = torch.from_numpy(img2.transpose(2, 0, 1))
        
        return img1, img2, name


# === Usage ===

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Option 1: Use all subdirectories with auto-split
    train_data, val_data, test_data = make_splits(
        root='/path/to/piv_data',
        ext='tif',
        crop_size=(256, 256),
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42,
    )
    
    # Option 2: Use specific subsets only
    train_data, val_data, test_data = make_splits(
        root='/path/to/piv_data',
        subsets=['DNS_turbulence', 'JHTDB_channel', 'SQG'],
        crop_size=(256, 256),
    )
    
    # Option 3: Manual dataset creation with specific subsets
    train_tf, val_tf = get_transform((256, 256))
    train_data = PIVDataset(
        root='/path/to/piv_data',
        subsets=['DNS_turbulence', 'JHTDB_channel'],
        transform=train_tf,
    )
    
    # Option 4: Inference (no ground truth)
    test_data = PIVInference('/path/to/piv_data', subsets=['cylinder'])
    
    # DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    
    for (imgs, flows) in train_loader:
        print(f"imgs: {imgs.shape}, flows: {flows.shape}")
        break
