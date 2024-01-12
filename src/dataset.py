import os 
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

#Dataset-V2
class KelpDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, filename_list=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if filename_list is not None:
            self.images = [file for file in os.listdir(image_dir) if file in filename_list]
        else:
            self.images = os.listdir(image_dir)

        #self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace('_satellite.tif', '_kelp.tif'))
            mask = xr.open_rasterio(mask_path, parse_coordinates=True)
            mask = mask.values.squeeze().astype(np.float32)
        else:
            mask = np.zeros((1, 1, 350, 350), dtype=np.float32)
            
        # Open images using xarray
        img = xr.open_rasterio(img_path, parse_coordinates=True)
        #mask = xr.open_rasterio(mask_path, parse_coordinates=True)
        # Bands
        image = img.values.transpose(1, 2, 0).astype(np.float32)
        
        # Normalize each channel independently along the last two dimensions
        min_vals = np.min(image, axis=2, keepdims=True)
        max_vals = np.max(image, axis=2, keepdims=True)
        image = (image - min_vals) / (max_vals - min_vals)
        # Mask
        mask = mask.squeeze().astype(np.float32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Masks
        #mask = mask.astype(np.float32)

        return image, mask