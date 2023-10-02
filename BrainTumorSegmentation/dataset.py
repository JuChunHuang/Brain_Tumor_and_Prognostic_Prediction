from pathlib import Path

import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
    
    @staticmethod
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        not_included = ["0.npy", "1.npy", "2.npy", "3.npy", "4.npy", "5.npy", "6.npy", "7.npy",
        "8.npy", "9.npy", "10.npy", "11.npy", "12.npy", "13.npy", "14.npy", "15.npy", "16.npy",
        "17.npy", "18.npy", "19.npy", "20.npy", "154.npy", "153.npy", "152.npy", "151.npy", 
        "150.npy", "149.npy", "148.npy", "147.npy", "146.npy", "145.npy"]
        for subject in root.glob("*"):   # Iterate over the subjects
            slice_path = subject/"data"  # Get the slices for current subject
            for slice in slice_path.glob("*.npy"):
                if slice not in not_included:
                    files.append(slice)
                
        return files
    
    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("data")] = "mask"
        
        return Path(*parts)

    def augment(self, slice, mask):
        """
        Augments slice and segmentation mask in the exact same way
        """
        random_seed = torch.randint(0, 1000000, (1,)).item()
        imgaug.seed(random_seed)

        mask = mask.astype(np.int32)
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr().round().astype(int)
        
        return slice_aug, mask_aug
    
    def pad(self, slice, mask):
        """
        Change the image size (UCSF-PDGM) from (240x240) to (256x256)
        """
        slice_pad = np.pad(slice, ((8, 8),(8, 8)), "constant")
        mask_pad = np.pad(mask, ((8, 8),(8, 8)), "constant")

        return slice_pad, mask_pad
    
    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)
    
    def convert(self, mask):
        """
        Convert all the subregions labels to 1
        """
        res = mask

        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j]>0:
                    res[i][j] = 1
        
        return res


    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slice = np.load(file_path).astype(np.float32)  # Convert to float for torch
        mask = np.load(mask_path).round().astype(int)
        
        if self.augment_params:
            slice, mask = self.augment(slice, mask)

        slice, mask = self.pad(slice, mask)

        return np.expand_dims(slice, 0), np.expand_dims(mask, 0)