
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

def make_holes(image, min_hole_size=25, max_hole_size=100, max_holes=10):
    rgb_image = image[:,:,:3]
    mask = image[:,:,3]
    
    num_holes = np.random.randint(1, max_holes + 1)
    
    h, w, c = rgb_image.shape
    holes_mask = np.zeros((h, w), dtype=np.uint8)


    mask = (mask > 127).astype(np.uint8) # mask is either 0 or 255, so let's make it False or True
    for _ in range(num_holes):
        hole_size = np.random.randint(min_hole_size, max_hole_size + 1)
        y = np.random.randint(0, h - hole_size)
        x = np.random.randint(0, w - hole_size)

        region = mask[y:y+hole_size, x:x+hole_size]
        if region.mean() > 0.8:
            holes_mask[y:y+hole_size, x:x+hole_size]=1
    
    return rgb_image, mask, holes_mask


def normalize(rgb):
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb


class get_data(Dataset):
    '''
    '''
    def __init__(self, root_dir, imgs_folder_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs_folder_name = imgs_folder_name
        self.imgs_filenames = os.listdir(os.path.join(self.root_dir, self.imgs_folder_name))
    
    def __len__(self):
        return len(self.imgs_filenames)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.root_dir, self.imgs_folder_name, self.imgs_filenames[idx])
        image = np.array(Image.open(img_full_path))          # H x W x 4
        rgb, text_mask, holes_mask = make_holes(image)


        # Normalize FIRST
        rgb = normalize(rgb)
        rgb_corrupt = rgb.copy()

        # Fill holes with true "no data" noise
        n = np.sum(holes_mask)
        rgb_corrupt[holes_mask == 1] = np.random.uniform(
            low=0.0, high=1.0, size=(n, 3)
        )

        # Stack inputs
        input_tensor = np.concatenate(
            [
                rgb_corrupt,
                text_mask[..., None].astype(np.float32),
                holes_mask[..., None].astype(np.float32)
            ],
            axis=2
        )


        return (
            torch.tensor(input_tensor).permute(2, 0, 1),
            torch.tensor(rgb).permute(2, 0, 1))