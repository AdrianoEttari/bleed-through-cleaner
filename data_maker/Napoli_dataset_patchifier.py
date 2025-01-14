#%%
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
import shutil 
from concurrent.futures import ThreadPoolExecutor
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(destination_path)
from Aggregation_Sampling import split_aggregation_sampling

destination_path = os.path.join(destination_path, "Napoli_patches")
os.makedirs(destination_path, exist_ok=True)

transform = transforms.ToTensor()

source_path = os.path.join(destination_path, "Napoli_Biblioteca_dei_Girolamini_CF_2_16_Filippino")

# Function to process a single image
def process_image(img_name, source_path, destination_path, transform, split_aggregation_sampling):
    img_path = os.path.join(source_path, img_name)
    if img_path.endswith('.jpg') or img_path.endswith('.png'):
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0)
        aggregation_sampling = split_aggregation_sampling(img, 400, 400, 1, device='cpu')
        
        # Save the patches
        for i, patch in enumerate(aggregation_sampling.patches_lr):
            patch = patch.squeeze(0).cpu().permute(1, 2, 0).numpy()
            patch = Image.fromarray((patch * 255).astype(np.uint8))
            patch.save(os.path.join(destination_path, img_name.split('.')[0] + f'_{i}.png'))

# Function to process images in parallel
def process_images_in_parallel(source_path, destination_path, transform, split_aggregation_sampling, max_workers=8):
    img_names = [img_name for img_name in os.listdir(source_path) if img_name.endswith(('.jpg', '.png'))]
    
    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda img_name: process_image(img_name, source_path, destination_path, 
                                                             transform, split_aggregation_sampling), 
                               img_names), total=len(img_names)))

# Assuming you already have your transform and split_aggregation_sampling functions
process_images_in_parallel(source_path, destination_path, transform, split_aggregation_sampling, max_workers=8)
# %%
