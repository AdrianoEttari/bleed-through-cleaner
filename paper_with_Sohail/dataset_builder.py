#%% IMPORT LIBRARIES
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bleed_through_cleaner import bleed_through_cleaner
# from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
import numpy as np
import json
from patchify import patchify


#%% FIND TEXT
device = "cuda"
patch_size = 1024
stride = 512
batch_size = 8

json_path = "images_with_and_without_bleed_through.json"
models_folder = os.path.join("..","models")

with open(json_path, "r") as f:
    images_with_and_without_bleed_through = json.load(f)

#%% MASKED PATCHES SAVE
# dataset_folder_path = os.path.join("/data1","aettari","dataset_MAGIC_PatchSize"+str(patch_size))
dataset_folder_path = os.path.join("dataset_MAGIC_PatchSize"+str(patch_size))
os.makedirs(dataset_folder_path, exist_ok=True)

already_saved = os.listdir(dataset_folder_path)
transform = transforms.Compose([transforms.ToTensor()])

# root_dataset_path = os.path.join("/data1", "aettari")

for img_path in images_with_and_without_bleed_through["no"]:
    if os.path.basename(img_path.replace(".jpg","_idx_0.png")) in already_saved:
        print(img_path, " already saved.")
        continue
    
    # img_path = img_path.replace("..", ".")
    # img_path = img_path.replace("\\", "/")

    # img_path = os.path.join(root_dataset_path, img_path)
    cleaner = bleed_through_cleaner(img_path, models_folder, False, device)

    image = Image.open(img_path)
    page_filtered_image = np.array(image)
    page_filtered_image_tensor = transform(page_filtered_image).to(device)

    # aggregation_sampling = split_aggregation_sampling(img_lr=page_filtered_image_tensor, patch_size=patch_size, stride=stride,
    #                                                     batch_size=batch_size, magnification_factor=1, device=device, multiple_gpus=False)
    # text_mask, text_GPU_time = cleaner.text_detect(aggregation_sampling, model_name='Residual_attention_UNet_text_extraction')
    
    page_filtered_image, mask, _ = cleaner.bleed_through_finder(page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction')
    
    
    img_mask_concat = np.concatenate([page_filtered_image, mask[:,:,None]], axis=2) 
    patches = patchify(img_mask_concat, (patch_size, patch_size, 4), step=stride)
    patches = patches.reshape(patches.shape[0]*patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5])
    for idx, patch in enumerate(patches):
        filename = os.path.basename(img_path).replace(".jpg", f"_idx_{idx}.png")
        Image.fromarray(patch).save(os.path.join(dataset_folder_path, filename))
    
            
            
        


# %%
