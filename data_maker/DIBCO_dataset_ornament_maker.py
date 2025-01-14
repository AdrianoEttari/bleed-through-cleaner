#%%
import os
import numpy as np
from tqdm import tqdm
from PIL import Image  

DIBCTO_DATA_patches_ALL = '../DIBCO_DATA_patches_ALL'
pages = os.path.join(DIBCTO_DATA_patches_ALL, 'train', 'pages')

os.makedirs(os.path.join('..','DIBCO_DATA_patches_ALL_ornaments', 'train', 'pages'), exist_ok=True)
os.makedirs(os.path.join('..','DIBCO_DATA_patches_ALL_ornaments', 'train', 'masks'), exist_ok=True)

for page in tqdm(os.listdir(pages)):
    page_path = os.path.join(pages, page)
    page_img = Image.open(page_path)
    page_img = np.array(page_img)
    page_mask = np.zeros_like(page_img)[:,:,0]
    page_mask = Image.fromarray(page_mask)
    Image.fromarray(page_img).save(os.path.join('..','DIBCO_DATA_patches_ALL_ornaments', 'train', 'pages', page))
    page_mask.save(os.path.join('..','DIBCO_DATA_patches_ALL_ornaments', 'train', 'masks', page))
# %%
