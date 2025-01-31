#%%
import os
import numpy as np
from tqdm import tqdm
from PIL import Image  

cut_year = 2017

pages_path = os.path.join('..', f'DIBCO_DATA_patches_until_{cut_year}', 'train', 'pages')


os.makedirs(os.path.join('..',f'DIBCO_DATA_patches_until_{cut_year}_ornaments', 'train', 'pages'), exist_ok=True)
os.makedirs(os.path.join('..',f'DIBCO_DATA_patches_until_{cut_year}_ornaments', 'train', 'masks'), exist_ok=True)

for page in tqdm(os.listdir(pages_path)):
    page_path = os.path.join(pages_path, page)
    page_img = Image.open(page_path)
    page_img = np.array(page_img)
    page_mask = np.zeros_like(page_img)[:,:,0]
    page_mask = Image.fromarray(page_mask)
    Image.fromarray(page_img).save(os.path.join('..',f'DIBCO_DATA_patches_until_{cut_year}_ornaments', 'train', 'pages', page))
    page_mask.save(os.path.join('..',f'DIBCO_DATA_patches_until_{cut_year}_ornaments', 'train', 'masks', page))
# %%
