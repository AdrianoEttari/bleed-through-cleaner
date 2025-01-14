#%%
import numpy as np
from PIL import Image
import os

origin_folder_path = os.path.join('..', 'Bleed_Through_Database')
files_name = os.listdir(origin_folder_path)
os.makedirs(os.path.join('..', 'Bleed_Through_Database', 'masks'), exist_ok=True)
os.makedirs(os.path.join('..', 'Bleed_Through_Database', 'rgb'), exist_ok=True)
os.makedirs(os.path.join('..', 'Bleed_Through_Database', 'gray'), exist_ok=True)
for file_name in files_name:
    if not os.path.isdir(os.path.join(origin_folder_path, file_name)):
        if 'rgb' in file_name:
            Image.open(os.path.join(origin_folder_path, file_name)).save(os.path.join('..', 'Bleed_Through_Database', 'rgb', file_name))
        elif 'gt' in file_name:
            Image.open(os.path.join(origin_folder_path, file_name)).save(os.path.join('..', 'Bleed_Through_Database', 'masks', file_name))
        else:
            Image.open(os.path.join(origin_folder_path, file_name)).save(os.path.join('..', 'Bleed_Through_Database', 'gray', file_name))
        os.remove(os.path.join(origin_folder_path, file_name))
# %%
