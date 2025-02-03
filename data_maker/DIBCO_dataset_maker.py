#%% RENAME THE _gt FILES OF THE GT FOLDERS BY REMOVING _gt
import shutil
import sys
import os
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(destination_path)
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

DIBCO_folder_path = os.path.join('..', 'DIBCO_DATA')

for subfolder in os.listdir(DIBCO_folder_path):
    # if subfolder != 'test':
        if 'GT' in os.path.join(DIBCO_folder_path, subfolder):
            for file_name in os.listdir(os.path.join(DIBCO_folder_path, subfolder)):
                img = Image.open(os.path.join(DIBCO_folder_path, subfolder, file_name))
                new_name = file_name.replace('_gt','')
                # os.remove(os.path.join(DIBCO_folder_path, subfolder, file_name))
                if not os.path.exists(os.path.join(DIBCO_folder_path, subfolder, new_name)):
                    img.save(os.path.join(DIBCO_folder_path, subfolder, new_name))
                    assert os.path.exists(os.path.join(DIBCO_folder_path, subfolder.replace('_GT',''), new_name))

# #%%
# DIBCO2014_GT_tiff_path = '../DIBCO_DATA/test/DIBCO2014_GT/H06_estGT.tiff'
# img_tiff = Image.open(DIBCO2014_GT_tiff_path)

# %% CONSIDER ONLY THE FILES OF THE FOLDER BEFORE THE cut_year AND REMOVE THE FILES WHICH GET THE _gt SUFFIX
transform = transforms.ToTensor()

cut_year = 2019

pages_path = os.path.join('..', f'DIBCO_DATA_patches_until_{cut_year}', 'train', 'pages')
masks_path = os.path.join('..',  f'DIBCO_DATA_patches_until_{cut_year}', 'train', 'masks')
os.makedirs(masks_path, exist_ok=True)
os.makedirs(pages_path, exist_ok=True)

page_file_paths = []
gt_file_paths = []
for dirpath, dirnames, filenames in os.walk(DIBCO_folder_path):
    if os.path.basename(dirpath).startswith('DIBCO2') and int(os.path.basename(dirpath)[5:9]) < cut_year:
        if 'GT' in os.path.basename(dirpath):
                for filename in filenames:
                    if 'gt' not in os.path.basename(filename):
                        gt_file_paths.append(os.path.join(dirpath, filename))
                    else:
                        os.remove(os.path.join(dirpath, filename))
        else:
            for filename in filenames:
                page_file_paths.append(os.path.join(dirpath, filename))
    
page_file_paths = sorted(page_file_paths)
gt_file_paths = sorted(gt_file_paths)
#%% BUILD PATCHES OF 400x400 AND DISCARD THE IMAGE WITH EITHER WIDTH OR HEIGHT LESS THAN 400.
width = 400
height = 400

for j,file_path in tqdm(enumerate(gt_file_paths)):
    img = Image.open(file_path)
    img = transform(img).unsqueeze(0)[:,0,:,:].unsqueeze(0)
    aggregation_sampling = split_aggregation_sampling(img, width, height, 1, device='cpu')
    for i, patch in enumerate(aggregation_sampling.patches_lr):
        patch = patch.squeeze(0).squeeze(0).cpu().permute(0,1).numpy()
        patch = Image.fromarray((patch*255).astype(np.uint8))
        if patch.size[0] != width or patch.size[1] != height:
            print(f'Discarded patch {i} from image {file_path} ')
        else:
            new_file_name = f'img_{j}' + f'patch_{i}.png'
            patch.save(os.path.join(masks_path, new_file_name))

for j,file_path in tqdm(enumerate(page_file_paths)):
    img = Image.open(file_path)
    img = transform(img).unsqueeze(0)
    aggregation_sampling = split_aggregation_sampling(img, width, height, 1, device='cpu')
    for i, patch in enumerate(aggregation_sampling.patches_lr):
        patch = patch.squeeze(0).cpu().permute(1,2,0).numpy()
        patch = Image.fromarray((patch*255).astype(np.uint8))
        if patch.size[0] != width or patch.size[1] != height:
            print(f'Discarded patch {i} from image {file_path} ')
        else:
            new_file_name = f'img_{j}' + f'patch_{i}.png'
            patch.save(os.path.join(pages_path, new_file_name))

# %% CHECK THE NUMBER OF PATCHES
print(len(os.listdir(pages_path)))
print(len(os.listdir(masks_path)))
# %% CHECK THE WIDTH, HEIGHT AND CHANNELS OF THE IMAGES WITH THREE HISTOGRAMS
import shutil
import sys
import os
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(destination_path)
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

cut_year = 2019

pages_path = os.path.join('..', f'DIBCO_DATA_patches_until_{cut_year}', 'train', 'pages')
masks_path = os.path.join('..',  f'DIBCO_DATA_patches_until_{cut_year}', 'train', 'masks')

tot_widths = []
tot_heights = []
tot_channels = []

for page_file_name in os.listdir(pages_path):
            img = np.array(Image.open(os.path.join(pages_path, page_file_name)))
            height, width, channels = img.shape
            if width < 400 or height < 400 or channels != 3:
                print(page_file_name)
            tot_widths.append(width)
            tot_heights.append(height)
            tot_channels.append(channels)

plt.hist(tot_widths, bins=100)
plt.show()
plt.hist(tot_heights, bins=100)
plt.show()
plt.hist(tot_channels, bins=100)
plt.show()





# %%
