#%% MOVE THE PAGES AND THE MASKS OF ALL THE READ-BAD DATASET TO A SINGLE FOLDER CALLED dataset_text_extraction
import os
import shutil
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(destination_path)

output_path = os.path.join(destination_path,'dataset_text_extraction')

os.makedirs(os.path.join(output_path, 'train', 'pages'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test', 'pages'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test', 'masks'), exist_ok=True)

src_path = os.path.join(destination_path,'READ-BAD')

for root, dirs, files in os.walk(src_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            if 'gt' in root:
                if 'training' in root or 'validation' in root:
                    shutil.copyfile(os.path.join(root, file), os.path.join(output_path, 'train', 'masks', file))
                elif 'public-test' in root:
                    shutil.copyfile(os.path.join(root, file), os.path.join(output_path, 'test', 'masks', file))
            else:
                if 'training' in root or 'validation' in root:
                    shutil.copyfile(os.path.join(root, file), os.path.join(output_path, 'train', 'pages', file))
                elif 'public-test' in root:
                    shutil.copyfile(os.path.join(root, file), os.path.join(output_path, 'test', 'pages', file))

# %% KEEP JUST 5 PAGES AND MASKS IN THE TEST FOLDER. MOVE THE REST TO THE TRAIN FOLDER
import os
import shutil
output_path = os.path.join(destination_path,'dataset_text_extraction')

while len(os.listdir(os.path.join(output_path,'test','masks'))) > 5:
    img_name = os.listdir(os.path.join(output_path,'test','masks'))[0]
    shutil.move(os.path.join(output_path, 'test', 'masks', img_name), os.path.join(output_path, 'train', 'masks'))

while len(os.listdir(os.path.join(output_path,'test','pages'))) > 5:
    img_name = os.listdir(os.path.join(output_path,'test','pages'))[0]
    shutil.move(os.path.join(output_path, 'test', 'pages', img_name), os.path.join(output_path, 'train', 'pages'))

#%% LEVERAGE THE INFORMATION GIVEN FROM THE ZEROTH AND THE THIRD CHANNELS OF THE
# MASKS TO CREATE A BINARY MASK. SAVE THE NEW MASKS IN dataset_text_extraction
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

sets = ['train', 'test']
output_path = os.path.join(destination_path,'dataset_text_extraction')

for set in sets:
    masks_path = os.path.join(output_path, set, 'masks')
    list_of_masks = os.listdir(masks_path)
    for mask in tqdm(list_of_masks):
        mask_path = os.path.join(masks_path, mask)
        mask = np.array(Image.open(mask_path))
        mask_zero_channel = mask[:,:,0]
        mask_second_channel = mask[:,:,2]
        mask_second_channel[mask_second_channel==1] = 0
        mask_zero_channel[mask_second_channel==0] = 255
        mask_zero_channel[mask_zero_channel==128] = 255
        mask = mask_zero_channel
        os.remove(mask_path)
        Image.fromarray(mask).save(mask_path)

#%%
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# img_RGB = Image.open(os.path.join('Napoli_Biblioteca_dei_Girolamini_CF_2_16_Filippino','CNMD0000263308_0022_Carta_8v.jpg'))
# img_HSV = np.array(img_RGB.convert('HSV'))
# img_RGB = np.array(img_RGB)

# fig, axs = plt.subplots(2,3, figsize=(10,10))
# axs = axs.ravel()
# axs[0].imshow(img_RGB[:,:,0])
# axs[1].imshow(img_RGB[:,:,1])
# axs[2].imshow(img_RGB[:,:,2])
# axs[3].imshow(img_HSV[:,:,0])
# axs[4].imshow(img_HSV[:,:,1])
# axs[5].imshow(img_HSV[:,:,2])

# plt.show()

# %% SPLIT THE PAGES AND MASKS INTO PATCHES OF 400x400 AND SAVE THEM IN dataset_text_extraction_patches
import os
import shutil
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(destination_path)
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil 

transform = transforms.ToTensor()

sets = ['train','test']
img_types = ['pages','masks']

for set in sets:
    for img_type in img_types:
        src_path = os.path.join(destination_path,'dataset_text_extraction',set, img_type)
        output_path = os.path.join(destination_path,'dataset_text_extraction_patches',set, img_type)
        os.makedirs(output_path, exist_ok=True)

        for img_name in tqdm(os.listdir(src_path)):
            img_path = os.path.join(src_path, img_name)
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)
            aggregation_sampling = split_aggregation_sampling(img, 400, 400, 1, device='mps')
            for i, patch in enumerate(aggregation_sampling.patches_lr):
                patch = patch.squeeze(0).cpu().permute(1,2,0).numpy()
                if img_type=='pages':
                    patch = Image.fromarray((patch*255).astype(np.uint8))
                elif img_type=='masks':
                    patch = Image.fromarray((patch.squeeze(2)*255).astype(np.uint8))
                patch.save(os.path.join(output_path, img_name.split('.')[0] + f'_{i}.png'))

shutil.rmtree(os.path.join(destination_path,'dataset_text_extraction'))
# %%
