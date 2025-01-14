#%%
import os
import shutil
from tqdm import tqdm

destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

pages_path = os.path.join(destination_path,"ornament_detection_data", "pages")
masks_path = os.path.join(destination_path,"ornament_detection_data", "masks")

os.makedirs(os.path.join(destination_path,"ornament_detection_data"), exist_ok=True)
os.makedirs(pages_path, exist_ok=True)
os.makedirs(masks_path, exist_ok=True)

data_path = '/Users/adrianoettari/Desktop/Synchronized/LAVORO/labels_ornament_extraction'
# data_path = os.path.join("..", "labels_ornament_extraction")

for i in tqdm(range(len(os.listdir(data_path)))):
    if os.path.isdir(os.path.join(data_path, os.listdir(data_path)[i])):
        subfolder_name = os.listdir(data_path)[i]
        if subfolder_name.endswith("_json"):
            subfolder_path = os.path.join(data_path, subfolder_name)
            new_file_name = subfolder_name.replace("_json",".png")
            shutil.copy(os.path.join(subfolder_path, "img.png"), os.path.join(pages_path, new_file_name))
            shutil.copy(os.path.join(subfolder_path, "label.png"), os.path.join(masks_path, new_file_name))

# %% SPLIT THE PAGES AND MASKS INTO PATCHES OF 400x400 AND SAVE THEM IN ornament_detection_data_patches
import sys
import os
sys.path.append(destination_path)
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil 
# import matplotlib.pyplot as plt

transform = transforms.ToTensor()

sets = ['train']
img_types = ['pages','masks']
# img_types = ['masks']
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(destination_path,"ornament_detection_data")

for set in sets:
    for img_type in img_types:
        src_path = os.path.join(data_path, img_type)
        destination_path = os.path.join(data_path+"_patches", set, img_type)
        os.makedirs(destination_path, exist_ok=True)

        for img_name in tqdm(os.listdir(src_path)):
            img_path = os.path.join(src_path, img_name)
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)
            aggregation_sampling = split_aggregation_sampling(img, 400, 400, 1, device='cpu')
            for i, patch in enumerate(aggregation_sampling.patches_lr):
                if img_type == 'masks':
                    patch = patch.squeeze(0).cpu().permute(1,2,0).numpy().squeeze(2)
                    # fig, axs = plt.subplots(1,2)
                    # axs[0].imshow(patch)
                    # _max = patch.max()
                    # _min = patch.min()
                    # axs[0].set_title(f'Min: {_min}, Max: {_max}')
                    if len(np.unique(patch))>1 or np.min(patch) != 0:
                        patch = patch/patch.max() # ATTENTION: DON'T USE (X-MIN)/(X-MAX) BECAUSE IT IS POSSIBLE THAT ONE PATCH HAS JUST A UNIQUE VALUE WHICH IS AN ORNAMENT PIXEL ONE AND SO THE PATCH WOULD NOT BE CORRECTLY STANDARDIZED. INSTEAD, FIX THE MINIMUM TO 0 LIKE HERE.
                    # _max = patch.max()
                    # _min = patch.min()
                    # axs[1].imshow(patch)
                    # axs[1].set_title(f'Min: {_min}, Max: {_max}')
                    # plt.show()
                    # import ipdb; ipdb.set_trace()
                elif img_type == 'pages':
                    patch = patch.squeeze(0).cpu().permute(1,2,0).numpy()
                patch = Image.fromarray((patch*255).astype(np.uint8))
                patch.save(os.path.join(destination_path, img_name.split('.')[0] + f'_{i}.png'))

shutil.rmtree(data_path)
# %%
import numpy as np
from PIL import Image

counter_patches_with_ornaments = 0
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mask_folder = os.path.join(destination_path,'ornament_detection_data_patches','train','masks')

for mask in os.listdir(mask_folder):
    img = np.array(Image.open(os.path.join(mask_folder, mask)))
    if np.sum(img) > 0:
        counter_patches_with_ornaments += 1
    
print(f"Percentage of patches with ornament = {counter_patches_with_ornaments/len(os.listdir(mask_folder))*100:.1f}%")
    
#%%
from PIL import Image
import numpy as np

np.array(Image.open(os.path.join(destination_path,'ornament_detection_data_patches','train','masks','CNMD0000250043_0012_Carta_1r_0.png'))).shape
# %% 
# IF YOU DON'T STANDARDIZE DURING THE PRODUCTION OF THE MASK PATCHES (patch = (patch-patch.min())/(patch.max()-patch.min()))
# YOU END UP WITH MASK PATCHES THAT HAVE A MAXIMUM VALUE OF 1, INSTEAD OF 255; THEY ARE NOT CORRECTLY VISIBLE WITH PIL AND 
# IF YOU SEE THEM IN THE FOLDER (THEY'RE CORRECTLY VISIBLE JUST WITH MATPLOTLIB).

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# mask_1 = os.path.join('..','ornament_detection_data_patches','train','masks')
# mask_255 = os.path.join('..','ornament_detection_data_patches','train','masks_255')
# pages = os.path.join('..','ornament_detection_data_patches','train','pages')

# for file_name in os.listdir(mask_1)[:1000]:
#     img_1 = np.array(Image.open(os.path.join(mask_1, file_name)))
#     img_255 = np.array(Image.open(os.path.join(mask_255, file_name)))
#     page = np.array(Image.open(os.path.join(pages, file_name)))
#     is_ornament_1 = np.sum(img_1)>=1
#     is_ornament_255 = np.sum(img_255)>=1
#     if is_ornament_1 != is_ornament_255:
#         fig, axs = plt.subplots(1,3)
#         print(np.unique(img_1))
#         print(np.unique(img_255))
#         axs[0].imshow(img_1)
#         axs[1].imshow(img_255)
#         axs[2].imshow(page)
#         plt.show()


## %%
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# from tqdm import tqdm

# visible_path = os.path.join('ornament_detection_data_patches_visible','train')
# not_visible_path = os.path.join('ornament_detection_data_patches_not_visible','train')

# for i, img_name in tqdm(enumerate(os.listdir(os.path.join(visible_path, 'masks')))):
#     # print(i)
#     # fig, axs = plt.subplots(2,2)
#     # axs = axs.ravel()
#     # axs[0].imshow(np.array(Image.open(os.path.join(visible_path, 'pages', img_name))))
#     # axs[0].axis('off')
#     # axs[0].set_title('Visible (page)')
#     # axs[1].imshow(np.array(Image.open(os.path.join(visible_path, 'masks', img_name))))
#     # axs[1].axis('off')
#     # _min_visible = np.min(np.array(Image.open(os.path.join(visible_path, 'masks', img_name))))
#     # _max_visible = np.max(np.array(Image.open(os.path.join(visible_path, 'masks', img_name))))
#     # axs[1].set_title(f'Visible (mask) {_min_visible} {_max_visible}')
#     # axs[2].imshow(np.array(Image.open(os.path.join(not_visible_path, 'pages', img_name))))
#     # axs[2].axis('off')
#     # axs[2].set_title('Not visible (page)')
#     # axs[3].imshow(np.array(Image.open(os.path.join(not_visible_path, 'masks', img_name))))
#     # axs[3].axis('off')
#     # _min_not_visible = np.min(np.array(Image.open(os.path.join(not_visible_path, 'masks', img_name))))
#     # _max_not_visible = np.max(np.array(Image.open(os.path.join(not_visible_path, 'masks', img_name))))
#     # axs[3].set_title(f'Not visible (mask) {_min_not_visible} {_max_not_visible}')
#     # plt.show()

#     mask_visible = np.array(Image.open(os.path.join(visible_path, 'masks', img_name)))
#     mask_not_visible = np.array(Image.open(os.path.join(not_visible_path, 'masks', img_name)))
    
#     comparison = mask_not_visible*255 == mask_visible
#     if not comparison.all():
#         print(img_name)