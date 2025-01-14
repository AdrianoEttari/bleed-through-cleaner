#%% PRE-PROCESSING OF THE PAGE_DETECTION_DATA FOLDER
import os
import shutil
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(destination_path)

data_path = 'pages'
shutil.move(destination_path, data_path, 'page_detection_data')
os.remove(os.path.join(destination_path,'page_detection_data', 'classes.txt'))
shutil.move(os.path.join(destination_path,'page_detection_data', 'test_a1'), os.path.join(destination_path,'page_detection_data', 'test'))
shutil.move(os.path.join(destination_path,'page_detection_data', 'val_a1'), os.path.join(destination_path,'page_detection_data', 'val'))

sets = ['train', 'test', 'val']
for set in sets:
    shutil.move(os.path.join(destination_path,'page_detection_data', set, 'labels'), os.path.join(destination_path,'page_detection_data', set, 'masks'))
    shutil.move(os.path.join(destination_path,'page_detection_data', set, 'images'), os.path.join(destination_path,'page_detection_data', set, 'pages'))


# %% KEEP JUST 5 PAGES AND MASKS IN THE TEST FOLDER. MOVE THE REST TO THE TRAIN FOLDER
import os
import shutil
output_folder_name = 'page_detection_data'

while len(os.listdir(os.path.join(destination_path,output_folder_name,'test','masks'))) > 5:
    img_name = os.listdir(os.path.join(destination_path,output_folder_name,'test','masks'))[0]
    shutil.move(os.path.join(destination_path,output_folder_name, 'test', 'masks', img_name), os.path.join(destination_path,output_folder_name,'train','masks'))

while len(os.listdir(os.path.join(destination_path,output_folder_name,'test','pages'))) > 5:
    img_name = os.listdir(os.path.join(destination_path,output_folder_name,'test','pages'))[0]
    shutil.move(os.path.join(destination_path,output_folder_name, 'test', 'pages', img_name), os.path.join(destination_path,output_folder_name,'train','pages'))

if os.path.exists(os.path.join(destination_path,output_folder_name,'val')):
    for img_name in os.listdir(os.path.join(destination_path,output_folder_name,'val','masks')):
        shutil.move(os.path.join(destination_path,output_folder_name, 'val', 'masks', img_name), os.path.join(destination_path,output_folder_name,'train','masks'))

    for img_name in os.listdir(os.path.join(destination_path,output_folder_name,'val','pages')):
        shutil.move(os.path.join(destination_path,output_folder_name, 'val', 'pages', img_name), os.path.join(destination_path,output_folder_name,'train','pages'))
    shutil.rmtree(os.path.join(destination_path,output_folder_name,'val'))

#%% SOME INPUT IMAGES HAVE ONLY ONE CHANNEL. WE NEED TO CONVERT THEM TO RGB
import os
import shutil
from PIL import Image
import numpy as np

sets = ['train', 'test']
output_folder_name = 'page_detection_data'

for set in sets:
    for i, img_path in enumerate(os.listdir(os.path.join(destination_path,output_folder_name, set, 'pages'))):
        img = np.array(Image.open(os.path.join(destination_path,output_folder_name, set, 'pages', img_path)))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)
            os.remove(os.path.join(destination_path,output_folder_name, set, 'pages', img_path))
            img.save(os.path.join(destination_path,output_folder_name, set, 'pages', img_path))


# %% RESIZE THE IMAGES SO THAT THEY HAVE THE SAME NUMBER OF PIXELS (NOT SAME SHAPE) AND THE SAME ASPECT RATIO.
# YOU CANNOT SPLIT THEM INTO PATCHES BECAUSE WE NEED THE WHOLE IMAGE.
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil 

target_pixels = 6*10**5

sets = ['train', 'test']
img_types = ['pages', 'masks']

src_path = os.path.join(destination_path,'page_detection_data')

for set in sets:
    for img_type in img_types:
        for img_path in tqdm(os.listdir(os.path.join(src_path, set, img_type))):
            img = Image.open(os.path.join(src_path, set, img_type, img_path))
            number_of_pixels = img.size[0]*img.size[1]
            scale = np.sqrt(target_pixels/number_of_pixels)
            aspect_ratio = img.size[0]/img.size[1]
            img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
            os.remove(os.path.join(src_path, set, img_type, img_path))
            img.save(os.path.join(src_path, set, img_type, img_path))
            assert np.abs(img.size[0]/img.size[1] - aspect_ratio) < 0.1

#%% MAKE THE MASK HAVE JUST ONE CHANNEL (THE OTHER TWO ARE FULL OF ZEROS)
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

sets = ['train', 'test']
img_types = ['masks']

src_path = os.path.join(destination_path,'page_detection_data')

for set in sets:
    for mask_name in tqdm(os.listdir(os.path.join(src_path,set,'masks'))):
        mask_path = os.path.join(src_path,set,'masks',mask_name)
        mask = np.array(Image.open(mask_path))
        mask = mask[:,:,0]
        os.remove(mask_path)
        Image.fromarray(mask).save(mask_path)

# %% MAKE THE MASK BINARY
import os
import shutil
destination_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(destination_path)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

data_path = 'page_detection_data'
train_path = os.path.join('..', data_path, 'train')
test_path = os.path.join('..', data_path, 'test')

for mask in tqdm(os.listdir(os.path.join(train_path,'masks'))):
    mask_path = os.path.join(train_path,'masks',mask)
    mask = np.array(Image.open(mask_path))
    mask = np.where(mask>0, 255, 0).astype(np.uint8)
    os.remove(mask_path)
    Image.fromarray(mask).save(mask_path)

for mask in tqdm(os.listdir(os.path.join(test_path,'masks'))):
    mask_path = os.path.join(test_path,'masks',mask)
    mask = np.array(Image.open(mask_path))
    mask = np.where(mask>0, 255, 0).astype(np.uint8)
    os.remove(mask_path)
    Image.fromarray(mask).save(mask_path)

# %%
