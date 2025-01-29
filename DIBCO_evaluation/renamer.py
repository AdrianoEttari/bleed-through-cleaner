#%% DIBCO 2013 input
from PIL import Image
import os

dibco2013_input_path = os.path.join('..', 'DIBCO_DATA', 'DIBCO2013')
for filename in os.listdir(dibco2013_input_path):
    if filename.lower().endswith('.tiff'):
        input_path = os.path.join(dibco2013_input_path, filename)
        output_filename = filename.replace('.tiff', '.bmp')
        output_path = os.path.join(dibco2013_input_path, output_filename)
        with Image.open(input_path) as img:
            img.save(output_path, 'BMP')
        os.remove(input_path)

# %% DIBCO 2013 ground truth
from PIL import Image
import os

dibco2013_GT_path = os.path.join('..', 'DIBCO_DATA', 'DIBCO2013_GT')
for filename in os.listdir(dibco2013_GT_path):
    if filename.lower().endswith('.tiff'):
        input_path = os.path.join(dibco2013_GT_path, filename)
        output_filename = filename.replace('estGT.tiff', 'gt.bmp')
        output_path = os.path.join(dibco2013_GT_path, output_filename)
        with Image.open(input_path) as img:
            img.save(output_path, 'BMP')
        os.remove(input_path)
#%% DIBCO 2014 input
from PIL import Image
import os

dibco2014_input_path = os.path.join('..', 'DIBCO_DATA', 'DIBCO2014')
for filename in os.listdir(dibco2014_input_path):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(dibco2014_input_path, filename)
        output_filename = filename.replace('.png', '.bmp')
        output_path = os.path.join(dibco2014_input_path, output_filename)
        with Image.open(input_path) as img:
            img.save(output_path, 'BMP')
        os.remove(input_path)

# %% DIBCO 2014 ground truth
from PIL import Image
import os

dibco2014_GT_path = os.path.join('..', 'DIBCO_DATA', 'DIBCO2014_GT')
for filename in os.listdir(dibco2014_GT_path):
    if filename.lower().endswith('.tiff'):
        input_path = os.path.join(dibco2014_GT_path, filename)
        output_filename = filename.replace('estGT.tiff', 'gt.bmp')
        output_path = os.path.join(dibco2014_GT_path, output_filename)
        with Image.open(input_path) as img:
            img.save(output_path, 'BMP')
        os.remove(input_path)
# %%

