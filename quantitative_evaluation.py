#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import weighted_total_error_function
import os
from scipy import stats

DIBCO_year = 2019
input_data_folder = os.path.join('DIBCO_DATA',f'DIBCO{DIBCO_year}')
real_data_folder = os.path.join('DIBCO_DATA',f'DIBCO{DIBCO_year}_GT')
real_images = os.listdir(real_data_folder)
input_images = os.listdir(input_data_folder)
predictions_folder = os.path.join("DIBCO_evaluation","DIBCO_DATA_until_2019_pred_400_epochs","Images",f"DIBCO{DIBCO_year}")

# for file in os.listdir(predictions_folder):
#     os.rename(os.path.join(predictions_folder, file), os.path.join(predictions_folder,file.replace('.bmp_', '_')))

foreground_errors, background_errors, weighted_total_errors = [], [], []

extension_input_images = input_images[0].split('.')[1]
indexes_input_images = [int(img_name.split('.')[0]) for img_name in input_images]

for idx in sorted(indexes_input_images):
    img_name = str(idx)+ '.' + extension_input_images
    real_img = np.array(Image.open(os.path.join(real_data_folder, img_name)), dtype=float)
    if len(real_img.shape) == 3:
        real_img = real_img[:,:,0]
    if np.max(real_img)==255:
        real_img /= 255
    try:
        pred_img_name = img_name.split('_')[0]+'_BLEED_THROUGH_MASK.png'
        prediction_img = np.array(Image.open(os.path.join(predictions_folder, pred_img_name)))/255
        foreground_error_ratio_finetuning, background_error_ratio_finetuning, weighted_total_error_finetuning = weighted_total_error_function(real_img, prediction_img)
    except:
        pred_img_name = img_name.split('.')[0]+'_BLEED_THROUGH_MASK.png'
        prediction_img = np.array(Image.open(os.path.join(predictions_folder, pred_img_name)))/255
        foreground_error_ratio_finetuning, background_error_ratio_finetuning, weighted_total_error_finetuning = weighted_total_error_function(real_img, prediction_img)

    foreground_errors.append(foreground_error_ratio_finetuning)
    background_errors.append(background_error_ratio_finetuning)
    weighted_total_errors.append(weighted_total_error_finetuning)

    try:
        input_img = np.array(Image.open(os.path.join(input_data_folder, img_name.split('_')[0]+'.png')))
    except:
        input_img = np.array(Image.open(os.path.join(input_data_folder, img_name)))
    print(f"The foreground error ratio with finetuning of the image {img_name} is {foreground_error_ratio_finetuning:.5f}")
    print(f"The background error ratio with finetuning of the image {img_name} is {background_error_ratio_finetuning:.5f}")
    print(f"The weighted total error with finetuning of the image {img_name} is {weighted_total_error_finetuning:.5f}")
    fig, axs = plt.subplots(1,3, figsize=(10,15))
    axs[0].imshow(input_img)
    axs[0].set_title('Input img')
    axs[1].imshow(real_img,cmap='gray')
    axs[1].set_title('Real mask')
    axs[2].imshow(prediction_img,cmap='gray')
    axs[2].set_title('Prediction mask')
    plt.show()

# Assuming foreground_errors is a NumPy array or list
foreground_median = np.median(foreground_errors)
foreground_mad = stats.median_abs_deviation(foreground_errors, scale=1)

# Compute upper and lower boundaries
upper_boundary = foreground_median + foreground_mad
lower_boundary = foreground_median - foreground_mad

# Plot errors
plt.plot(np.arange(len(foreground_errors)), foreground_errors, marker='o', label='Foreground Error', color='blue')
plt.plot(np.arange(len(background_errors)), background_errors, marker='o', label='Background Error', color='orange')
plt.plot(np.arange(len(weighted_total_errors)), weighted_total_errors, marker='o', label='Weighted Total Error', color='gray')

# Plot median line
plt.axhline(foreground_median, label='Median Foreground Error', linestyle='dashed', linewidth=2, color='blue')

# Plot upper and lower boundary lines
plt.axhline(upper_boundary, linestyle='dotted', linewidth=2, color='blue', label='Upper Bound (Median + MAD)')
plt.axhline(lower_boundary, linestyle='dotted', linewidth=2, color='blue', label='Lower Bound (Median - MAD)')

# Shaded region between upper and lower boundaries
plt.fill_between(np.arange(len(foreground_errors)), lower_boundary, upper_boundary, color='blue', alpha=0.2)

# Labels and legend
plt.xticks(np.arange(len(foreground_errors)))
plt.ylabel("Error (ratio)")
plt.xlabel("Image number")
plt.legend()
plt.show()

# def mad(x):
#     return np.mean([np.abs(value-np.median(x)) for value in x])
# %%
# zero_one = np.zeros_like(real_img)
# zero_one[:200, :750]+=1
# print(weighted_total_error_function(real_img, zero_one))
# fig, axs = plt.subplots(1,2)
# axs[0].imshow(real_img)
# axs[1].imshow(zero_one)
# plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import weighted_total_error_function
import os

input_data_folder = os.path.join('DIBCO_DATA','DIBCO2017')
real_data_folder = os.path.join('DIBCO_DATA','DIBCO2017_GT')
real_images = os.listdir(real_data_folder)
predictions_folder = os.path.join('assets', 'DIBCO_2017')

foreground_errors, background_errors, weighted_total_errors = [], [], []

for img_name in sorted(real_images):
    real_img = np.array(Image.open(os.path.join(real_data_folder, img_name)), dtype=float)
    if len(real_img.shape) == 3:
        real_img = real_img[:,:,0]
    if np.max(real_img)==255:
        real_img /= 255
    pred_img_name = img_name.split('.')[0]+'_BLEED_THROUGH_MASK.png'
    prediction_img = np.array(Image.open(os.path.join(predictions_folder, pred_img_name)))/255
    foreground_error_ratio_finetuning, background_error_ratio_finetuning, weighted_total_error_finetuning = weighted_total_error_function(real_img, prediction_img)

    foreground_errors.append(foreground_error_ratio_finetuning)
    background_errors.append(background_error_ratio_finetuning)
    weighted_total_errors.append(weighted_total_error_finetuning)

    input_img = np.array(Image.open(os.path.join(input_data_folder, img_name.split('_')[0])))
    print(f"The foreground error ratio with finetuning of the image {img_name} is {foreground_error_ratio_finetuning:.5f}")
    print(f"The background error ratio with finetuning of the image {img_name} is {background_error_ratio_finetuning:.5f}")
    print(f"The weighted total error with finetuning of the image {img_name} is {weighted_total_error_finetuning:.5f}")
    fig, axs = plt.subplots(1,3, figsize=(10,15))
    axs[0].imshow(input_img)
    axs[0].set_title('Input img')
    axs[1].imshow(real_img,cmap='gray')
    axs[1].set_title('Real mask')
    axs[2].imshow(prediction_img,cmap='gray')
    axs[2].set_title('Prediction mask')
    plt.show()

plt.plot(np.arange(len(foreground_errors)), foreground_errors, marker='o', label='foreground error', color='blue')
plt.plot(np.arange(len(background_errors)), background_errors, marker='o', label='background error', color='orange')
plt.plot(np.arange(len(weighted_total_errors)), weighted_total_errors, marker='o', label='weighted total error', color='gray')
plt.legend()
plt.show()


# %%
