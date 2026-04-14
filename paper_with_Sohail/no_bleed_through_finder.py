#%% LIBRARIES
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

#%% Identify not bleed-through affected area
# Identify the full paths of the images that are not afftected by bleed-through,
# so that we can build a dataset to create back the background.

datasets = ["Napoli_Biblioteca_dei_Girolamini_CF_2_16_Filippino", "Firenze_BibliotecaMediceaLaurenziana_Plut_40_1"]
json_path = "images_with_and_without_bleed_through.json"

if os.path.exists(json_path):
    with open(json_path, "r") as f:
        images_with_and_without_bleed_through = json.load(f)
else:
    images_with_and_without_bleed_through = {"yes":[], "no":[]}

ornament_model_name = "Residual_attention_UNet_ornament_extraction"
text_model_name = "Residual_attention_UNet_text_extraction_finetuning"
page_extraction_model_name = "Residual_attention_UNet_page_extraction"

for dataset_name in datasets:
    folder_data_path = os.path.join("..",dataset_name)
    img_names = os.listdir(folder_data_path)
    for idx, img_name in enumerate(img_names):
        print(f"\nProcessing image {idx + 1}/{len(img_names)}: {img_name}")
        img_path = os.path.join(folder_data_path, img_name)
        if img_path in images_with_and_without_bleed_through["yes"] or img_path in images_with_and_without_bleed_through["no"]:
            print("Already considered: ", img_path,"\n")
            continue
        fig, axs = plt.subplots(1,1, figsize=(15,8))
        img = Image.open(img_path)
        axs.imshow(img)
        axs.set_title(img_path)
        axs.axis("off")
        plt.show()
        consider_or_not = input("Is there bleed-through effect in this image? (y/n/exit) ")
        while consider_or_not.lower() not in ["y","n","exit"]:
            consider_or_not = input("(WRONG INPUT. TYPE AGAIN). Is there bleed-through effect in this image? (y/n/exit) ")
        if consider_or_not == "y":
            images_with_and_without_bleed_through["yes"].append(img_path)
        elif consider_or_not == "n":
            images_with_and_without_bleed_through["no"].append(img_path)
        else:
            break

            
with open(json_path, "w") as f:
    json.dump(images_with_and_without_bleed_through, f, indent=4)


#%% CHECK IMAGES
json_path = "images_with_and_without_bleed_through.json"

with open(json_path, "r") as f:
    images_with_and_without_bleed_through = json.load(f)
    
for img_path in images_with_and_without_bleed_through["no"]:
    img = Image.open(img_path)
    fig, axs = plt.subplots(1,1, figsize=(15,8))
    axs.imshow(img)
    axs.set_title(img_path)
    axs.axis("off")
    plt.show()

