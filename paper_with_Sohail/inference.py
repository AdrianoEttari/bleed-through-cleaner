#%% SETTINGS
#I'm working on inpainting of bleed-through areas in ancient document parchemnts. Now I'm focusing on the re-creation of the parchment style background. To do so,  
import torch
import os
from UNet_model_inpainting import ResidualUNet

multiple_gpus=False
save_every=1
batch_size=16
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("USING", device, " device")

base_dir = os.path.dirname(os.path.abspath(__file__))


# snapshot_path=os.path.join(base_dir, "./snapshots/snapshot.pt")

normalization = "group"
loss_func_type = "vgg"

if normalization == "batch" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024.pt") # ConvTranspose
elif normalization == "inst" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_InstNorm.pt") # ConvTranspose
elif normalization == "group" and loss_func_type == "vgg":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_vgg.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_l1.pt") # Upsample, Conv

model=ResidualUNet(in_channels=3, out_channels=3, channels=(32, 64, 128, 256), device=device, normalization=normalization).to(device)

snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(snapshot["MODEL_STATE"])
model.to(device)
print(sum(p.numel() for p in model.parameters()))
epochs_run = snapshot["EPOCHS_RUN"]
print(f"Snapshot loaded from {snapshot_path}")
print(f"Number of trained epochs: {epochs_run}")

model.eval()

#%% PROVA 1
# from utils_inpainting import get_data

# dataset_path = os.path.join("dataset_MAGIC_PatchSize1024_partial")
# # dataset_path = r"D:/MAGIC/dataset_MAGIC/dataset_MAGIC"
# train_dataset = get_data(".", dataset_path, max_hole_size=300)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# for source, targets in train_loader:
#     source = source.to(device)
#     rgb_corrupt = source[:, :3, :, :]
#     text_mask = source[:, 3, :, :]
#     holes_mask = source[:, 4, :, :]
#     with torch.no_grad():
#         outputs = model(rgb_corrupt)
#     break

# import matplotlib.pyplot as plt
# import numpy as np

# def pred_inpainted_page(pred, holes_mask_n, rgb_clean, text_mask_n):
#     pred[holes_mask_n==0] = rgb_clean[holes_mask_n==0]
#     pred[text_mask_n==0] = rgb_clean[text_mask_n==0]
#     return pred 

# for i in range(source.shape[0]):
#     img_n=i
#     pred = outputs[img_n].permute(1,2,0).detach().cpu()
#     orig = rgb_corrupt[img_n].permute(1,2,0).detach().cpu()
#     holes_mask_n = holes_mask[img_n].detach().cpu()
#     rgb_n = targets[img_n].permute(1,2,0).detach().cpu()
#     text_mask_n = text_mask[img_n].detach().cpu()
    
#     full_pred = pred_inpainted_page(pred, holes_mask_n, rgb_n, text_mask_n)

#     fig, axs = plt.subplots(1,3,figsize=(15,10)) 
#     axs = axs.ravel()
#     axs[0].imshow(orig)
#     axs[0].set_title("Input")
#     axs[1].imshow(rgb_n)
#     axs[1].set_title("GT")
#     axs[2].imshow(full_pred)
#     axs[2].set_title("Pred")
#     plt.show()
# %% PROVA 2
import json
import numpy as np
from utils_inpainting import make_holes, normalize, make_holes_with_mouse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bleed_through_cleaner import bleed_through_cleaner
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

json_path = os.path.join(base_dir, "images_with_and_without_bleed_through.json")

with open(json_path, "r") as f:
    images_with_and_without_bleed_through = json.load(f)
    
img_path = os.path.join(base_dir, images_with_and_without_bleed_through["yes"][15])


models_folder = os.path.join(base_dir, "..", "models")

def process_img(img_path, models_folder, device):
    cleaner = bleed_through_cleaner(img_path, models_folder, False, device)
    page_filtered_image, mask, _ = cleaner.bleed_through_finder(page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction')
    img_mask_concat = np.concatenate([page_filtered_image, mask[:,:,None]], axis=2) 
    # rgb_image, mask, holes_mask = make_holes(img_mask_concat)
    rgb_image, mask, holes_mask = make_holes_with_mouse(img_mask_concat)
    return rgb_image, mask, holes_mask

patch_size = 512


######## LITTLE TEST
# from PIL import Image
# img = np.array(Image.open(img_path))
# img = img[:patch_size,250:250+patch_size, :]
# Image.fromarray(img).save("test_patch.png")
# rgb_image, text_mask, holes_mask = process_img("test_patch.png", models_folder, device)
##########

rgb_image, text_mask, holes_mask = process_img(img_path, models_folder, device)

n = np.sum(holes_mask)
rgb_image = normalize(rgb_image)
rgb_corrupt = rgb_image.copy()

rgb_corrupt[holes_mask == 1] = np.random.uniform(low=0.0, high=1.0, size=(n, 3))
input_tensor = np.concatenate(
            [
                rgb_corrupt,
                text_mask[..., None].astype(np.float32),
                holes_mask[..., None].astype(np.float32)
            ],
            axis=2
        )
source = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
target = torch.tensor(rgb_image).permute(2, 0, 1).unsqueeze(0).to(device)

if source.shape[2] != patch_size or source.shape[3] != patch_size:
    pad_h = (patch_size - source.shape[2] % patch_size) % patch_size
    pad_w = (patch_size - source.shape[3] % patch_size) % patch_size
    source = F.pad(source, (0, pad_w, 0, pad_h))
    target = F.pad(target, (0, pad_w, 0, pad_h))

#%%
import torchvision.transforms.functional as TF
import cv2

small_hole_size = 80

def BigHole_2_SmallHoles(hole_mask,
                         small_hole_size=50,
                         overlap=10):

    """
    Take as input a big binary image with 0 values for the non masked areas and 1 values for the masked areas.
    From the starting binary image are generated a series of binary images of the same size of the starting image, but with a smaller 
    masked area: the masked area is splitted in smaller masked areas of size small_hole_size*small_hole_size
    """
    hole_mask = hole_mask[0, 0]  # [H,W]

    coords = np.argwhere(hole_mask == 1)

    if len(coords) == 0:
        return []

    H, W = hole_mask.shape

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    stride = small_hole_size - overlap

    if stride <= 0:
        raise ValueError("overlap must be smaller than patch size")

    small_masks = []

    for y in range(y_min, y_max, stride):
        for x in range(x_min, x_max, stride):

            end_y = min(y + small_hole_size, y_max)
            end_x = min(x + small_hole_size, x_max)

            patch = hole_mask[y:end_y, x:end_x]

            # keep only windows that intersect the hole
            if np.any(patch):

                small_mask = np.zeros_like(hole_mask)

                # preserve only the hole pixels in this window
                small_mask[y:end_y, x:end_x] = patch

                small_masks.append(small_mask)

    return small_masks

def split_holes(holes_mask, connectivity=8):
    """
    holes_mask: [1,1,H,W] or [H,W], binary {0,1}
    returns: list of [1,1,H,W] or [H,W] masks, one per connected hole
    """

    if holes_mask.ndim == 4:
        mask = holes_mask[0, 0]
    else:
        mask = holes_mask

    mask = mask.astype(np.uint8)

    # connected components
    num_labels, labels = cv2.connectedComponents(mask, connectivity)

    masks = []

    for label in range(1, num_labels):  # 0 = background
        component = (labels == label).astype(np.uint8)

        out = np.zeros_like(mask)
        out[component == 1] = 1

        # restore original shape
        if holes_mask.ndim == 4:
            out = out[None, None, :, :]
        else:
            out = out

        masks.append(out)

    return masks

def run_patchwise_inference(source, model, rgb_image, device, hole_size, patch_size=512, one_shot=False):

    source = source.to(device)  # [1, 5, H, W]
    B, C, H, W = source.shape

    final_pred = torch.zeros_like(rgb_image)
    final_overlapping_mask = torch.zeros_like(rgb_image)

    holes_mask = source[:, 4:5, :, :].detach().cpu().numpy()
    holes_unique_masks = split_holes(holes_mask)

    text_ornament_mask = source[:, 3:4, :, :]  # [1,1,H,W]

    for hole_mask in holes_unique_masks:
        # =========================================================
        # ONE-SHOT MODE
        # =========================================================
        if one_shot:

            margin = 64
            mask_np = hole_mask[0, 0].astype(bool)

            coords = np.argwhere(mask_np)
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1

            y0 = max(0, y_min - margin)
            x0 = max(0, x_min - margin)
            y1 = min(H, y_max + margin)
            x1 = min(W, x_max + margin)

            rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()

            mask_crop_np = mask_np[y0:y1, x0:x1]
            mask_crop = torch.from_numpy(mask_crop_np).to(device).bool()

            corrupted_crop = rgb_crop.clone()

            expanded_mask = mask_crop.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)

            corrupted_crop[expanded_mask] = torch.rand(
                corrupted_crop[expanded_mask].shape,
                device=device
            )

            crop_H, crop_W = corrupted_crop.shape[2], corrupted_crop.shape[3]

            pad_h = (patch_size - crop_H % patch_size) % patch_size
            pad_w = (patch_size - crop_W % patch_size) % patch_size

            corrupted_crop = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))
            mask_crop = F.pad(mask_crop.float(), (0, pad_w, 0, pad_h))

            rgb_patches = corrupted_crop.unfold(2, patch_size, patch_size)\
                                        .unfold(3, patch_size, patch_size)

            _, _, nH, nW, _, _ = rgb_patches.shape

            rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5)\
                                     .reshape(-1, 3, patch_size, patch_size)

            with torch.no_grad():
                pred_patches = model(rgb_patches)

            pred_crop = pred_patches.reshape(
                1, nH, nW, 3, patch_size, patch_size
            ).permute(0, 3, 1, 4, 2, 5)\
             .reshape(1, 3, nH * patch_size, nW * patch_size)

            pred_crop = pred_crop[:, :, :crop_H, :crop_W]

            expanded_mask = mask_crop.bool().unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)

            region_pred = final_pred[:, :, y0:y1, x0:x1]
            region_mask = final_overlapping_mask[:, :, y0:y1, x0:x1]

            region_pred += pred_crop * expanded_mask
            region_mask += expanded_mask.float()

        # =========================================================
        # MULTI-HOLE MODE
        # =========================================================
        else:

            small_holes_masks = BigHole_2_SmallHoles(
                hole_mask, small_hole_size=hole_size, overlap=10
            )

            for small_holes_mask in small_holes_masks:

                mask_np = small_holes_mask.astype(bool)
                coords = np.argwhere(mask_np)

                if len(coords) == 0:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1

                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2

                y0 = max(0, center_y - patch_size // 2)
                x0 = max(0, center_x - patch_size // 2)
                y1 = min(H, center_y + patch_size // 2)
                x1 = min(W, center_x + patch_size // 2)

                rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()
                corrupted_crop = rgb_crop.clone()

                # ---- shift coords into crop space ----
                y_coords = coords[:, 0] - y0 # rgb_crop is the crop on rgb_image and coords works on rgb_image, so we must shift coords into the domain of rgb_crop
                x_coords = coords[:, 1] - x0 #

                valid = (
                    (y_coords >= 0) & (y_coords < (y1 - y0)) &
                    (x_coords >= 0) & (x_coords < (x1 - x0))
                ) # It's possible that some coord is smaller than y0 or x0 and also be sure that they are inside [y1,y0] and [x1,x0]

                y_coords = y_coords[valid]
                x_coords = x_coords[valid]

                if len(y_coords) == 0:
                    continue

                y_t = torch.from_numpy(y_coords).to(device)
                x_t = torch.from_numpy(x_coords).to(device)

                corrupted_crop[:, :, y_t, x_t] = torch.rand(
                    (3, len(y_t)),
                    device=device
                )

                crop_H, crop_W = corrupted_crop.shape[2], corrupted_crop.shape[3]
                # In case the crop is not (1024,1024) sized (that is possible if the (center_x, center_y) point was on the board of rgb_img), pad the image in order to be that size.
                pad_h = (patch_size - crop_H % patch_size) % patch_size
                pad_w = (patch_size - crop_W % patch_size) % patch_size

                if pad_h!=0:
                    # print("Padding H!!!")
                    y0_new = y0-pad_h
                    if y0_new < 0:
                        y1 = y1+pad_h
                    else:
                        y0 = y0_new
                if pad_w!=0:
                    # print("Padding W!!!")
                    x0_new = x0-pad_w
                    if x0_new < 0:
                        x1 = x1+pad_w
                    else:
                        x0 = x0_new
                
                corrupted_pad = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))


                with torch.no_grad():
                    pred = model(corrupted_pad)

                mask_crop_np = mask_np[y0:y1, x0:x1]
                # mask_crop_np = np.pad(mask_crop_np, [(0,pad_h), (0, pad_w)], mode="constant")
                
                expanded_mask = torch.from_numpy(mask_crop_np)\
                    .to(device)\
                    .bool()\
                    .unsqueeze(0)\
                    .unsqueeze(0)\
                    .expand(1, 3, -1, -1) 

                region_pred = final_pred[:, :, y0:y1, x0:x1] # It updates final_pred and final_overlapping_mask with aliasing 
                region_mask = final_overlapping_mask[:, :, y0:y1, x0:x1]

                region_pred += pred * expanded_mask
                region_mask += expanded_mask.float()
                
                # fig, axs = plt.subplots(1,2, figsize=(15,10))
                # img1 = final_pred.clone()
                # img2 = final_overlapping_mask.clone()
                # img1 = (img1-img1.min())/(img1.max()-img1.min())
                # img2 = (img2-img2.min())/(img2.max()-img2.min())
                # axs[0].imshow(img1[0].permute(1,2,0).detach().cpu())
                # axs[1].imshow(img2[0].permute(1,2,0).detach().cpu())
                # plt.show()
    # =========================================================
    # FINAL NORMALIZATION
    # =========================================================
    final_pred = final_pred / torch.clamp(final_overlapping_mask, min=1.0)

    holes_mask_3c = np.repeat(holes_mask, 3, axis=1)
    holes_mask_t = torch.from_numpy(holes_mask_3c).to(device).bool()

    final_pred[~holes_mask_t] = rgb_image[~holes_mask_t] # The areas of final_pred not masked in holes_mask_t are equal to the original image 

    final_pred[(text_ornament_mask == 0).expand(-1, 3, -1, -1)] = \
        rgb_image[(text_ornament_mask == 0).expand(-1, 3, -1, -1)] # The text and the ornaments are placed on the predicted area

    return final_pred

pred_source = run_patchwise_inference(source, model, target, device, small_hole_size, patch_size=1024, one_shot=False)

# PROBLEMS: 
# * too small contiguity
# * chessboard effect
# * patterns are not perfectly aligned with the parchment

#%% SHOW RESULTS
rgb_corrupt = source[:, :3, :, :]
text_ornament_mask = source[:, 3, :, :]
holes_mask = source[:, 4, :, :]

full_mask = (holes_mask == 1) | (text_ornament_mask == 0)
full_mask = ~full_mask

fig, axs = plt.subplots(1,4,figsize=(15,10))
axs = axs.ravel()

axs[0].imshow(rgb_corrupt[0].permute(1,2,0).detach().cpu())
axs[0].set_title("rgb corrupt")
axs[1].imshow(full_mask.permute(1,2,0).detach().cpu(), cmap="gray")
axs[1].set_title("full mask")
axs[2].imshow(target[0].permute(1,2,0).detach().cpu())
axs[2].set_title("rgb clean")
axs[3].imshow(pred_source[0].permute(1,2,0).detach().cpu())
axs[3].set_title("predicted inpainted")

plt.show()
# %%
# import streamlit as st
# from PIL import Image

# rgb_corrupt = source[:, :3, :, :]
# text_ornament_mask = source[:, 3, :, :]
# holes_mask = source[:, 4, :, :]

# full_mask = (holes_mask == 1) | (text_ornament_mask == 0)
# full_mask = ~full_mask

# predicted_img = Image.fromarray(pred_source[0].permute(1,2,0).detach().cpu().numpy())
# original_img = Image.fromarray(rgb_corrupt[0].permute(1,2,0).detach().cpu().numpy())


# tab1, tab2 = st.tabs(["Cleaned", "Original"])
# tab1.image(predicted_img, use_column_width=True)
# tab2.image(original_img, use_column_width=True)