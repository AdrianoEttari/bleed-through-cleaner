#%%
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

# THIS VERSION TRIES TO USE 1024x1024 PATCHES FOR THE MODEL WITHOUT USING PADDING.
# def run_patchwise_inference(source, model, rgb_image, device, hole_size, patch_size=512, one_shot=False):
#     source = source.to(device)  # [1, 5, H, W]
#     B, C, H, W = source.shape
#     final_pred = torch.zeros_like(rgb_image)
#     final_overlapping_mask = torch.zeros_like(rgb_image)
     
#     holes_mask = source[:, 4:5, :, :].detach().cpu().numpy()
#     holes_unique_masks = split_holes(holes_mask)
    
#     for hole_mask in holes_unique_masks: 
#         if one_shot:
#             margin = 64  # context around the hole
            
#             text_ornament_mask = source[:, 3:4, :, :]
            
#             # --------------------------------------------------
#             # 1. Build mask tensor
#             # --------------------------------------------------

#             mask_np = hole_mask==1
            
#             coords = np.argwhere(mask_np[0][0])

#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0) + 1

#             # --------------------------------------------------
#             # 2. Add context margin
#             # --------------------------------------------------
#             y0 = max(0, y_min - margin)
#             x0 = max(0, x_min - margin)

#             y1 = min(H, y_max + margin)
#             x1 = min(W, x_max + margin)

#             # --------------------------------------------------
#             # 3. Crop only local region
#             # --------------------------------------------------
#             rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()

#             mask_crop = mask_np[0][0][y0:y1, x0:x1]

#             mask_crop = torch.tensor(
#                 mask_crop,
#                 device=device,
#                 dtype=torch.bool
#             )

#             # --------------------------------------------------
#             # 4. Corrupt ONLY masked region
#             # --------------------------------------------------
#             corrupted_crop = rgb_crop.clone()

#             expanded_mask = mask_crop.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

#             corrupted_crop[expanded_mask] = torch.rand(
#                 corrupted_crop[expanded_mask].shape,
#                 device=device
#             )

#             # --------------------------------------------------
#             # 5. Pad locally
#             # --------------------------------------------------
#             crop_H = corrupted_crop.shape[2]
#             crop_W = corrupted_crop.shape[3]

#             pad_h = (patch_size - crop_H % patch_size) % patch_size
#             pad_w = (patch_size - crop_W % patch_size) % patch_size

#             corrupted_crop = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))
#             mask_crop = F.pad(mask_crop.float(), (0, pad_w, 0, pad_h))

#             # --------------------------------------------------
#             # 6. Patchify LOCAL crop only
#             # --------------------------------------------------
#             rgb_patches = corrupted_crop.unfold(2, patch_size, patch_size) \
#                                         .unfold(3, patch_size, patch_size)

#             B, C, nH, nW, _, _ = rgb_patches.shape

#             rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5) \
#                                     .reshape(-1, 3, patch_size, patch_size)

#             # --------------------------------------------------
#             # 7. Inference
#             # --------------------------------------------------
#             with torch.no_grad():
#                 pred_patches = model(rgb_patches)

#             # --------------------------------------------------
#             # 8. Rebuild local prediction
#             # --------------------------------------------------
#             pred_crop = pred_patches.reshape(
#                 B,
#                 nH,
#                 nW,
#                 3,
#                 patch_size,
#                 patch_size
#             )

#             pred_crop = pred_crop.permute(0, 3, 1, 4, 2, 5)

#             pred_crop = pred_crop.reshape(
#                 B,
#                 3,
#                 nH * patch_size,
#                 nW * patch_size
#             )

#             # Remove padding
#             pred_crop = pred_crop[:, :, :crop_H, :crop_W]

#             # --------------------------------------------------
#             # 9. Paste ONLY masked area into final_pred
#             # --------------------------------------------------

#             expanded_mask = mask_crop.unsqueeze(0).unsqueeze(0)[:, :, :crop_H, :crop_W].bool() \
#                 .expand(-1, 3, -1, -1)
                
#             final_pred[:, :, y0:y1, x0:x1][expanded_mask] = pred_crop[expanded_mask]
                            
#         else:  
#             small_holes_masks = BigHole_2_SmallHoles(hole_mask, small_hole_size=hole_size, overlap=0)
#             # plt.imshow(np.sum(small_holes_masks, axis=0));plt.show() # check the overlapping
            
#             text_ornament_mask = source[:, 3:4, :, :]
            
#             for i, small_holes_mask in enumerate(small_holes_masks):

#                 # --------------------------------------------------
#                 # 1. Build mask tensor (holes)
#                 # --------------------------------------------------
#                 mask_np = (small_holes_mask == 1)

#                 coords = np.argwhere(mask_np)

#                 if len(coords) == 0:
#                     continue

#                 y_min, x_min = coords.min(axis=0)
#                 y_max, x_max = coords.max(axis=0) + 1

#                 y_distance = y_max - y_min
#                 x_distance = x_max - x_min

#                 assert y_distance <= 1024
#                 assert x_distance <= 1024
#                 y_margin = (1024 - y_distance) // 2
#                 x_margin = (1024 - x_distance) // 2
                
#                 # --------------------------------------------------
#                 # 2. Add context margin
#                 # --------------------------------------------------
#                 y0 = max(0, y_min - y_margin)
#                 x0 = max(0, x_min - x_margin)

#                 y1 = min(H, y_max + y_margin)
#                 x1 = min(W, x_max + x_margin)

#                 # --------------------------------------------------
#                 # 3. Crop only local region
#                 # --------------------------------------------------
#                 rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()

#                 mask_crop = mask_np[y0:y1, x0:x1]

#                 mask_crop = torch.tensor(
#                     mask_crop,
#                     device=device,
#                     dtype=torch.bool
#                 ).unsqueeze(0).unsqueeze(0)

#                 # --------------------------------------------------
#                 # 4. Corrupt ONLY masked region
#                 # --------------------------------------------------
#                 corrupted_crop = rgb_crop.clone()

#                 expanded_mask = mask_crop.expand(-1, 3, -1, -1)

#                 corrupted_crop[expanded_mask] = torch.rand(
#                     corrupted_crop[expanded_mask].shape,
#                     device=device
#                 )

#                 # --------------------------------------------------
#                 # 5. Pad locally
#                 # --------------------------------------------------
#                 crop_H = corrupted_crop.shape[2]
#                 crop_W = corrupted_crop.shape[3]

#                 pad_h = (patch_size - crop_H % patch_size) % patch_size
#                 pad_w = (patch_size - crop_W % patch_size) % patch_size

#                 corrupted_crop = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))
#                 mask_crop = F.pad(mask_crop.float(), (0, pad_w, 0, pad_h))

#                 # --------------------------------------------------
#                 # 6. Patchify LOCAL crop only
#                 # --------------------------------------------------
#                 rgb_patches = corrupted_crop.unfold(2, patch_size, patch_size) \
#                                             .unfold(3, patch_size, patch_size)

#                 B, C, nH, nW, _, _ = rgb_patches.shape

#                 rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5) \
#                                         .reshape(-1, 3, patch_size, patch_size)

#                 # --------------------------------------------------
#                 # 7. Inference
#                 # --------------------------------------------------
#                 with torch.no_grad():
#                     pred_patches = model(rgb_patches)

#                 if (x1-x0) != patch_size:
#                     x_distance = patch_size - (x1-x0)
#                     x0_adj = x0 - x_distance//2
#                     x1_adj = x1 + x_distance//2
#                     if x0_adj < 0:
#                         x1_adj = x1 + x_distance
#                         x0_adj = 0
#                     if x1_adj > W:
#                         x0_adj = x0 - x_distance
#                         x1_adj = W
#                 else:
#                     x0_adj = x0
#                     x1_adj = x1
                    
#                 if (y1-y0) != patch_size:
#                     y_distance = patch_size - (y1-y0)
#                     y0_adj = y0 - y_distance//2
#                     y1_adj = y1 + y_distance//2
#                     if y0_adj < 0:
#                         y1_adj = y1 + y_distance
#                         y0_adj = 0
#                     if y1_adj > H:
#                         y0_adj = y0 - y_distance
#                         y1_adj = H
#                 else:
#                     y0_adj = y0
#                     y1_adj = y1
                    
#                 if y1_adj - y0_adj != patch_size:
#                     y1_adj += patch_size - (y1_adj - y0_adj)
#                 if x1_adj - x0_adj != patch_size:
#                     x1_adj += patch_size - (x1_adj - x0_adj)
#                 mask = mask_crop.repeat(1,3,1,1)==1
#                 final_pred[:, :, y0_adj:y1_adj, x0_adj:x1_adj][mask] += pred_patches[mask]
#                 final_overlapping_mask[:, :, y0_adj:y1_adj, x0_adj:x1_adj][mask] += 1
                
#                 # plt.imshow(local_final[0].permute(1,2,0).detach().cpu().numpy())
#                 # plt.savefig(str(i)+".png")
            
#     coords = np.argwhere(hole_mask[0][0]==1)

#     final_pred /= final_overlapping_mask
#     holes_mask = np.repeat(holes_mask, 3, axis=1)  # [1,3,H,W]
#     final_pred[holes_mask == 0] = rgb_image[holes_mask == 0] # overwrite non-hole pixels with original, to be sure not to alter them
    
#     # final_pred_crop = final_pred[:, :, y0:y1, x0:x1].clone()
#     # final_pred_crop_smooth = TF.gaussian_blur(
#     #     final_pred_crop,
#     #     kernel_size=11,   # keep small to avoid over-blurring
#     #     sigma=2
#     # )
#     # final_pred[:, :, y0:y1, x0:x1] = final_pred_crop_smooth
    
#     final_pred[(text_ornament_mask==0).expand(-1, 3, -1, -1)] = rgb_image[(text_ornament_mask==0).expand(-1, 3, -1, -1)] # overwrite text and ornaments with original (non-bleed-through) pixels, to be sure not to alter them

#     return final_pred


def run_patchwise_inference(source, model, rgb_image, device, hole_size, patch_size=512, one_shot=False):
    source = source.to(device)  # [1, 5, H, W]
    B, C, H, W = source.shape
    final_pred = torch.zeros_like(rgb_image)
    final_overlapping_mask = torch.zeros_like(rgb_image)
     
    holes_mask = source[:, 4:5, :, :].detach().cpu().numpy()
    holes_unique_masks = split_holes(holes_mask)
    
    for hole_mask in holes_unique_masks: 
        if one_shot:
            margin = 64  # context around the hole
            
            text_ornament_mask = source[:, 3:4, :, :]
            
            # --------------------------------------------------
            # 1. Build mask tensor
            # --------------------------------------------------

            mask_np = hole_mask==1
            
            coords = np.argwhere(mask_np[0][0])

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1

            # --------------------------------------------------
            # 2. Add context margin
            # --------------------------------------------------
            y0 = max(0, y_min - margin)
            x0 = max(0, x_min - margin)

            y1 = min(H, y_max + margin)
            x1 = min(W, x_max + margin)

            # --------------------------------------------------
            # 3. Crop only local region
            # --------------------------------------------------
            rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()

            mask_crop = mask_np[0][0][y0:y1, x0:x1]

            mask_crop = torch.tensor(
                mask_crop,
                device=device,
                dtype=torch.bool
            )

            # --------------------------------------------------
            # 4. Corrupt ONLY masked region
            # --------------------------------------------------
            corrupted_crop = rgb_crop.clone()

            expanded_mask = mask_crop.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

            corrupted_crop[expanded_mask] = torch.rand(
                corrupted_crop[expanded_mask].shape,
                device=device
            )

            # --------------------------------------------------
            # 5. Pad locally
            # --------------------------------------------------
            crop_H = corrupted_crop.shape[2]
            crop_W = corrupted_crop.shape[3]

            pad_h = (patch_size - crop_H % patch_size) % patch_size
            pad_w = (patch_size - crop_W % patch_size) % patch_size

            corrupted_crop = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))
            mask_crop = F.pad(mask_crop.float(), (0, pad_w, 0, pad_h))

            # --------------------------------------------------
            # 6. Patchify LOCAL crop only
            # --------------------------------------------------
            rgb_patches = corrupted_crop.unfold(2, patch_size, patch_size) \
                                        .unfold(3, patch_size, patch_size)

            B, C, nH, nW, _, _ = rgb_patches.shape

            rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5) \
                                    .reshape(-1, 3, patch_size, patch_size)

            # --------------------------------------------------
            # 7. Inference
            # --------------------------------------------------
            with torch.no_grad():
                pred_patches = model(rgb_patches)

            # --------------------------------------------------
            # 8. Rebuild local prediction
            # --------------------------------------------------
            pred_crop = pred_patches.reshape(
                B,
                nH,
                nW,
                3,
                patch_size,
                patch_size
            )

            pred_crop = pred_crop.permute(0, 3, 1, 4, 2, 5)

            pred_crop = pred_crop.reshape(
                B,
                3,
                nH * patch_size,
                nW * patch_size
            )

            # Remove padding
            pred_crop = pred_crop[:, :, :crop_H, :crop_W]

            # --------------------------------------------------
            # 9. Paste ONLY masked area into final_pred
            # --------------------------------------------------

            expanded_mask = mask_crop.unsqueeze(0).unsqueeze(0)[:, :, :crop_H, :crop_W].bool() \
                .expand(-1, 3, -1, -1)
                
            final_pred[:, :, y0:y1, x0:x1][expanded_mask] = pred_crop[expanded_mask]
                            
        else:  
            small_holes_masks = BigHole_2_SmallHoles(hole_mask, small_hole_size=hole_size, overlap=10)
            # plt.imshow(np.sum(small_holes_masks, axis=0));plt.show() # check the overlapping

            margin = 64  # context around the hole
            
            text_ornament_mask = source[:, 3:4, :, :]
            
            for i, small_holes_mask in enumerate(small_holes_masks):

                # --------------------------------------------------
                # 1. Build mask tensor (holes)
                # --------------------------------------------------
                mask_np = (small_holes_mask == 1)

                coords = np.argwhere(mask_np)

                if len(coords) == 0:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1

                # --------------------------------------------------
                # 2. Add context margin
                # --------------------------------------------------
                y0 = max(0, y_min - margin)
                x0 = max(0, x_min - margin)

                y1 = min(H, y_max + margin)
                x1 = min(W, x_max + margin)

                # --------------------------------------------------
                # 3. Crop only local region
                # --------------------------------------------------
                rgb_crop = rgb_image[:, :, y0:y1, x0:x1].clone()

                mask_crop = mask_np[y0:y1, x0:x1]

                mask_crop = torch.tensor(
                    mask_crop,
                    device=device,
                    dtype=torch.bool
                ).unsqueeze(0).unsqueeze(0)

                # --------------------------------------------------
                # 4. Corrupt ONLY masked region
                # --------------------------------------------------
                corrupted_crop = rgb_crop.clone()

                expanded_mask = mask_crop.expand(-1, 3, -1, -1)

                corrupted_crop[expanded_mask] = torch.rand(
                    corrupted_crop[expanded_mask].shape,
                    device=device
                )

                # --------------------------------------------------
                # 5. Pad locally
                # --------------------------------------------------
                crop_H = corrupted_crop.shape[2]
                crop_W = corrupted_crop.shape[3]

                pad_h = (patch_size - crop_H % patch_size) % patch_size
                pad_w = (patch_size - crop_W % patch_size) % patch_size

                corrupted_crop = F.pad(corrupted_crop, (0, pad_w, 0, pad_h))
                mask_crop = F.pad(mask_crop.float(), (0, pad_w, 0, pad_h))

                # --------------------------------------------------
                # 6. Patchify LOCAL crop only
                # --------------------------------------------------
                rgb_patches = corrupted_crop.unfold(2, patch_size, patch_size) \
                                            .unfold(3, patch_size, patch_size)

                B, C, nH, nW, _, _ = rgb_patches.shape

                rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5) \
                                        .reshape(-1, 3, patch_size, patch_size)

                # --------------------------------------------------
                # 7. Inference
                # --------------------------------------------------
                with torch.no_grad():
                    pred_patches = model(rgb_patches)

                # --------------------------------------------------
                # 8. Rebuild local prediction
                # --------------------------------------------------
                pred_crop = pred_patches.reshape(
                    B,
                    nH,
                    nW,
                    3,
                    patch_size,
                    patch_size
                )

                pred_crop = pred_crop.permute(0, 3, 1, 4, 2, 5)

                pred_crop = pred_crop.reshape(
                    B,
                    3,
                    nH * patch_size,
                    nW * patch_size
                )

                # Remove padding
                pred_crop = pred_crop[:, :, :crop_H, :crop_W]

                # --------------------------------------------------
                # 9. Paste ONLY masked area into final_pred
                # --------------------------------------------------

                expanded_mask = mask_crop[:, :, :crop_H, :crop_W].bool() \
                    .expand(-1, 3, -1, -1)

                final_pred[:, :, y0:y1, x0:x1][expanded_mask] += pred_crop[expanded_mask]
                final_overlapping_mask[:, :, y0:y1, x0:x1][expanded_mask] += 1
                
                # plt.imshow(local_final[0].permute(1,2,0).detach().cpu().numpy())
                # plt.savefig(str(i)+".png")
            
    coords = np.argwhere(hole_mask[0][0]==1)

    final_pred /= final_overlapping_mask
    holes_mask = np.repeat(holes_mask, 3, axis=1)  # [1,3,H,W]
    final_pred[holes_mask == 0] = rgb_image[holes_mask == 0] # overwrite non-hole pixels with original, to be sure not to alter them
    
    # y_min, x_min = coords.min(axis=0)
    # y_max, x_max = coords.max(axis=0) + 1
    # y0 = max(0, y_min - margin)-margin
    # x0 = max(0, x_min - margin)-margin
    # y1 = min(H, y_max + margin)+margin
    # x1 = min(W, x_max + margin)+margin
    # final_pred_crop = final_pred[:, :, y0:y1, x0:x1].clone()
    # final_pred_crop_smooth = TF.gaussian_blur(
    #     final_pred_crop,
    #     kernel_size=11,   # keep small to avoid over-blurring
    #     sigma=2
    # )
    # final_pred[:, :, y0:y1, x0:x1] = final_pred_crop_smooth
    
    final_pred[(text_ornament_mask==0).expand(-1, 3, -1, -1)] = rgb_image[(text_ornament_mask==0).expand(-1, 3, -1, -1)] # overwrite text and ornaments with original (non-bleed-through) pixels, to be sure not to alter them

    return final_pred

pred_source = run_patchwise_inference(source, model, target, device, small_hole_size, patch_size=1024, one_shot=False)

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