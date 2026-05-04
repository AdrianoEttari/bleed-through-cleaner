#%%
import torch
import os
from UNet_model_inpainting import ResidualUNet
from utils_inpainting import get_data
multiple_gpus=False
save_every=1
batch_size=16
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("USING", device, " device")
snapshot_path="./snapshots/snapshot.pt"
dataset_path = os.path.join("dataset")
train_dataset = get_data(".", dataset_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model=ResidualUNet(in_channels=3, out_channels=3, channels=(32, 64, 128, 256), device=device).to(device)

snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(snapshot["MODEL_STATE"])
model.to(device)
print(sum(p.numel() for p in model.parameters()))
epochs_run = snapshot["EPOCHS_RUN"]
print(f"Snapshot loaded from {snapshot_path}")

model.eval()

#%% PROVA 1
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
# from torchvision import transforms
import numpy as np
from utils_inpainting import make_holes, normalize, make_holes_with_mouse
# from patchify import patchify
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bleed_through_cleaner import bleed_through_cleaner
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

json_path = "images_with_and_without_bleed_through.json"

with open(json_path, "r") as f:
    images_with_and_without_bleed_through = json.load(f)
    
img_path = images_with_and_without_bleed_through["yes"][10]

models_folder = os.path.join("..","models")

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
    # pad
    pad_h = (patch_size - source.shape[2] % patch_size) % patch_size
    pad_w = (patch_size - source.shape[3] % patch_size) % patch_size
    source = F.pad(source, (0, pad_w, 0, pad_h))
    target = F.pad(target, (0, pad_w, 0, pad_h))
    

#%%
def pred_inpainted_page(pred, holes_mask_n, rgb_clean, text_mask_n):
    holes_mask_n = holes_mask_n.repeat(1,3,1,1)
    text_mask_n = text_mask_n.repeat(1,3,1,1)
    pred[holes_mask_n == 0] = rgb_clean[holes_mask_n == 0]
    pred[text_mask_n == 0] = rgb_clean[text_mask_n == 0]
    return pred

def run_patchwise_inference(source, model, rgb_image, device, patch_size=512):

    source = source.to(device)  # [1, 5, H, W]
    B, C, H, W = source.shape
    if H == patch_size and W == patch_size:
        with torch.no_grad():
            pred = model(source[:, :3, :, :])
        pred = pred_inpainted_page(pred, source[:, 4:5, :, :], rgb_image, source[:, 3:4, :, :])
        return pred
    # --------------------------------------------------
    # 1. Pad so H, W are divisible by patch_size
    # --------------------------------------------------
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    source = F.pad(source, (0, pad_w, 0, pad_h))  # pad W then H
    _, _, H_pad, W_pad = source.shape

    # --------------------------------------------------
    # 2. Split channels
    # --------------------------------------------------
    rgb_input  = source[:, :3]   # [1, 3, H, W]
    text_mask  = source[:, 3:4]  # [1, 1, H, W]
    holes_mask = source[:, 4:5]  # [1, 1, H, W]

    # --------------------------------------------------
    # 3. Patchify using unfold
    # --------------------------------------------------
    rgb_patches = rgb_input.unfold(2, patch_size, patch_size)\
                           .unfold(3, patch_size, patch_size)
    text_patches = text_mask.unfold(2, patch_size, patch_size)\
                             .unfold(3, patch_size, patch_size)
    holes_patches = holes_mask.unfold(2, patch_size, patch_size)\
                               .unfold(3, patch_size, patch_size)
    rgb_image_patches = rgb_image.unfold(2, patch_size, patch_size)\
                                 .unfold(3, patch_size, patch_size)

    # Shape: [B, C, nH, nW, 512, 512]
    B, C, nH, nW, _, _ = rgb_patches.shape

    # Flatten patches into batch dimension
    rgb_patches = rgb_patches.permute(0, 2, 3, 1, 4, 5)\
                             .reshape(-1, 3, patch_size, patch_size)

    text_patches = text_patches.permute(0, 2, 3, 1, 4, 5)\
                               .reshape(-1, 1, patch_size, patch_size)

    holes_patches = holes_patches.permute(0, 2, 3, 1, 4, 5)\
                                 .reshape(-1, 1, patch_size, patch_size)
                                 
    rgb_image_patches = rgb_image_patches.permute(0, 2, 3, 1, 4, 5)\
                                 .reshape(-1, 3, patch_size, patch_size)

    # --------------------------------------------------
    # 4. Model inference on RGB only
    # --------------------------------------------------
    with torch.no_grad():
        pred_patches = model(rgb_patches)  # [N, 3, 512, 512]

    # --------------------------------------------------
    # 5. Apply inpainting logic per patch
    # --------------------------------------------------
    pred_patches = pred_inpainted_page(
        pred_patches,
        holes_patches,
        rgb_image_patches,
        text_patches
    )

    # --------------------------------------------------
    # 6. Unpatchify (reconstruct)
    # --------------------------------------------------
    pred_patches = pred_patches.view(B, nH, nW, 3, patch_size, patch_size)
    pred_patches = pred_patches.permute(0, 3, 1, 4, 2, 5)

    # Merge spatial blocks
    pred_full = pred_patches.reshape(
        B, 3,
        nH * patch_size,
        nW * patch_size
    )

    # --------------------------------------------------
    # 7. Remove padding
    # --------------------------------------------------
    pred_full = pred_full[:, :, :H, :W]

    return pred_full

pred_source = run_patchwise_inference(source, model, target, device, patch_size=512)

rgb_corrupt = source[:, :3, :, :]
text_mask = source[:, 3, :, :]
holes_mask = source[:, 4, :, :]

full_mask = (holes_mask == 1) | (text_mask == 0)
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
