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

normalization = "group"
loss_func_type = "vgg_dominant"

if normalization == "batch" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024.pt") # ConvTranspose
elif normalization == "inst" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_InstNorm.pt") # ConvTranspose
elif normalization == "group" and loss_func_type == "vgg":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_vgg.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "l1":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_l1.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "grad":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_grad.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "vgg_strong":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_vgg_strong.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "vgg_strong05":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_vgg_strong05.pt") # Upsample, Conv
elif normalization == "group" and loss_func_type == "vgg_dominant":
    snapshot_path = os.path.join(base_dir, "snapshots", "snapshot_PathSize_1024_Norm_group_Loss_vgg_dominant.pt") # Upsample, Conv


model=ResidualUNet(in_channels=3, out_channels=3, channels=(32, 64, 128, 256), device=device, normalization=normalization).to(device)

snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(snapshot["MODEL_STATE"])
model.to(device)
print(sum(p.numel() for p in model.parameters()))
epochs_run = snapshot["EPOCHS_RUN"]
print(f"Snapshot loaded from {snapshot_path}")
print(f"Number of trained epochs: {epochs_run}")

model.eval()

#%% INFERENCE ON THE TRANINING DATASET IMAGES
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
import cv2

json_path = os.path.join(base_dir, "images_with_and_without_bleed_through.json")

with open(json_path, "r") as f:
    images_with_and_without_bleed_through = json.load(f)
    
img_path = os.path.join(base_dir, images_with_and_without_bleed_through["yes"][15])

# img_path = os.path.join(base_dir, "imgs_to_clean", "IT-FR0084_ams_0271_0052_pa_0048.jpg")

models_folder = os.path.join(base_dir, "..", "models")

def process_img(img_path, models_folder, device):
    cleaner = bleed_through_cleaner(img_path, models_folder, False, device)
    page_filtered_image, mask, _ = cleaner.bleed_through_finder(page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                # ornament_model_name=None,
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

small_hole_size = 100

def plot_tensor_first_ch(tensor):
    plt.imshow(tensor[0].permute(1,2,0).detach().cpu())
    plt.show()

def BigHole_2_SmallHoles(hole_mask,
                         small_hole_size=50,
                         overlap_perc=25):

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

    # x_dist = x_max-x_min
    # y_dist = y_max-y_min
    
    # if x_dist < 100 or y_dist < 100:
    #     overlap = int(min(x_dist, y_dist) * overlap_perc/100)
    #     stride = min(x_dist, y_dist) - overlap
    # else:
    #     overlap = int(small_hole_size * overlap_perc/100)
    #     stride = small_hole_size - overlap
    
    overlap = int(small_hole_size * overlap_perc/100)
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
            if end_x == x_max and end_y == y_max:
                break
        if end_x == x_max and end_y == y_max:
            break 

    return small_masks

def blend_borders(pred_img, orig_img, mask_of_holes, kernel_size=21):
    """
    Smoothly blend the boundary between inpainted regions and original image
    using a soft transition band around the mask edges.

    This function reduces visible seams between reconstructed (predicted)
    regions and the original image by creating a narrow blending band along
    the border of the masked (inpainted) areas. Within this band, pixel values
    are averaged between the predicted image and the original image, producing
    a smoother visual transition.

    The blending band is obtained by eroding the binary mask and subtracting
    the eroded version from the original mask, effectively isolating a thin
    border region around each masked area.

    Parameters
    ----------
    pred_img : torch.Tensor
        Tensor of shape (1, C, H, W) representing the predicted/inpainted image.
        This tensor is modified in-place within the blending band.

    orig_img : torch.Tensor
        Tensor of shape (1, C, H, W) representing the original image
        (with holes or missing regions).

    mask_of_holes : numpy.ndarray
        Binary mask of shape (1, 1, H, W) or equivalent, where:
            - Values > 0 indicate inpainted (hole) regions
            - Values == 0 indicate original regions
        The mask is converted internally to uint8.

    kernel_size : int, optional (default=21)
        Size of the elliptical structuring element used for erosion.
        Controls the thickness of the blending band:
            - Larger values → wider, smoother transition
            - Smaller values → sharper boundary

    Returns
    -------
    pred_img : torch.Tensor
        Updated predicted image with border blending applied.
        Blending is applied only within the transition band.

    mask_of_holes_t : torch.Tensor
        Torch tensor version of the input mask, moved to the same device.

    Processing Steps
    ----------------
    1. Convert mask to a binary uint8 format.
    2. Apply morphological erosion to shrink the mask.
    3. Compute the blending band as:
           blend_band = original_mask - eroded_mask
       This isolates a thin border around the masked regions.
    4. Expand the blending band to match the number of image channels.
    5. In the blending band, replace pixel values with:
           0.5 * pred_img + 0.5 * orig_img
       creating a smooth transition between images.

    Notes
    -----
    - This operation is local and lightweight compared to gradient-domain
      blending (e.g., Poisson blending).
    - Useful as a post-processing step after inpainting.
    - The function modifies `pred_img` in-place.
    - Assumes `pred_img` and `orig_img` are on the same device.

    Limitations
    -----------
    - Performs simple linear blending; does not account for gradient consistency.
    - May not fully remove seams in cases of strong color or illumination mismatch.
    - Kernel size must be tuned depending on resolution and mask size.
    """
    mask_of_holes = (mask_of_holes > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask_of_holes[0][0], kernel, iterations=1)
        
    mask_of_holes_t = torch.from_numpy(mask_of_holes).to(device)
    eroded_t = torch.from_numpy(eroded).to(device).unsqueeze(0).unsqueeze(0)

    blend_band = mask_of_holes_t-eroded_t
    blend_band = blend_band.expand(-1,3,-1,-1)
    
    pred_img[blend_band==1] = pred_img[blend_band==1]*0.5 + orig_img[blend_band==1]*0.5
    return pred_img, mask_of_holes_t

def gradient_domain_blending(src, dst, mask):

    """
    Perform gradient-domain (Poisson) blending of inpainted regions into an image
    using OpenCV's seamlessClone, handling multiple disconnected areas independently.

    This function blends the content of a source image (`src`) into a destination
    image (`dst`) only within the regions specified by `mask`. It uses gradient-domain
    (Poisson) blending to ensure seamless transitions, eliminating visible seams,
    color discontinuities, and texture inconsistencies between inpainted and original areas.

    Unlike naive blending or direct pixel replacement, this method reconstructs pixel
    values inside the masked regions by matching image gradients (edges and local
    variations) from the source image while enforcing boundary continuity with the
    destination image.

    Parameters
    ----------
    src : torch.Tensor
        Tensor of shape (1, C, H, W) representing the source image.
        Typically contains the inpainted result (i.e., reconstructed areas).
        Expected value range: [0, 1], RGB format.

    dst : torch.Tensor
        Tensor of shape (1, C, H, W) representing the destination image.
        Typically the original image with missing/corrupted regions.
        Expected value range: [0, 1], RGB format.

    mask : torch.Tensor
        Tensor of shape (1, 1, H, W) or (1, C, H, W), where:
            - Value 1 indicates regions to blend (inpainted areas)
            - Value 0 indicates unchanged regions
        The mask is converted internally to a single-channel uint8 image
        with values {0, 255}.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (1, C, H, W), representing the blended image.
        Values are in [0, 1] range, RGB format.

    Processing Steps
    ----------------
    1. Convert PyTorch tensors to NumPy arrays and scale to [0, 255].
    2. Convert RGB → BGR format (required by OpenCV).
    3. Extract a binary mask from the first channel.
    4. Identify connected components in the mask to separate distinct regions.
    5. For each region:
        a. Compute its bounding box
        b. Apply padding to include surrounding context
        c. Crop corresponding regions from the source and mask
        d. Compute the region center in the destination image
        e. Perform seamless cloning using cv2.NORMAL_CLONE
    6. Convert result back to RGB and normalize to [0, 1].
    7. Convert NumPy array back to PyTorch tensor format.
    """

    src = src[0].permute(1,2,0).detach().cpu().numpy()
    dst = dst[0].permute(1,2,0).detach().cpu().numpy()
    mask = mask[0].permute(1,2,0).detach().cpu().numpy()
    src = (src * 255).astype(np.uint8)
    dst = (dst * 255).astype(np.uint8)
    mask = (mask[:,:,0] * 255).astype(np.uint8)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    
    num_labels, labels = cv2.connectedComponents(mask)
    output = dst.copy()
    
    for label in range(1, num_labels):
        region_mask = (labels == label).astype(np.uint8)*255
        ys, xs = np.where(region_mask > 0)
        
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        
        pad = 10
        y1 = max(0, y1 - pad)
        y2 = min(mask.shape[0], y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(mask.shape[1], x2 + pad)
        
        src_crop = src[y1:y2, x1:x2]
        mask_crop = region_mask[y1:y2, x1:x2]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        output = cv2.seamlessClone(
            src_crop, 
            output, 
            mask_crop, 
            center, 
            cv2.NORMAL_CLONE
            # cv2.MIXED_CLONE
        )
        

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = output.astype(np.float32) / 255.0

    output_torch = torch.from_numpy(output).permute(2,0,1).unsqueeze(0)
    return output_torch

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
                # hole_mask, small_hole_size=hole_size, overlap_perc=10
                hole_mask, small_hole_size=hole_size, overlap_perc=60     
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

                left,right,top,bottom=0,0,0,0
                if pad_h!=0:
                    # print("Padding H!!!")
                    y0_new = y0-pad_h
                    if y0_new < 0:
                        bottom=pad_h
                        y1 = y1+pad_h
                    else:
                        top=pad_h
                        y0 = y0_new
                if pad_w!=0:
                    # print("Padding W!!!")
                    x0_new = x0-pad_w
                    if x0_new < 0:
                        right=pad_w
                        x1 = x1+pad_w
                    else:
                        left=pad_w
                        x0 = x0_new
                
                corrupted_pad = F.pad(corrupted_crop, (left, right, top, bottom)) #(left, right, top, bottom)

                with torch.no_grad():
                    pred = model(corrupted_pad)

                mask_crop_np = mask_np[y0:y1, x0:x1]
                
                expanded_mask = torch.from_numpy(mask_crop_np)\
                    .to(device)\
                    .bool()\
                    .unsqueeze(0)\
                    .unsqueeze(0)\
                    .expand(1, 3, -1, -1) 

                region_pred = final_pred[:, :, y0:y1, x0:x1] # It updates final_pred and final_overlapping_mask with aliasing 
                region_mask = final_overlapping_mask[:, :, y0:y1, x0:x1]

                # fig, axs = plt.subplots(1,2, figsize=(15,10))
                # axs=axs.ravel()
                # axs[0].imshow(pred[0].permute(1,2,0).detach().cpu())
                # prova = pred*0.5+expanded_mask*0.5
                # axs[1].imshow(prova[0].permute(1,2,0).detach().cpu())
                # plt.show()
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

    final_pred = final_pred / torch.clamp(final_overlapping_mask, min=1.0)

    final_pred, holes_mask_t = blend_borders(final_pred, rgb_image, holes_mask)
        
    holes_mask_t = holes_mask_t.expand(-1,3,-1,-1)
    final_pred[holes_mask_t==0] = rgb_image[holes_mask_t==0] # The areas of final_pred not masked in holes_mask_t are equal to the original image 
    
    final_pred = gradient_domain_blending(final_pred, rgb_image, holes_mask_t)
    final_pred = final_pred.to(device)
    final_pred[(text_ornament_mask == 0).expand(-1, 3, -1, -1)] = \
        rgb_image[(text_ornament_mask == 0).expand(-1, 3, -1, -1)] # The text and the ornaments are placed on the predicted area

    return final_pred

pred_source = run_patchwise_inference(source, model, target, device, small_hole_size, patch_size=1024, one_shot=False)

# PROBLEMS: 
# * too small contiguity # SOLVED with blend_borders() and gradient_domain_blending() functions
# * chessboard effect # SOLVED with overlapping patches and averaging the overlapping parts
# * patterns are not perfectly aligned with the parchment # SOLVED with strong vgg loss and gradient_domain_blending() function

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