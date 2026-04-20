#%%
import torch
import os
from UNet_model import ResidualUNet
from utils import get_data

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

#%%
for source, targets in train_loader:
    source = source.to(device)
    rgb_corrupt = source[:, :3, :, :]
    text_mask = source[:, 3, :, :]
    holes_mask = source[:, 4, :, :]
    outputs = model(rgb_corrupt)
    break

# %%
import matplotlib.pyplot as plt
import numpy as np

def pred_inpainted_page(pred, holes_mask_n, rgb_clean, text_mask_n):
    pred[holes_mask_n==0] = rgb_clean[holes_mask_n==0]
    pred[text_mask_n==0] = rgb_clean[text_mask_n==0]
    return pred 


for i in range(source.shape[0]):
    img_n=i
    pred = outputs[img_n].permute(1,2,0).detach().cpu()
    orig = rgb_corrupt[img_n].permute(1,2,0).detach().cpu()
    holes_mask_n = holes_mask[img_n].detach().cpu()
    rgb_n = targets[img_n].permute(1,2,0).detach().cpu()
    text_mask_n = text_mask[img_n].detach().cpu()
    
    full_pred = pred_inpainted_page(pred, holes_mask_n, rgb_n, text_mask_n)


    fig, axs = plt.subplots(1,3,figsize=(15,10)) 
    axs = axs.ravel()
    axs[0].imshow(orig)
    axs[0].set_title("Input")
    axs[1].imshow(rgb_n)
    axs[1].set_title("GT")
    axs[2].imshow(full_pred)
    axs[2].set_title("Pred")
    plt.show()
# %%
