#%%
from json.tool import main

import torch
import numpy as np
import os
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from utils_inpainting import get_data
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

def prepare_data_loader(base_dir, dataset_path, multiple_gpus, batch_size):
    dataset = get_data(base_dir, dataset_path, max_hole_size=300)

    if multiple_gpus:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                sampler=DistributedSampler(dataset, shuffle=True),
                                                num_workers=min(4, os.cpu_count()),
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=4,
                                                drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=min(4, os.cpu_count()),
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=4,
                                                drop_last=True)
    
    return data_loader
    
    
class masked_l1(torch.nn.Module):
    def __init__(self):
        super(masked_l1, self).__init__()

    def forward(self, pred, gt, mask):
        return torch.mean(
            torch.abs(pred - gt) * mask.unsqueeze(1)
        )
           
        
class grad_loss(torch.nn.Module):
    def __init__(self):
        super(grad_loss, self).__init__()

    def forward(self, pred, gt, mask):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        mask_dx = mask[:, :, 1:] * mask[:, :, :-1]
        mask_dy = mask[:, 1:, :] * mask[:, :-1, :]
        loss_x = torch.mean(torch.abs(pred_dx - gt_dx) * mask_dx.unsqueeze(1))
        loss_y = torch.mean(torch.abs(pred_dy - gt_dy) * mask_dy.unsqueeze(1))
        return loss_x + loss_y


class vgg_loss(torch.nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features[:16]))
        self.features.eval()
        self.features.to(device)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, target, mask, mean, std):
        
        masked_pred = pred * mask.unsqueeze(1)
        masked_target = target * mask.unsqueeze(1)
        
        pred = (masked_pred - mean) / std
        target = (masked_target - mean) / std
        pred_feat = self.features(pred)

        with torch.no_grad():
            target_feat = self.features(target)

        return torch.mean(
            torch.abs(pred_feat - target_feat)
        )

class Trainer:
    def __init__(
            self,
            multiple_gpus: bool,
            save_every: int,
            model: torch.nn.Module,
            snapshot_path,
            results_path,
            snapshot_filename,
            train_data,
            optimizer,
            loss_func_type,
            device,
            ) -> None:
        
        self.multiple_gpus = multiple_gpus
        self.save_every = save_every

        self.device = device
        
        self.loss_func_type = loss_func_type
        self.grad_loss_function = grad_loss()
        self.maskedL1_loss_function = masked_l1()
        self.vgg_loss_function = vgg_loss().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)     

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        if self.multiple_gpus:
            self.model = DDP(
                model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False, # DDP does not synchronize model buffers (e.g. BatchNorm.running_mean and BatchNorm.running_var) from rank 0 to all other ranks every iteration
                gradient_as_bucket_view=True, # gradients become views into the communication buckets. This lower GPU memory usage and there is faster gradient reduction
                static_graph=True # If the computation graph never changes, then this option is preferred (less CPU overhead)
            )
        else:
            self.model = model.to(self.device)

        # self.scaler = torch.cuda.amp.GradScaler("cuda")
        self.scaler = torch.amp.GradScaler("cuda")
        
        self.model.train()
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.results_path = results_path
        self.snapshot_filename = snapshot_filename
        
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, exist_ok=True)
            print(f"Checkpoint folder '{self.snapshot_path}' created successfully.")

        if os.path.exists(os.path.join(self.snapshot_path, self.snapshot_filename)):
            print(f"Loading snapshot from {self.snapshot_path}")
            self.load_snapshot(os.path.join(self.snapshot_path, self.snapshot_filename))   

    def run_epoch(self, epoch: int):
        b_sz = self.train_data.batch_size
        print(f"\n\n[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if self.multiple_gpus:
            self.train_data.sampler.set_epoch(epoch)

        running_train_loss = torch.zeros(1, device=self.device)
        
        for idx, (source, targets) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            source = source.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            loss = self.run_batch(source, targets)
            running_train_loss += loss.detach()

        if self.multiple_gpus:
            saving_condition = (self.device == 0 and epoch % self.save_every == 0)
        else:
            saving_condition = (epoch % self.save_every == 0)
        if saving_condition:
            output_to_plot = self.output.to(torch.float32)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(source[0, :3, :, :].cpu().permute(1, 2, 0))
            axs[0].set_title("Input")
            axs[1].imshow(targets[0].detach().cpu().permute(1, 2, 0))
            axs[1].set_title("Ground Truth")
            axs[2].imshow(output_to_plot[0].detach().cpu().permute(1, 2, 0))
            axs[2].set_title("Output")
            plt.savefig(os.path.join(self.results_path, f"Epoch_{epoch}.png"))
            
        if self.multiple_gpus:
            torch.distributed.all_reduce(
                running_train_loss,
                op=torch.distributed.ReduceOp.SUM
            ) # perform SUM of the running_train_loss among all ranks

            running_train_loss /= (
                len(self.train_data) * # len(self.train_data) is the amount of batches each rank processed
                torch.distributed.get_world_size() # num ranks
            )
        else:
            running_train_loss /= len(self.train_data)
        epoch_loss = running_train_loss.cpu().item()
        print(f"Epoch: {epoch} | Training Loss: {epoch_loss:.6f}")

    def run_batch(self, source, targets):
        self.optimizer.zero_grad(set_to_none=True)
        rgb_corrupt = source[:, :3, :, :]
        text_mask = source[:, 3, :, :]
        holes_mask = source[:, 4, :, :]
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            self.output = self.model(rgb_corrupt)
        
        mask = text_mask * holes_mask # text_mask 1 for background and 0 for text and ornaments. holes_mask 1 for holes and 0 for non-holes.
        # So their product is 1 only for hole pixels that are background, which are the pixels we want to inpaint.
                
        l1_loss = self.maskedL1_loss_function(self.output, targets, mask)

        if self.loss_func_type.lower() == "vgg":
            vgg_loss = self.vgg_loss_function(self.output, targets, mask, self.mean, self.std)
            loss = l1_loss + 0.1 * vgg_loss
        elif self.loss_func_type.lower() == "l1":
            loss = l1_loss
        elif self.loss_func_type.lower() == "grad":
            loss = self.maskedL1_loss_function(self.output, targets, mask) + 0.1 * self.grad_loss_function(self.output, targets, mask)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss

    def save_snapshot(self, epoch: int, snapshot_path: str):
        snapshot = {}
        if self.multiple_gpus:
            snapshot["MODEL_STATE"] = self.model.module.state_dict() # model.module is the state dict of the model wrapped by DDP
        else:
            snapshot["MODEL_STATE"] = self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        PATH = os.path.join(snapshot_path, self.snapshot_filename)
        torch.save(snapshot, PATH)
        print(f"Epoch {epoch} | Training snapshot saved at {PATH}")

    def load_snapshot(self, snapshot_path: str):
        snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
        if self.multiple_gpus:
            self.model.module.load_state_dict(snapshot["MODEL_STATE"])
            self.model.module.to(self.device)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.model.to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Snapshot loaded from {snapshot_path}")

    def train(self, max_epochs: int, snapshot_path: str, scheduler):
        if os.path.exists(os.path.join(snapshot_path, self.snapshot_filename)):
            snapshot = torch.load(os.path.join(snapshot_path, self.snapshot_filename),map_location=torch.device('cpu'), weights_only=True)
            print(f"Restart training from epoch {snapshot['EPOCHS_RUN']}")
            
        for epoch in tqdm(range(self.epochs_run + 1, max_epochs +1), desc="Training the network"):
            self.run_epoch(epoch)

            if scheduler:
                scheduler.step()
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")

            # Save checkpoint from the rank 0 process
            if self.multiple_gpus:
                if self.device == 0 and epoch % self.save_every == 0:
                    self.save_snapshot(epoch, snapshot_path)
            else:
                if epoch % self.save_every == 0:
                    self.save_snapshot(epoch, snapshot_path)


# %%

if __name__ == "__main__":
    from UNet_model_inpainting import ResidualUNet
    import multiprocessing as mp
    
    mp.freeze_support()  # important on Windows

    patch_size = 1024

    multiple_gpus=False
    save_every=10
    batch_size=4
    base_dir = os.path.dirname(os.path.abspath(__file__))

    snapshot_path=os.path.join(base_dir, "snapshots")
    os.makedirs(snapshot_path, exist_ok=True)

    # dataset_path = os.path.join(base_dir, "dataset_MAGIC_PatchSize"+str(patch_size)+"_partial")
    dataset_path = os.path.join("/data1","aettari","dataset_MAGIC_PatchSize"+str(patch_size))

    if multiple_gpus:
        print("Using multiple GPUs")
        init_process_group(backend="nccl")
        device = int(os.environ["LOCAL_RANK"]) 
        torch.cuda.set_device(device)
    else:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("USING", device, " device")

    normalization="group"
    loss_func_type = "l1" # "vgg" or "l1" or "grad"

    snapshot_filename = f"snapshot_PathSize_{str(patch_size)}_Norm_{normalization}_Loss_{loss_func_type}.pt"
    results_path = os.path.join(base_dir, "results", f"Results_PatchSize_{str(patch_size)}_Norm_{normalization}_Loss_{loss_func_type}")
    os.makedirs(results_path, exist_ok=True)

    train_loader = prepare_data_loader(base_dir, dataset_path, multiple_gpus, batch_size)
    model=ResidualUNet(in_channels=3, out_channels=3, channels=(32, 64, 128, 256), normalization=normalization, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs=500

    trainer = Trainer(multiple_gpus, save_every, model, snapshot_path, results_path, snapshot_filename, train_loader, optimizer, loss_func_type, device)

    print(f"\nUsing {loss_func_type} loss function for training.")
    print(f"Using {normalization} normalization in the model.\n")

    trainer.train(num_epochs, snapshot_path, scheduler=None)

    if multiple_gpus:
        destroy_process_group()